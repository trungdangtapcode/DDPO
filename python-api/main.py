from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import io
import json
import asyncio
import random
import base64
import os
from dotenv import load_dotenv
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global device, clip_model, clip_processor
    print("Starting up FastAPI server...")
    
    # Set device
    device_name = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device configured: {device}")
    
    # Load CLIP image encoder for image-as-prompt feature
    print("\n" + "="*80)
    print("Loading CLIP image encoder (for image-as-prompt)...")
    print("="*80)
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = clip_model.to(device)
        clip_model.eval()  # Set to evaluation mode
        print("✓ CLIP image encoder loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Could not load CLIP encoder: {e}")
        print("  Image-as-prompt feature will be disabled")
    print("="*80 + "\n")
    
    # Load all 4 models at startup
    print("\n" + "="*80)
    print("Loading all 4 DDPO models into VRAM...")
    print("="*80)
    
    for model_key in AVAILABLE_MODELS.keys():
        load_pipeline(model_key)
        print()
    
    print("="*80)
    print(f"✓ All {len(pipelines)}/4 models loaded successfully")
    print(f"✓ Models ready: {list(pipelines.keys())}")
    print("="*80 + "\n")
    print("✓ API ready - models are in VRAM and ready for inference")
    
    yield
    
    # Shutdown
    unload_pipeline()
    print("Server shutting down...")

app = FastAPI(lifespan=lifespan)

# Global pipelines - support multiple models
pipelines = {}
device = None
pipeline_lock = asyncio.Lock()  # Prevent concurrent loading

# CLIP image encoder for image-as-prompt
clip_model = None
clip_processor = None

# Available models
AVAILABLE_MODELS = {
    "aesthetic": "kvablack/ddpo-aesthetic",
    "alignment": "kvablack/ddpo-alignment",
    "compressibility": "kvablack/ddpo-compressibility",
    "incompressibility": "kvablack/ddpo-incompressibility"
}

# Enable CORS
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for POST endpoint
class GenerateRequest(BaseModel):
    prompt: str = ""  # Make prompt optional when using prompt_image
    steps: int = 20
    model: str = "compressibility"
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    start_step: int = 0
    start_image: Optional[str] = None
    noise_mask_type: str = "none"
    noise_strength: float = 1.0
    inject_at_step: int = -1
    init_image: Optional[str] = None
    strength: float = 0.75
    prompt_image: Optional[str] = None  # Base64 image to use as prompt instead of text

def latent_to_image(vae, latent):
    """Convert latent tensor to PIL Image."""
    with torch.no_grad():
        # Decode the latent
        image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        if image.shape[0] == 1:
            image = image[0]
        return Image.fromarray(image)

def create_noise_mask(mask_type: str, height: int = 64, width: int = 64, device=None):
    """
    Create a noise mask in latent space (64x64).
    
    Args:
        mask_type: Type of mask pattern
        height: Latent height (default 64 for 512px images)
        width: Latent width (default 64 for 512px images)
        device: Torch device
    
    Returns:
        Tensor of shape (1, 1, height, width) with values 0-1
    """
    import math
    
    mask = torch.zeros((1, 1, height, width), device=device)
    
    if mask_type == "center_circle":
        # Circle in the center
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 4
        for y in range(height):
            for x in range(width):
                dist = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                if dist <= radius:
                    mask[0, 0, y, x] = 1.0
    
    elif mask_type == "center_square":
        # Square in the center
        h_start, h_end = height // 4, 3 * height // 4
        w_start, w_end = width // 4, 3 * width // 4
        mask[0, 0, h_start:h_end, w_start:w_end] = 1.0
    
    elif mask_type == "edges":
        # Border/edges
        border_size = height // 8
        mask[0, 0, :border_size, :] = 1.0  # Top
        mask[0, 0, -border_size:, :] = 1.0  # Bottom
        mask[0, 0, :, :border_size] = 1.0  # Left
        mask[0, 0, :, -border_size:] = 1.0  # Right
    
    elif mask_type == "corners":
        # Four corners
        corner_size = height // 4
        mask[0, 0, :corner_size, :corner_size] = 1.0  # Top-left
        mask[0, 0, :corner_size, -corner_size:] = 1.0  # Top-right
        mask[0, 0, -corner_size:, :corner_size] = 1.0  # Bottom-left
        mask[0, 0, -corner_size:, -corner_size:] = 1.0  # Bottom-right
    
    elif mask_type == "left_half":
        # Left half
        mask[0, 0, :, :width//2] = 1.0
    
    elif mask_type == "right_half":
        # Right half
        mask[0, 0, :, width//2:] = 1.0
    
    elif mask_type == "top_half":
        # Top half
        mask[0, 0, :height//2, :] = 1.0
    
    elif mask_type == "bottom_half":
        # Bottom half
        mask[0, 0, height//2:, :] = 1.0
    
    elif mask_type == "checkerboard":
        # Checkerboard pattern
        for y in range(height):
            for x in range(width):
                if (y // 8 + x // 8) % 2 == 0:
                    mask[0, 0, y, x] = 1.0
    
    # "none" or any other value returns zero mask (no noise injection)
    
    return mask

def image_to_clip_embedding(image_b64: str) -> Optional[torch.Tensor]:
    """
    Convert a base64 image to CLIP image embedding for use as prompt.
    
    Args:
        image_b64: Base64 encoded image
    
    Returns:
        Tensor of shape [1, 77, 768] compatible with SD text encoder output
        Returns None if CLIP model not loaded or error occurs
    """
    global clip_model, clip_processor, device
    
    if clip_model is None or clip_processor is None:
        print("❌ CLIP model not loaded - image-as-prompt feature unavailable")
        return None
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Process image with CLIP
        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract CLIP image features
        with torch.no_grad():
            clip_features = clip_model.get_image_features(**inputs)  # Shape: [1, 768]
        
        # Expand to match SD text encoder output shape: [1, 77, 768]
        # Repeat the image embedding across all 77 token positions
        image_prompt_embed = clip_features.unsqueeze(1).repeat(1, 77, 1)
        
        print(f"✓ CLIP image embedding extracted: {image_prompt_embed.shape}")
        return image_prompt_embed
        
    except Exception as e:
        print(f"❌ Error extracting CLIP embedding: {e}")
        return None

def load_pipeline(model_key: str = "compressibility"):
    """Load a specific Stable Diffusion pipeline."""
    global pipelines, device
    
    # Check if model key is valid
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Invalid model key: {model_key}")
        return None
    
    # Return if already loaded
    if model_key in pipelines:
        print(f"✓ Model '{model_key}' already loaded")
        return pipelines[model_key]
    
    model_name = AVAILABLE_MODELS[model_key]
    device_name = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    hf_token = os.getenv("HF_TOKEN")
    
    print(f"Loading model '{model_key}': {model_name}")
    print(f"Device: {device_name}")
    
    if device is None:
        device = torch.device(device_name)
    
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            cache_dir=os.getenv("HF_CACHE_DIR", None),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_auth_token=hf_token,
            local_files_only=False,
            low_cpu_mem_usage=True,  # Enable low CPU memory usage
        )
        
        # Set scheduler to DDIM
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        # Disable safety checker for faster inference
        pipeline.safety_checker = None
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations for CUDA
        if device.type == "cuda":
            pipeline.enable_attention_slicing(1)  # Slice attention to reduce memory
            pipeline.enable_vae_slicing()  # Slice VAE decoding
            
            # Clear cache before generation
            torch.cuda.empty_cache()
        
        # Store in pipelines dict
        pipelines[model_key] = pipeline
        print(f"✓ Model '{model_key}' loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Failed to load model '{model_key}': {e}")
        return None
    
    return pipeline

def unload_pipeline():
    """Unload all pipelines to free GPU memory."""
    global pipelines
    if pipelines:
        for key in list(pipelines.keys()):
            del pipelines[key]
        pipelines = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ All pipelines unloaded, memory freed")

def create_mock_image(step: int, total_steps: int, prompt: str, width: int = 512, height: int = 512):
    """
    Create a mock image that simulates diffusion process
    Early steps: more noise/blur
    Later steps: clearer image with text
    """
    # Create base image with gradient
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Progress from noisy to clear
    progress = step / total_steps
    
    # Add colored gradient based on progress
    for y in range(height):
        # Color transitions from gray (noisy) to colorful (clear)
        r = int(100 + (155 * progress))
        g = int(120 + (100 * progress * (y / height)))
        b = int(150 + (105 * progress * (1 - y / height)))
        
        # Add noise for early steps
        if progress < 0.7:
            noise_factor = int((1 - progress) * 50)
            r += random.randint(-noise_factor, noise_factor)
            g += random.randint(-noise_factor, noise_factor)
            b += random.randint(-noise_factor, noise_factor)
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add geometric shapes to simulate "content"
    if progress > 0.3:
        # Draw some shapes that become clearer
        opacity = int(progress * 255)
        
        # Circle
        circle_size = int(progress * 100)
        circle_pos = (width // 2 - circle_size // 2, height // 2 - circle_size // 2)
        draw.ellipse(
            [circle_pos, (circle_pos[0] + circle_size, circle_pos[1] + circle_size)],
            outline=(255, 200, 0),
            width=3
        )
        
        # Rectangles
        if progress > 0.5:
            draw.rectangle(
                [50, 50, 150, 150],
                outline=(0, 200, 255),
                width=2
            )
    
    # Add text overlay showing progress and prompt
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add prompt text (becomes clearer over time)
    if progress > 0.4:
        text_alpha = int((progress - 0.4) / 0.6 * 255)
        prompt_display = prompt[:30] + "..." if len(prompt) > 30 else prompt
        draw.text((20, height - 80), f"Prompt: {prompt_display}", fill=(0, 0, 0), font=font)
    
    # Always show step counter
    draw.text((20, height - 40), f"Step {step}/{total_steps}", fill=(0, 0, 0), font=font)
    
    return img

async def generate_image_stream(
    prompt: str, 
    num_inference_steps: int, 
    model_name: str,
    guidance_scale: float = 5.0,
    negative_prompt: str = "",
    start_step: int = 0,
    start_image_b64: Optional[str] = None,
    noise_mask_type: str = "none",
    noise_strength: float = 1.0,
    inject_at_step: int = -1,
    init_image_b64: Optional[str] = None,
    strength: float = 0.75,
    prompt_image_b64: Optional[str] = None
):
    """
    Generator that yields intermediate diffusion steps using real Stable Diffusion
    All models are pre-loaded, so this just performs denoising.
    
    Args:
        start_step: Skip to this step (0 = start from noise, >0 = start from partial denoising)
        start_image_b64: Optional base64 encoded image to start from (will be encoded to latent)
        noise_mask_type: Type of noise mask pattern ('none', 'center_circle', 'center_square', etc.)
        noise_strength: Strength of noise injection (0.0 to 1.0)
        inject_at_step: Step number to inject noise (-1 = no injection)
        init_image_b64: Optional base64 image for img2img mode
        strength: How much to transform init_image (0.0=no change, 1.0=complete transformation)
    """
    global pipelines, device
    
    import time
    start_time = time.time()
    
    # Alias for convenience
    steps = num_inference_steps
    
    # Validate model
    if model_name not in AVAILABLE_MODELS:
        model_name = "compressibility"  # Default fallback
    
    # Get pre-loaded pipeline (should already be in VRAM)
    pipeline = pipelines.get(model_name)
    if pipeline is None:
        # This shouldn't happen if startup loaded correctly, but handle it
        print(f"⚠️ Warning: Model '{model_name}' not pre-loaded, loading now...")
        async with pipeline_lock:
            pipeline = load_pipeline(model_name)
    
    print(f"[TIMING] Pipeline retrieved: {time.time() - start_time:.3f}s")
    
    # Use real pipeline if available, otherwise fall back to mock
    use_real_model = pipeline is not None
    
    if use_real_model:
        # Clear CUDA cache before generation
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"[TIMING] CUDA cache cleared: {time.time() - start_time:.3f}s")
        
        # Real Stable Diffusion generation
        height = 512
        width = 512
        
        # Encode prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        
        with torch.no_grad():  # Disable gradient computation to save memory
            # Check if we're using image as prompt
            if prompt_image_b64:
                print("[IMAGE-AS-PROMPT] Using CLIP image embeddings instead of text prompt")
                prompt_embeds = image_to_clip_embedding(prompt_image_b64)
                
                if prompt_embeds is None:
                    print("❌ Failed to extract CLIP embeddings - falling back to text prompt")
                    # Fall back to text encoding
                    text_inputs = pipeline.tokenizer(
                        prompt if prompt else "a photograph",
                        padding="max_length",
                        max_length=pipeline.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids.to(device)
                    prompt_embeds = pipeline.text_encoder(text_input_ids)[0]
                else:
                    # Convert dtype to match pipeline
                    prompt_embeds = prompt_embeds.to(dtype=pipeline.text_encoder.dtype)
                    print(f"[TIMING] CLIP image embedding extracted: {time.time() - start_time:.3f}s")
            else:
                # Standard text encoding
                text_inputs = pipeline.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)  # Move to GPU immediately
                
                print(f"[TIMING] Prompt tokenized: {time.time() - start_time:.3f}s")
                
                prompt_embeds = pipeline.text_encoder(text_input_ids)[0]
                
                print(f"[TIMING] Prompt encoded: {time.time() - start_time:.3f}s")
            
            # Encode negative prompt for classifier-free guidance
            if do_classifier_free_guidance:
                # Use provided negative prompt or default to empty
                neg_prompt = negative_prompt if negative_prompt else ""
                uncond_tokens = pipeline.tokenizer(
                    neg_prompt,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_input_ids = uncond_tokens.input_ids.to(device)  # Move to GPU immediately
                negative_prompt_embeds = pipeline.text_encoder(uncond_input_ids)[0]
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
            print(f"[TIMING] Negative prompt encoded: {time.time() - start_time:.3f}s")
            
            # Prepare timesteps
            pipeline.scheduler.set_timesteps(steps, device=device)
            timesteps = pipeline.scheduler.timesteps
            
            print(f"[TIMING] Scheduler prepared: {time.time() - start_time:.3f}s")
            
            # Prepare latent variables
            num_channels_latents = pipeline.unet.config.in_channels
            batch_size = 1
            shape = (batch_size, num_channels_latents, height // 8, width // 8)
            
            # Check if we're using img2img mode
            if init_image_b64:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(init_image_b64)
                    from PIL import Image
                    import numpy as np
                    
                    init_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    init_image = init_image.resize((width, height))
                    
                    # Convert PIL image to tensor
                    image_np = np.array(init_image).astype(np.float32) / 255.0
                    image_np = image_np[None].transpose(0, 3, 1, 2)  # HWC to BCHW
                    image_tensor = torch.from_numpy(image_np).to(device, dtype=prompt_embeds.dtype)
                    
                    # Normalize to [-1, 1]
                    image_tensor = 2.0 * image_tensor - 1.0
                    
                    # Encode to latent space
                    with torch.no_grad():
                        latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
                        latents = latents * pipeline.vae.config.scaling_factor
                    
                    # Calculate how many steps to denoise based on strength
                    # strength=0.0 means no change (start at last step)
                    # strength=1.0 means complete transformation (start from first step)
                    init_timestep = int(steps * strength)
                    init_timestep = min(init_timestep, steps)
                    
                    # Get the timestep to start from
                    t_start = max(steps - init_timestep, 0)
                    timestep = timesteps[t_start]
                    
                    # Add noise to the latents
                    noise = torch.randn_like(latents)
                    latents = pipeline.scheduler.add_noise(latents, noise, timestep)
                    
                    # Set start_step for the loop
                    start_step = t_start
                    
                    print(f"[INFO] IMG2IMG mode: strength={strength}, starting from step {start_step}/{steps}, timestep {timestep}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to process init image: {e}. Starting from random noise.")
                    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)
                    latents = latents * pipeline.scheduler.init_noise_sigma
                    start_step = 0
            
            # Check if we're starting from a custom image (noise interaction mode)
            elif start_image_b64 and start_step > 0:
                try:
                    # Decode base64 image
                    image_bytes = base64.b64decode(start_image_b64)
                    from PIL import Image
                    import numpy as np
                    
                    start_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    start_image = start_image.resize((width, height))
                    
                    # Convert PIL image to tensor
                    image_np = np.array(start_image).astype(np.float32) / 255.0
                    image_np = image_np[None].transpose(0, 3, 1, 2)  # HWC to BCHW
                    image_tensor = torch.from_numpy(image_np).to(device, dtype=prompt_embeds.dtype)
                    
                    # Normalize to [-1, 1]
                    image_tensor = 2.0 * image_tensor - 1.0
                    
                    # Encode to latent space
                    with torch.no_grad():
                        latents = pipeline.vae.encode(image_tensor).latent_dist.sample()
                        latents = latents * pipeline.vae.config.scaling_factor
                    
                    # Add noise to match the start_step
                    # Skip to the specified timestep by adding appropriate noise
                    if start_step < len(timesteps):
                        noise = torch.randn_like(latents)
                        timestep = timesteps[start_step]
                        latents = pipeline.scheduler.add_noise(latents, noise, timestep)
                        print(f"[INFO] Starting from custom image at step {start_step}, timestep {timestep}")
                    else:
                        print(f"[WARNING] start_step {start_step} >= total steps {len(timesteps)}, starting from pure noise")
                        latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)
                        latents = latents * pipeline.scheduler.init_noise_sigma
                        start_step = 0
                        
                except Exception as e:
                    print(f"[ERROR] Failed to process start image: {e}. Starting from random noise.")
                    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)
                    latents = latents * pipeline.scheduler.init_noise_sigma
                    start_step = 0
            else:
                # Start from random noise
                latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)
                latents = latents * pipeline.scheduler.init_noise_sigma
                start_step = 0
            
            print(f"[TIMING] Latents initialized: {time.time() - start_time:.3f}s")
            print(f"[INFO] Starting denoising loop with {steps} steps (starting from step {start_step})...")
            
            # Create noise mask if needed
            noise_mask = None
            if noise_mask_type != "none" and inject_at_step >= 0 and inject_at_step < steps:
                noise_mask = create_noise_mask(noise_mask_type, height=shape[2], width=shape[3], device=device)
                print(f"[INFO] Noise mask '{noise_mask_type}' created, will inject at step {inject_at_step} with strength {noise_strength}")
            
            # Denoising loop - start from start_step
            for step_idx, t in enumerate(timesteps[start_step:], start=start_step):
                # Inject noise at specified step
                if noise_mask is not None and step_idx == inject_at_step:
                    # Generate random noise with matching dtype
                    injected_noise = torch.randn_like(latents) * noise_strength
                    # Ensure mask has the same dtype as latents
                    noise_mask_typed = noise_mask.to(dtype=latents.dtype)
                    # Apply mask: keep original latents where mask is 0, add noise where mask is 1
                    latents = latents * (1 - noise_mask_typed) + (latents + injected_noise) * noise_mask_typed
                    print(f"[INFO] Injected noise at step {step_idx} using mask '{noise_mask_type}'")
                
                # Expand latents if doing classifier-free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise residual
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                # Send every step to show real-time progress
                should_send = True
                
                if should_send:
                    step_start = time.time()
                    image = latent_to_image(pipeline.vae, latents)
                    
                    if step_idx == 0:
                        print(f"[TIMING] First image decoded: {time.time() - start_time:.3f}s")
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=85 if step_idx < steps - 1 else 95)
                    image_bytes = buffer.getvalue()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Yield as Server-Sent Event
                    data = {
                        'step': step_idx + 1,
                        'total_steps': steps,
                        'image': image_base64,
                        'progress': ((step_idx + 1) / steps) * 100,
                        'done': step_idx == steps - 1,
                        'model': f'stable-diffusion-{model_name}'
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    if step_idx == 0:
                        print(f"[TIMING] First image sent to client: {time.time() - start_time:.3f}s")
                        print(f"[TIMING] First image encoding took: {time.time() - step_start:.3f}s")
                    
                    # Small delay to prevent overwhelming the client (reduced for first image)
                    await asyncio.sleep(0.01 if step_idx == 0 else 0.02)
        
        # Clear CUDA cache after generation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    else:
        # Fall back to mock generation
        for step in range(1, steps + 1):
            # Simulate processing time
            await asyncio.sleep(0.3)
            
            # Generate mock image for this step
            image = create_mock_image(step, steps, prompt)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85 if step < steps else 95)
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Yield as Server-Sent Event
            data = {
                'step': step,
                'total_steps': steps,
                'image': image_base64,
                'progress': (step / steps) * 100,
                'done': step == steps,
                'model': 'mock'
            }
            
            yield f"data: {json.dumps(data)}\n\n"
    
    # Send completion signal
    yield "data: [DONE]\n\n"

@app.get("/")
async def root():
    loaded_models = list(pipelines.keys())
    return {
        "message": "Diffusion API",
        "status": "running",
        "models_loaded": loaded_models,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "device": str(device) if device else "N/A"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate image with streaming intermediate steps (POST version for large payloads)
    
    Body:
        JSON object with all generation parameters including base64 images
    """
    # Extract parameters from request
    prompt = request.prompt
    steps = request.steps
    model = request.model
    guidance_scale = request.guidance_scale
    negative_prompt = request.negative_prompt
    start_step = request.start_step
    start_image = request.start_image
    noise_mask_type = request.noise_mask_type
    noise_strength = request.noise_strength
    inject_at_step = request.inject_at_step
    init_image = request.init_image
    strength = request.strength
    prompt_image = request.prompt_image  # Image to use as prompt
    
    # Validate parameters
    # Allow empty prompt if prompt_image is provided
    if not prompt.strip() and not prompt_image:
        return {"error": "Either prompt or prompt_image is required"}
    
    if steps < 5 or steps > 50:
        steps = 20
    
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        guidance_scale = 7.5
    
    if start_step < 0 or start_step >= steps:
        start_step = 0
    
    # Validate noise parameters
    if noise_strength < 0.0 or noise_strength > 2.0:
        noise_strength = 1.0
    
    if inject_at_step < -1 or inject_at_step >= steps:
        inject_at_step = -1
    
    # Validate img2img parameters
    if strength < 0.0 or strength > 1.0:
        strength = 0.75
    
    # Validate model
    if model not in AVAILABLE_MODELS:
        return {"error": f"Invalid model. Choose from: {list(AVAILABLE_MODELS.keys())}"}
    
    # Ensure we have a prompt string (use placeholder if using image-as-prompt)
    if not prompt.strip() and prompt_image:
        prompt = "image-guided generation"
    
    return StreamingResponse(
        generate_image_stream(
            prompt, steps, model, guidance_scale, negative_prompt, 
            start_step, start_image, noise_mask_type, noise_strength, inject_at_step,
            init_image, strength, prompt_image
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health():
    loaded_models = list(pipelines.keys())
    health_info = {
        "status": "healthy",
        "models_loaded": loaded_models,
        "total_models": len(loaded_models),
        "device": str(device) if device else "N/A"
    }
    
    # Add GPU memory info if using CUDA
    if device and device.type == "cuda":
        try:
            health_info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(device) / 1024**3, 2)
            health_info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(device) / 1024**3, 2)
            health_info["gpu_name"] = torch.cuda.get_device_name(device)
        except:
            pass
    
    return health_info

@app.get("/diagnostics")
async def diagnostics():
    """Detailed diagnostic information"""
    diag_info = {
        "device": str(device) if device else "N/A",
        "models_loaded": list(pipelines.keys()),
        "available_models": list(AVAILABLE_MODELS.keys()),
    }
    
    # GPU information
    if device and device.type == "cuda":
        try:
            diag_info["cuda_available"] = torch.cuda.is_available()
            diag_info["cuda_device_count"] = torch.cuda.device_count()
            diag_info["cuda_current_device"] = torch.cuda.current_device()
            diag_info["gpu_name"] = torch.cuda.get_device_name(device)
            diag_info["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(device) / 1024**3, 2)
            diag_info["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(device) / 1024**3, 2)
            diag_info["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2)
        except Exception as e:
            diag_info["gpu_error"] = str(e)
    
    # Check model components location
    for model_key, pipeline in pipelines.items():
        try:
            diag_info[f"model_{model_key}_unet_device"] = str(next(pipeline.unet.parameters()).device)
            diag_info[f"model_{model_key}_text_encoder_device"] = str(next(pipeline.text_encoder.parameters()).device)
            diag_info[f"model_{model_key}_vae_device"] = str(next(pipeline.vae.parameters()).device)
        except:
            diag_info[f"model_{model_key}_error"] = "Could not determine device"
    
    return diag_info

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)





# @app.on_event("startup")
# async def startup_event():
#     """Load all 4 models at startup."""
#     global device
#     print("Starting up FastAPI server...")
    
#     # Set device
#     device_name = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
#     device = torch.device(device_name)
#     print(f"Device configured: {device}")
    
#     # Load all 4 models
#     load_all_pipelines()
    
#     print(f"✓ API ready with {len(pipelines)} models loaded")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Clean up on shutdown."""
#     unload_pipeline()
#     print("Server shutting down...")