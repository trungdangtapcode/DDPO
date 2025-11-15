from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global device
    print("Starting up FastAPI server...")
    
    # Set device
    device_name = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"Device configured: {device}")
    
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

# Available models
AVAILABLE_MODELS = {
    "aesthetic": "kvablack/ddpo-aesthetic",
    "alignment": "kvablack/ddpo-alignment",
    "compressibility": "kvablack/ddpo-compressibility",
    "incompressibility": "kvablack/ddpo-incompressibility"
}

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

async def generate_image_stream(prompt: str, steps: int = 20, model: str = "compressibility", guidance_scale: float = 7.5, negative_prompt: str = ""):
    """
    Generator that yields intermediate diffusion steps using real Stable Diffusion
    All models are pre-loaded, so this just performs denoising.
    """
    global pipelines, device
    
    import time
    start_time = time.time()
    
    # Validate model
    if model not in AVAILABLE_MODELS:
        model = "compressibility"  # Default fallback
    
    # Get pre-loaded pipeline (should already be in VRAM)
    pipeline = pipelines.get(model)
    if pipeline is None:
        # This shouldn't happen if startup loaded correctly, but handle it
        print(f"⚠️ Warning: Model '{model}' not pre-loaded, loading now...")
        async with pipeline_lock:
            pipeline = load_pipeline(model)
    
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
            # Tokenize and encode prompt (make sure it's on GPU)
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
            latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)
            latents = latents * pipeline.scheduler.init_noise_sigma
            
            print(f"[TIMING] Latents initialized: {time.time() - start_time:.3f}s")
            print(f"[INFO] Starting denoising loop with {steps} steps...")
            
            # Denoising loop
            for step_idx, t in enumerate(timesteps):
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
                
                # Send first step immediately to show progress started
                # Then send every 3rd step, and always send the last step
                should_send = (step_idx == 0) or (step_idx % 3 == 0) or (step_idx == steps - 1)
                
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
                        'model': f'stable-diffusion-{model}'
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

@app.get("/generate")
async def generate(
    prompt: str, 
    steps: int = 20, 
    model: str = "compressibility",
    guidance_scale: float = 7.5,
    negative_prompt: str = ""
):
    """
    Generate image with streaming intermediate steps
    """
    if not prompt:
        return {"error": "Prompt is required"}
    
    if steps < 5 or steps > 50:
        steps = 20
    
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        guidance_scale = 7.5
    
    # Validate model
    if model not in AVAILABLE_MODELS:
        return {"error": f"Invalid model. Choose from: {list(AVAILABLE_MODELS.keys())}"}
    
    return StreamingResponse(
        generate_image_stream(prompt, steps, model, guidance_scale, negative_prompt),
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