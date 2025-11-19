"""
Image generation service with streaming support.
"""
import time
import asyncio
import base64
import io
import json
from typing import Optional
import torch
from PIL import Image

from app.core.config import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    LATENT_HEIGHT, LATENT_WIDTH
)
from app.utils.image_utils import latent_to_image, pil_to_latent
from app.utils.noise_mask import create_noise_mask
from app.services.clip_service import clip_service
from app.services.pipeline_service import pipeline_service


class GenerationService:
    """Service for image generation with streaming."""
    
    @staticmethod
    async def generate_stream(
        prompt: str,
        num_inference_steps: int,
        model_name: str,
        guidance_scale: float = 7.5,
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
        Generate images with streaming intermediate steps.
        
        Yields Server-Sent Events with step-by-step generation progress.
        """
        start_time = time.time()
        steps = num_inference_steps
        
        # Get pipeline
        pipeline = pipeline_service.get_pipeline(model_name)
        if pipeline is None:
            print(f"⚠️ Warning: Model '{model_name}' not loaded")
            yield "data: {\"error\": \"Model not loaded\"}\n\n"
            return
        
        device = pipeline_service.device
        
        print(f"[TIMING] Pipeline retrieved: {time.time() - start_time:.3f}s")
        
        # Clear CUDA cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Encode prompt or use CLIP image embedding
        do_classifier_free_guidance = guidance_scale > 1.0
        
        with torch.no_grad():
            prompt_embeds = await GenerationService._encode_prompt(
                pipeline, prompt, prompt_image_b64, device, start_time
            )
            
            if prompt_embeds is None:
                yield "data: {\"error\": \"Failed to encode prompt\"}\n\n"
                return
            
            # Encode negative prompt for CFG
            if do_classifier_free_guidance:
                negative_prompt_embeds = await GenerationService._encode_negative_prompt(
                    pipeline, negative_prompt, device
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
            print(f"[TIMING] Prompts encoded: {time.time() - start_time:.3f}s")
            
            # Prepare timesteps
            pipeline.scheduler.set_timesteps(steps, device=device)
            timesteps = pipeline.scheduler.timesteps
            
            # Initialize latents
            latents, start_step = await GenerationService._initialize_latents(
                pipeline, init_image_b64, start_image_b64, start_step,
                steps, strength, prompt_embeds.dtype, device, start_time
            )
            
            print(f"[INFO] Starting denoising from step {start_step}/{steps}")
            
            # Create noise mask if needed
            noise_mask = None
            if noise_mask_type != "none" and 0 <= inject_at_step < steps:
                noise_mask = create_noise_mask(
                    noise_mask_type,
                    height=LATENT_HEIGHT,
                    width=LATENT_WIDTH,
                    device=device
                )
                print(f"[INFO] Noise mask '{noise_mask_type}' ready for step {inject_at_step}")
            
            # Denoising loop
            async for data in GenerationService._denoise_loop(
                pipeline, latents, timesteps, prompt_embeds,
                guidance_scale, do_classifier_free_guidance,
                start_step, steps, noise_mask, noise_mask_type,
                inject_at_step, noise_strength, model_name, start_time
            ):
                yield f"data: {json.dumps(data)}\n\n"
        
        # Clear cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        yield "data: [DONE]\n\n"
    
    @staticmethod
    async def _encode_prompt(pipeline, prompt, prompt_image_b64, device, start_time):
        """Encode text prompt or extract CLIP image embeddings."""
        if prompt_image_b64:
            print("[IMAGE-AS-PROMPT] Using CLIP image embeddings")
            prompt_embeds = clip_service.image_to_embedding(prompt_image_b64)
            
            if prompt_embeds is None:
                print("❌ Failed to extract CLIP embeddings - falling back to text")
                prompt = prompt if prompt else "a photograph"
            else:
                prompt_embeds = prompt_embeds.to(dtype=pipeline.text_encoder.dtype)
                print(f"[TIMING] CLIP embedding: {time.time() - start_time:.3f}s")
                return prompt_embeds
        
        # Standard text encoding
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = pipeline.text_encoder(text_input_ids)[0]
        
        return prompt_embeds
    
    @staticmethod
    async def _encode_negative_prompt(pipeline, negative_prompt, device):
        """Encode negative prompt for classifier-free guidance."""
        uncond_tokens = pipeline.tokenizer(
            negative_prompt if negative_prompt else "",
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_tokens.input_ids.to(device)
        return pipeline.text_encoder(uncond_input_ids)[0]
    
    @staticmethod
    async def _initialize_latents(
        pipeline, init_image_b64, start_image_b64, start_step,
        steps, strength, dtype, device, start_time
    ):
        """Initialize latents based on mode (txt2img, img2img, or custom start)."""
        num_channels = pipeline.unet.config.in_channels
        shape = (1, num_channels, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8)
        
        # IMG2IMG mode takes priority
        if init_image_b64:
            return await GenerationService._init_img2img_latents(
                pipeline, init_image_b64, steps, strength, dtype, device, start_time
            )
        
        # Custom start image (noise interaction)
        if start_image_b64 and start_step > 0:
            return await GenerationService._init_custom_start_latents(
                pipeline, start_image_b64, start_step, steps, dtype, device
            )
        
        # Random noise (txt2img)
        latents = torch.randn(shape, device=device, dtype=dtype)
        latents = latents * pipeline.scheduler.init_noise_sigma
        return latents, 0
    
    @staticmethod
    async def _init_img2img_latents(pipeline, image_b64, steps, strength, dtype, device, start_time):
        """Initialize latents for img2img mode."""
        try:
            image_bytes = base64.b64decode(image_b64)
            init_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            latents = pil_to_latent(pipeline.vae, init_image, device, dtype)
            
            # Calculate timestep based on strength
            init_timestep = int(steps * strength)
            t_start = max(steps - init_timestep, 0)
            timestep = pipeline.scheduler.timesteps[t_start]
            
            # Add noise
            noise = torch.randn_like(latents)
            latents = pipeline.scheduler.add_noise(latents, noise, timestep)
            
            print(f"[INFO] IMG2IMG: strength={strength}, start={t_start}/{steps}")
            return latents, t_start
            
        except Exception as e:
            print(f"[ERROR] Failed to process init image: {e}")
            shape = (1, pipeline.unet.config.in_channels, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8)
            latents = torch.randn(shape, device=device, dtype=dtype)
            return latents * pipeline.scheduler.init_noise_sigma, 0
    
    @staticmethod
    async def _init_custom_start_latents(pipeline, image_b64, start_step, steps, dtype, device):
        """Initialize latents from custom start image."""
        try:
            image_bytes = base64.b64decode(image_b64)
            start_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            latents = pil_to_latent(pipeline.vae, start_image, device, dtype)
            
            if start_step < steps:
                noise = torch.randn_like(latents)
                timestep = pipeline.scheduler.timesteps[start_step]
                latents = pipeline.scheduler.add_noise(latents, noise, timestep)
                print(f"[INFO] Custom start at step {start_step}")
                return latents, start_step
            
        except Exception as e:
            print(f"[ERROR] Failed to process start image: {e}")
        
        # Fallback
        shape = (1, pipeline.unet.config.in_channels, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8)
        latents = torch.randn(shape, device=device, dtype=dtype)
        return latents * pipeline.scheduler.init_noise_sigma, 0
    
    @staticmethod
    async def _denoise_loop(
        pipeline, latents, timesteps, prompt_embeds,
        guidance_scale, do_classifier_free_guidance,
        start_step, total_steps, noise_mask, noise_mask_type,
        inject_at_step, noise_strength, model_name, start_time
    ):
        """Main denoising loop with noise injection and streaming."""
        for step_idx, t in enumerate(timesteps[start_step:], start=start_step):
            # Inject noise if configured
            if noise_mask is not None and step_idx == inject_at_step:
                injected_noise = torch.randn_like(latents) * noise_strength
                noise_mask_typed = noise_mask.to(dtype=latents.dtype)
                latents = latents * (1 - noise_mask_typed) + (latents + injected_noise) * noise_mask_typed
                print(f"[INFO] Noise injected at step {step_idx}: {noise_mask_type}")
            
            # Expand for CFG
            latent_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_input = pipeline.scheduler.scale_model_input(latent_input, t)
            
            # Predict noise
            noise_pred = pipeline.unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            
            # Apply CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Decode and yield
            image = latent_to_image(pipeline.vae, latents)
            
            # Convert to base64
            buffer = io.BytesIO()
            quality = 95 if step_idx == total_steps - 1 else 85
            image.save(buffer, format='JPEG', quality=quality)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if step_idx == 0:
                print(f"[TIMING] First image sent: {time.time() - start_time:.3f}s")
            
            yield {
                'step': step_idx + 1,
                'total_steps': total_steps,
                'image': image_b64,
                'progress': ((step_idx + 1) / total_steps) * 100,
                'done': step_idx == total_steps - 1,
                'model': f'stable-diffusion-{model_name}'
            }
            
            await asyncio.sleep(0.01 if step_idx == 0 else 0.02)


# Global generation service
generation_service = GenerationService()
