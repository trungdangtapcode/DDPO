"""
API route handlers.
"""
import torch
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import GenerateRequest
from app.services.generation_service import generation_service
from app.services.pipeline_service import pipeline_service
from app.core.config import (
    AVAILABLE_MODELS,
    MIN_STEPS, MAX_STEPS, DEFAULT_STEPS,
    MIN_GUIDANCE_SCALE, MAX_GUIDANCE_SCALE, DEFAULT_GUIDANCE_SCALE,
    MIN_STRENGTH, MAX_STRENGTH, DEFAULT_STRENGTH,
    MIN_NOISE_STRENGTH, MAX_NOISE_STRENGTH, DEFAULT_NOISE_STRENGTH
)

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with API information."""
    loaded_models = pipeline_service.get_loaded_models()
    device = pipeline_service.device
    
    return {
        "message": "Diffusion API",
        "status": "running",
        "models_loaded": loaded_models,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "device": str(device) if device else "N/A"
    }


@router.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate image with streaming intermediate steps.
    
    Accepts JSON body with all generation parameters including base64 images.
    Returns Server-Sent Events stream with step-by-step progress.
    """
    # Extract and validate parameters
    prompt = request.prompt
    steps = max(MIN_STEPS, min(request.steps, MAX_STEPS)) or DEFAULT_STEPS
    model = request.model if request.model in AVAILABLE_MODELS else "compressibility"
    guidance_scale = max(MIN_GUIDANCE_SCALE, min(request.guidance_scale, MAX_GUIDANCE_SCALE))
    negative_prompt = request.negative_prompt
    start_step = max(0, min(request.start_step, steps - 1))
    print("START STEP:", start_step)
    start_image = request.start_image
    noise_mask_type = request.noise_mask_type
    noise_strength = max(MIN_NOISE_STRENGTH, min(request.noise_strength, MAX_NOISE_STRENGTH))
    inject_at_step = request.inject_at_step
    init_image = request.init_image
    strength = max(MIN_STRENGTH, min(request.strength, MAX_STRENGTH))
    prompt_image = request.prompt_image
    
    # Validate: need either prompt or prompt_image
    if not prompt.strip() and not prompt_image:
        return {"error": "Either prompt or prompt_image is required"}
    
    # Use placeholder prompt for image-as-prompt mode
    if not prompt.strip() and prompt_image:
        prompt = "image-guided generation"
    
    return StreamingResponse(
        generation_service.generate_stream(
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


@router.get("/health")
async def health():
    """Health check endpoint."""
    loaded_models = pipeline_service.get_loaded_models()
    device = pipeline_service.device
    
    health_info = {
        "status": "healthy",
        "models_loaded": loaded_models,
        "total_models": len(loaded_models),
        "device": str(device) if device else "N/A"
    }
    
    # Add GPU info if available
    if device and device.type == "cuda":
        try:
            health_info["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(device) / 1024**3, 2
            )
            health_info["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved(device) / 1024**3, 2
            )
            health_info["gpu_name"] = torch.cuda.get_device_name(device)
        except:
            pass
    
    return health_info


@router.get("/diagnostics")
async def diagnostics():
    """Detailed diagnostic information."""
    device = pipeline_service.device
    pipelines = pipeline_service.pipelines
    
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
            diag_info["gpu_memory_allocated_gb"] = round(
                torch.cuda.memory_allocated(device) / 1024**3, 2
            )
            diag_info["gpu_memory_reserved_gb"] = round(
                torch.cuda.memory_reserved(device) / 1024**3, 2
            )
            diag_info["gpu_memory_total_gb"] = round(
                torch.cuda.get_device_properties(device).total_memory / 1024**3, 2
            )
        except Exception as e:
            diag_info["gpu_error"] = str(e)
    
    # Check model components
    for model_key, pipeline in pipelines.items():
        try:
            diag_info[f"model_{model_key}_unet_device"] = str(
                next(pipeline.unet.parameters()).device
            )
            diag_info[f"model_{model_key}_text_encoder_device"] = str(
                next(pipeline.text_encoder.parameters()).device
            )
            diag_info[f"model_{model_key}_vae_device"] = str(
                next(pipeline.vae.parameters()).device
            )
        except:
            diag_info[f"model_{model_key}_error"] = "Could not determine device"
    
    return diag_info
