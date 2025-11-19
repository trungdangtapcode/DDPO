"""
FastAPI application with lifespan management.
"""
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import DEVICE
from app.services.pipeline_service import pipeline_service
from app.services.clip_service import clip_service
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    print("Starting up FastAPI server...")
    
    # Configure device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device configured: {device}")
    
    # Load CLIP for image-as-prompt
    print("\n" + "="*80)
    print("Loading CLIP image encoder (for image-as-prompt)...")
    print("="*80)
    clip_service.load(device)
    print("="*80 + "\n")
    
    # Load all SD pipelines
    pipeline_service.set_device(device)
    pipeline_service.load_all_pipelines()
    
    print("✓ API ready - models are in VRAM and ready for inference\n")
    
    yield
    
    # Shutdown
    print("\nShutting down...")
    pipeline_service.unload_all()
    print("Server shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Stable Diffusion API",
        description="Real-time streaming diffusion with DDPO models",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


# Create app instance
app = create_app()
