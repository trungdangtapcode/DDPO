"""
Stable Diffusion pipeline management service.
"""
import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from typing import Dict, Optional

from app.core.config import AVAILABLE_MODELS, HF_TOKEN, HF_CACHE_DIR


class PipelineService:
    """Service for managing Stable Diffusion pipelines."""
    
    def __init__(self):
        self.pipelines: Dict[str, StableDiffusionPipeline] = {}
        self.device: Optional[torch.device] = None
    
    def set_device(self, device: torch.device):
        """Set the device for pipelines."""
        self.device = device
    
    def load_pipeline(self, model_key: str) -> Optional[StableDiffusionPipeline]:
        """
        Load a specific Stable Diffusion pipeline.
        
        Args:
            model_key: Key of the model to load
            
        Returns:
            Loaded pipeline or None if failed
        """
        # Check if model key is valid
        if model_key not in AVAILABLE_MODELS:
            print(f"❌ Invalid model key: {model_key}")
            return None
        
        # Return if already loaded
        if model_key in self.pipelines:
            print(f"✓ Model '{model_key}' already loaded")
            return self.pipelines[model_key]
        
        model_name = AVAILABLE_MODELS[model_key]
        
        print(f"Loading model '{model_key}': {model_name}")
        print(f"Device: {self.device}")
        
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                cache_dir=HF_CACHE_DIR,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                use_auth_token=HF_TOKEN,
                local_files_only=False,
                low_cpu_mem_usage=True,
            )
            
            # Set scheduler to DDIM
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            
            # Disable safety checker for faster inference
            pipeline.safety_checker = None
            
            # Move to device
            pipeline = pipeline.to(self.device)
            
            # Enable memory optimizations for CUDA
            if self.device.type == "cuda":
                pipeline.enable_attention_slicing(1)
                pipeline.enable_vae_slicing()
                torch.cuda.empty_cache()
            
            # Store in pipelines dict
            self.pipelines[model_key] = pipeline
            print(f"✓ Model '{model_key}' loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model '{model_key}': {e}")
            return None
        
        return pipeline
    
    def load_all_pipelines(self):
        """Load all available models."""
        print("\n" + "="*80)
        print("Loading all DDPO models into VRAM...")
        print("="*80)
        
        for model_key in AVAILABLE_MODELS.keys():
            self.load_pipeline(model_key)
            print()
        
        print("="*80)
        print(f"✓ All {len(self.pipelines)}/{len(AVAILABLE_MODELS)} models loaded")
        print(f"✓ Models ready: {list(self.pipelines.keys())}")
        print("="*80 + "\n")
    
    def get_pipeline(self, model_key: str) -> Optional[StableDiffusionPipeline]:
        """
        Get a loaded pipeline.
        
        Args:
            model_key: Key of the model
            
        Returns:
            Pipeline if loaded, None otherwise
        """
        return self.pipelines.get(model_key)
    
    def unload_all(self):
        """Unload all pipelines to free GPU memory."""
        if self.pipelines:
            for key in list(self.pipelines.keys()):
                del self.pipelines[key]
            self.pipelines = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ All pipelines unloaded, memory freed")
    
    def get_loaded_models(self) -> list:
        """Get list of loaded model keys."""
        return list(self.pipelines.keys())


# Global pipeline service instance
pipeline_service = PipelineService()
