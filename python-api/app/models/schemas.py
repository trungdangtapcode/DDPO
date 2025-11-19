"""
Pydantic models for request/response validation.
"""
from typing import Optional
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = ""  # Optional when using prompt_image
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
    prompt_image: Optional[str] = None  # Base64 image to use as prompt


class StreamData(BaseModel):
    """Data structure for streaming response."""
    step: int
    total_steps: int
    image: str
    progress: float
    done: bool
    model: str
