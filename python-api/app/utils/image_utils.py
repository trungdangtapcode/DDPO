"""
Image processing utilities.
"""
import torch
from PIL import Image
import numpy as np


def latent_to_image(vae, latent: torch.Tensor) -> Image.Image:
    """
    Convert latent tensor to PIL Image.
    
    Args:
        vae: VAE decoder
        latent: Latent tensor to decode
        
    Returns:
        PIL Image
    """
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


def pil_to_latent(vae, image: Image.Image, device, dtype) -> torch.Tensor:
    """
    Convert PIL Image to latent tensor.
    
    Args:
        vae: VAE encoder
        image: PIL Image to encode
        device: Target device
        dtype: Target dtype
        
    Returns:
        Latent tensor
    """
    # Resize to standard dimensions
    from app.core.config import IMAGE_WIDTH, IMAGE_HEIGHT
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    
    # Convert PIL image to tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)  # HWC to BCHW
    image_tensor = torch.from_numpy(image_np).to(device, dtype=dtype)
    
    # Normalize to [-1, 1]
    image_tensor = 2.0 * image_tensor - 1.0
    
    # Encode to latent space
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    
    return latents
