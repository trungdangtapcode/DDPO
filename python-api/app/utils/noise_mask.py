"""
Noise mask generation utilities.
"""
import math
import torch


def create_noise_mask(
    mask_type: str,
    height: int = 64,
    width: int = 64,
    device=None
) -> torch.Tensor:
    """
    Create a noise mask in latent space.
    
    Args:
        mask_type: Type of mask pattern
        height: Latent height (default 64 for 512px images)
        width: Latent width (default 64 for 512px images)
        device: Torch device
    
    Returns:
        Tensor of shape (1, 1, height, width) with values 0-1
    """
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
        mask[0, 0, :, :width//2] = 1.0
    
    elif mask_type == "right_half":
        mask[0, 0, :, width//2:] = 1.0
    
    elif mask_type == "top_half":
        mask[0, 0, :height//2, :] = 1.0
    
    elif mask_type == "bottom_half":
        mask[0, 0, height//2:, :] = 1.0
    
    elif mask_type == "checkerboard":
        # Checkerboard pattern
        for y in range(height):
            for x in range(width):
                if (y // 8 + x // 8) % 2 == 0:
                    mask[0, 0, y, x] = 1.0
    
    # "none" or any other value returns zero mask (no noise injection)
    return mask
