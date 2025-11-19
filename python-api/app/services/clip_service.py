"""
CLIP image encoder service for image-as-prompt functionality.
"""
from typing import Optional
import base64
import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

USE_LAST_HIDDEN_STATE = False


from app.core.config import CLIP_MODEL_NAME


class CLIPService:
    """Service for CLIP image embedding extraction."""
    
    def __init__(self):
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device = None
    
    def load(self, device: torch.device):
        """
        Load CLIP model and processor.
        
        Args:
            device: Target device for model
        """
        self.device = device
        
        try:
            print(f"Loading CLIP model: {CLIP_MODEL_NAME}")
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self.model = self.model.to(device)
            self.model.eval()
            print("✓ CLIP image encoder loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load CLIP encoder: {e}")
            print("  Image-as-prompt feature will be disabled")
            self.model = None
            self.processor = None
    
    def is_loaded(self) -> bool:
        """Check if CLIP model is loaded."""
        return self.model is not None and self.processor is not None
    
    def image_to_embedding(self, image_b64: str) -> Optional[torch.Tensor]:
        """
        Convert a base64 image to CLIP image embedding for use as prompt.
        
        Args:
            image_b64: Base64 encoded image
        
        Returns:
            Tensor of shape [1, 77, 768] compatible with SD text encoder output
            Returns None if CLIP model not loaded or error occurs
        """
        if not self.is_loaded():
            print("❌ CLIP model not loaded - image-as-prompt feature unavailable")
            return None
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process image with CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            if USE_LAST_HIDDEN_STATE:
                with torch.no_grad():
                    outputs = self.model.vision_model(**inputs, output_hidden_states=True)
                    # Use the last hidden state as image features
                    clip_features = outputs.hidden_states[-1]  # Shape: [1, 257, 1024], expecting [1, 257, 768]
                    
                
                print(f"✓ CLIP image embedding extracted: {clip_features.shape}")
                return clip_features
                
            # Extract CLIP image features
            with torch.no_grad():
                clip_features = self.model.get_image_features(**inputs)  # Shape: [1, 768]
            
            # Expand to match SD text encoder output shape: [1, 77, 768]
            # Repeat the image embedding across all 77 token positions
            image_prompt_embed = clip_features.unsqueeze(1).repeat(1, 1, 1)
            
            print(f"✓ CLIP image embedding extracted: {image_prompt_embed.shape}")
            return image_prompt_embed
            
        except Exception as e:
            print(f"❌ Error extracting CLIP embedding: {e}")
            return None


# Global CLIP service instance
clip_service = CLIPService()
