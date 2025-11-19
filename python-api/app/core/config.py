"""
Application configuration and constants.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Device configuration
DEVICE = os.getenv("DEVICE", "cuda:0")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", None)

# Available DDPO models
AVAILABLE_MODELS = {
    "aesthetic": "kvablack/ddpo-aesthetic",
    "alignment": "kvablack/ddpo-alignment",
    "compressibility": "kvablack/ddpo-compressibility",
    "incompressibility": "kvablack/ddpo-incompressibility"
}

# CLIP model for image-as-prompt
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Generation parameters
DEFAULT_STEPS = 20
MIN_STEPS = 5
MAX_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 20.0
DEFAULT_STRENGTH = 0.75
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0
DEFAULT_NOISE_STRENGTH = 1.0
MIN_NOISE_STRENGTH = 0.0
MAX_NOISE_STRENGTH = 2.0

# Image dimensions
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LATENT_HEIGHT = 64  # 512 / 8
LATENT_WIDTH = 64   # 512 / 8
