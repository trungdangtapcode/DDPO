#!/bin/bash

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Set PyTorch memory allocation config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting Python API with Real Stable Diffusion..."
echo ""
echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Model: $MODEL_NAME"
echo "  Port: $PORT"
echo "  Memory Config: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Start the API
python main.py
