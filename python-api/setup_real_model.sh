#!/bin/bash

echo "🚀 Setting up Python API with Real Stable Diffusion Model..."
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in a virtual environment. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
else
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
fi

echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Configuration:"
echo "  1. Edit .env file to set your HuggingFace token (if needed):"
echo "     HF_TOKEN=your_token_here"
echo ""
echo "  2. Set your device (default: cuda:3):"
echo "     DEVICE=cuda:3"
echo ""
echo "  3. Optional: Change the model:"
echo "     MODEL_NAME=kvablack/ddpo-compressibility"
echo ""
echo "🚀 To start the API:"
echo "   python main.py"
echo ""
echo "   The model will be downloaded on first run (~4-5 GB)"
echo "   Subsequent runs will use the cached model"
echo ""
