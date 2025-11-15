# Real Stable Diffusion Integration Guide

## Overview

The Python API now supports **real Stable Diffusion** image generation with streaming intermediate steps! The system automatically falls back to mock mode if the model cannot be loaded.

## Features

✅ **Real Stable Diffusion** - Uses `kvablack/ddpo-compressibility` model  
✅ **Streaming Generation** - See every denoising step in real-time  
✅ **GPU Support** - Optimized for CUDA (cuda:3 by default)  
✅ **Automatic Fallback** - Uses mock mode if model fails to load  
✅ **Memory Efficient** - Attention slicing enabled  
✅ **Configurable** - All settings via `.env` file  

## Quick Start

### 1. Setup Environment

```bash
cd python-api

# Run setup script
chmod +x setup_real_model.sh
./setup_real_model.sh

# Or manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
# python-api/.env
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model Configuration
MODEL_NAME=kvablack/ddpo-compressibility
DEVICE=cuda:3
HF_TOKEN=your_huggingface_token_if_needed
```

### 3. Start the API

```bash
cd python-api
source venv/bin/activate
python main.py
```

The model will download automatically on first run (~4-5 GB).

## Configuration Options

### Device Selection

```bash
# Use specific GPU
DEVICE=cuda:3

# Use different GPU
DEVICE=cuda:0

# Use CPU (slower)
DEVICE=cpu
```

### Model Selection

```bash
# DDPO Compressibility (default)
MODEL_NAME=kvablack/ddpo-compressibility

# Standard Stable Diffusion 1.5
MODEL_NAME=runwayml/stable-diffusion-v1-5

# Stable Diffusion 2.1
MODEL_NAME=stabilityai/stable-diffusion-2-1
```

### HuggingFace Token

Some models require authentication:

```bash
# Get token from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_token_here

# Or login via CLI
huggingface-cli login
```

## API Endpoints

### `GET /`
Get API status and model info

**Response:**
```json
{
  "message": "Diffusion API",
  "status": "running",
  "model": "stable-diffusion",
  "device": "cuda:3"
}
```

### `GET /generate?prompt=...&steps=...`
Generate image with streaming

**Parameters:**
- `prompt` (required): Text description
- `steps` (optional): Number of denoising steps (default: 20)

**Response:** Server-Sent Events (SSE) stream

**Event Data:**
```json
{
  "step": 15,
  "total_steps": 20,
  "image": "base64_encoded_jpeg",
  "progress": 75.0,
  "done": false,
  "model": "stable-diffusion"
}
```

### `GET /health`
Health check

**Response:**
```json
{
  "status": "healthy",
  "model": "stable-diffusion",
  "device": "cuda:3"
}
```

## How It Works

### Real Model Mode

1. **Startup**: Model loads into GPU memory
2. **Request**: Client sends prompt via SSE
3. **Encoding**: Text prompt → embeddings
4. **Denoising**: 20-50 steps (configurable)
5. **Streaming**: Each step → decode latent → send image
6. **Complete**: Final high-quality image sent

### Mock Mode (Fallback)

If the model fails to load:
- API continues running
- Generates mock gradient images
- Same SSE interface
- Useful for testing without GPU

## Performance

### Model Loading Time
- **First Run**: ~30-60 seconds (download model)
- **Subsequent Runs**: ~10-20 seconds (load from cache)

### Generation Speed (cuda:3)
- **20 steps**: ~3-5 seconds
- **50 steps**: ~8-12 seconds

### Memory Usage
- **GPU VRAM**: ~3-4 GB
- **RAM**: ~6-8 GB

## Troubleshooting

### Model Won't Download

**Error**: `Cannot find an appropriate cached...`

**Solution**:
```bash
# Set HuggingFace token
export HF_TOKEN=your_token

# Or download manually
python scripts/download_model.py --model kvablack/ddpo-compressibility

# Or use local model
MODEL_NAME=/path/to/local/stable-diffusion
```

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
```bash
# 1. Reduce steps
# In API request: ?steps=20

# 2. Use smaller model
MODEL_NAME=CompVis/stable-diffusion-v1-4

# 3. Use different GPU
DEVICE=cuda:0

# 4. Use CPU (slower)
DEVICE=cpu
```

### Wrong GPU

**Check available GPUs**:
```bash
nvidia-smi
```

**Set correct device**:
```bash
# Edit .env
DEVICE=cuda:3
```

### Port Already in Use

```bash
# Change port in .env
PORT=8001

# Update backend/.env too
PYTHON_API_URL=http://localhost:8001
```

## Integration with Frontend

The frontend automatically works with both modes:

```typescript
// Frontend receives SSE events
{
  "model": "stable-diffusion"  // or "mock"
  "step": 10,
  "image": "...",
  // ...
}
```

Display model badge:
```typescript
{data.model === 'stable-diffusion' ? '🎨 Real SD' : '🎭 Mock'}
```

## Development

### Test Real Model

```bash
cd python-api
source venv/bin/activate

# Test generation
curl "http://localhost:8000/generate?prompt=a%20cat&steps=20"

# Check status
curl http://localhost:8000/
```

### Monitor GPU

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Download model during build
RUN python -c "from diffusers import StableDiffusionPipeline; \
    StableDiffusionPipeline.from_pretrained('kvablack/ddpo-compressibility')"

CMD ["python", "main.py"]
```

### Environment Variables

```bash
# Production .env
HOST=0.0.0.0
PORT=8000
WORKERS=4
MODEL_NAME=kvablack/ddpo-compressibility
DEVICE=cuda:0
```

### Scaling

- **Multiple Workers**: Set `WORKERS=4` for better throughput
- **Load Balancer**: Use nginx/caddy in front
- **Queue System**: Add Redis for request queuing
- **Batch Processing**: Group multiple requests

## Comparison

| Feature | Mock Mode | Real Model |
|---------|-----------|------------|
| Speed | Fast (~0.3s/step) | Medium (~0.2-0.5s/step) |
| Quality | Low (gradients) | High (photorealistic) |
| Memory | Minimal | 3-4 GB VRAM |
| Setup | None | Model download |
| Use Case | Testing/Demo | Production |

## Next Steps

1. ✅ Setup and test with mock mode
2. 🔧 Configure `.env` with your GPU
3. 📥 Let model download on first run
4. 🎨 Test with real prompts
5. 🚀 Integrate with frontend

---

**The API now supports real Stable Diffusion! 🎉**
