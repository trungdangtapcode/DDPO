# Multi-Model Selection Guide

## Overview
The diffusion system now supports 4 DDPO-optimized Stable Diffusion models that can be loaded simultaneously and selected via the UI.

## Available Models

### 1. **Aesthetic Quality** (`aesthetic`)
- Model: `kvablack/ddpo-aesthetic`
- Optimized for: Visual appeal and aesthetic quality
- Best for: Creating visually pleasing, high-quality images

### 2. **Text Alignment** (`alignment`)
- Model: `kvablack/ddpo-alignment`
- Optimized for: Matching the text prompt accurately
- Best for: When prompt accuracy is critical

### 3. **Compressibility** (`compressibility`)
- Model: `kvablack/ddpo-compressibility`
- Optimized for: Images that compress well (lower file sizes)
- Best for: When you need smaller image files

### 4. **Incompressibility** (`incompressibility`)
- Model: `kvablack/ddpo-incompressibility`
- Optimized for: Maximum detail and information density
- Best for: High-detail images with complex textures

## Architecture Changes

### Python API (`python-api/main.py`)
```python
# Multi-pipeline management
pipelines = {}

AVAILABLE_MODELS = {
    "aesthetic": "kvablack/ddpo-aesthetic",
    "alignment": "kvablack/ddpo-alignment",
    "compressibility": "kvablack/ddpo-compressibility",
    "incompressibility": "kvablack/ddpo-incompressibility",
}

# Load all models at startup
@app.on_event("startup")
async def startup_event():
    await load_all_pipelines()

# Generate with model selection
@app.get("/generate")
async def generate(prompt: str, steps: int = 20, model: str = "compressibility"):
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    
    return StreamingResponse(
        generate_image_stream(prompt, steps, model),
        media_type="text/event-stream"
    )
```

### Backend (`backend/server.js`)
```javascript
app.get('/api/generate', async (req, res) => {
    const { prompt, steps, model } = req.query;
    
    const response = await axios.get(`${PYTHON_API_URL}/generate`, {
        params: { 
            prompt, 
            steps: steps || 20,
            model: model || 'compressibility'
        },
        responseType: 'stream'
    });
    
    // Forward stream...
});
```

### Frontend (`frontend/src/App.tsx`)
```typescript
const AVAILABLE_MODELS = [
  { value: 'aesthetic', label: 'Aesthetic Quality' },
  { value: 'alignment', label: 'Text Alignment' },
  { value: 'compressibility', label: 'Compressibility' },
  { value: 'incompressibility', label: 'Incompressibility' },
]

// Model selector in UI
<Select
  value={selectedModel}
  onChange={(e) => setSelectedModel(e.target.value)}
  disabled={isGenerating}
>
  {AVAILABLE_MODELS.map((model) => (
    <option key={model.value} value={model.value}>
      {model.label}
    </option>
  ))}
</Select>
```

## Memory Requirements

- **Per Model**: ~2.5 GB VRAM
- **Total (4 models)**: ~10 GB VRAM
- **GPU**: CUDA device (configured for cuda:3)

## API Endpoints

### `/` - Root
Returns available models and loaded status:
```json
{
  "message": "Diffusion Model API",
  "available_models": ["aesthetic", "alignment", "compressibility", "incompressibility"],
  "models_loaded": ["aesthetic", "alignment", "compressibility", "incompressibility"]
}
```

### `/generate` - Generate Image
Query parameters:
- `prompt` (required): Text description
- `steps` (optional, default=20): Number of denoising steps
- `model` (optional, default="compressibility"): Model to use

Returns: Server-Sent Events stream with progress updates

### `/health` - Health Check
Returns system status and loaded models:
```json
{
  "status": "healthy",
  "device": "cuda:3",
  "models_loaded": ["aesthetic", "alignment", "compressibility", "incompressibility"]
}
```

## Usage

### 1. Start the Services

```bash
# Terminal 1: Python API (loads all 4 models)
cd python-api
source venv/bin/activate  # or your virtual environment
python main.py

# Terminal 2: Node.js Backend
cd backend
npm start

# Terminal 3: Frontend
cd frontend
npm run dev
```

### 2. Use the UI

1. Open http://localhost:5174 in your browser
2. Select a model from the dropdown (default: Compressibility)
3. Enter your prompt
4. Click "Generate" to start image generation
5. Watch the real-time progress as the image forms
6. Try different models with the same prompt to compare results!

### 3. Direct API Usage

```bash
# Using curl with model selection
curl "http://localhost:8000/generate?prompt=a%20sunset&steps=20&model=aesthetic"

# Check loaded models
curl http://localhost:8000/health
```

## Model Comparison Tips

To compare models effectively:
1. Use the **same prompt** across different models
2. Keep **steps consistent** (e.g., 20 steps)
3. Compare results for:
   - Visual quality (aesthetic)
   - Prompt accuracy (alignment)
   - File size (compressibility)
   - Detail level (incompressibility)

## Troubleshooting

### Models not loading
- **Check VRAM**: Ensure at least 10 GB available
- **Check logs**: Look for error messages during startup
- **Verify cache**: Models download to `~/.cache/huggingface/`

### Out of memory errors
- **Reduce concurrent requests**: Only one generation at a time
- **Check GPU usage**: `nvidia-smi` to monitor VRAM
- **Memory optimizations**: Enabled via attention_slicing and vae_slicing

### Model selection not working
- **Check parameter**: Ensure model name is correct (case-sensitive)
- **Verify dropdown**: Should show 4 options
- **Check network**: Verify request includes model parameter

## Performance Notes

- **First generation**: Slower as models initialize
- **Subsequent generations**: Faster with cached pipelines
- **Memory usage**: Stays constant after all models loaded
- **Switch time**: Instant (models are pre-loaded)

## Future Enhancements

- [ ] Add model-specific settings (e.g., guidance scale)
- [ ] Side-by-side comparison mode
- [ ] Model performance metrics
- [ ] Dynamic model loading (load on-demand)
- [ ] Custom model support
