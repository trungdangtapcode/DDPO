# Refactored Python API Structure

## Overview

The codebase has been refactored from a monolithic `main.py` into a clean, modular architecture.

## Project Structure

```
python-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app creation & lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API endpoints (/generate, /health, etc.)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration & constants
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── clip_service.py     # CLIP image encoder service
│   │   ├── pipeline_service.py # SD pipeline management
│   │   └── generation_service.py # Image generation logic
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py      # Image conversion utilities
│       └── noise_mask.py       # Noise mask generation
├── main_new.py                 # New entry point
├── main.py                     # Old monolithic file (can be removed)
└── requirements.txt
```

## Module Responsibilities

### `app/core/config.py`
- Environment variables
- Constants (device, models, parameters)
- Configuration values

### `app/models/schemas.py`
- Pydantic models for request/response validation
- `GenerateRequest`, `StreamData`

### `app/services/clip_service.py`
- CLIP model loading and management
- Image-to-embedding conversion
- Singleton service instance

### `app/services/pipeline_service.py`
- Stable Diffusion pipeline loading
- Pipeline caching and retrieval
- Memory management

### `app/services/generation_service.py`
- Main image generation logic
- Streaming SSE implementation
- Denoising loop
- Latent initialization (txt2img, img2img, custom start)
- Noise injection

### `app/utils/image_utils.py`
- `latent_to_image()` - Decode latents to PIL images
- `pil_to_latent()` - Encode PIL images to latents

### `app/utils/noise_mask.py`
- `create_noise_mask()` - Generate various mask patterns
- Supports 9 mask types (circle, square, edges, etc.)

### `app/api/routes.py`
- FastAPI route handlers
- Request validation
- Response formatting

### `app/main.py`
- FastAPI app creation
- Lifespan management (startup/shutdown)
- Middleware configuration

## Running the Server

### Option 1: Using the new entry point
```bash
python main_new.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Option 3: With auto-reload (development)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Migration from Old main.py

The old `main.py` (900+ lines) has been split into:
- **Config**: 47 lines
- **Models**: 31 lines
- **Image Utils**: 65 lines
- **Noise Mask**: 86 lines
- **CLIP Service**: 99 lines
- **Pipeline Service**: 141 lines
- **Generation Service**: 342 lines
- **Routes**: 167 lines
- **Main App**: 73 lines

**Total**: ~1,051 lines across 9 focused modules (vs 900 lines in one file)

## Benefits

✅ **Separation of Concerns**: Each module has a single responsibility
✅ **Testability**: Services can be unit tested independently
✅ **Maintainability**: Easy to locate and modify specific functionality
✅ **Reusability**: Services can be imported and used elsewhere
✅ **Scalability**: Easy to add new features without touching core logic
✅ **Type Safety**: Better IDE support and type checking
✅ **Documentation**: Each module has clear docstrings

## API Compatibility

The refactored API is **100% backward compatible** with the old version:
- Same endpoints (`/generate`, `/health`, `/diagnostics`)
- Same request/response formats
- Same functionality (txt2img, img2img, noise injection, CLIP prompts)

No frontend or backend changes required!
