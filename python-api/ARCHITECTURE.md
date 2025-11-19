# Architecture Overview

## Refactored Python API Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                      │
│                          (app/main.py)                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              Lifespan Management                        │   │
│  │  - Startup: Load CLIP & SD Pipelines                   │   │
│  │  - Shutdown: Cleanup & Free Memory                     │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API Routes Layer                           │
│                      (app/api/routes.py)                        │
│                                                                  │
│  GET  /           → Root info                                  │
│  POST /generate   → Stream generation (SSE)                    │
│  GET  /health     → Health check + GPU stats                   │
│  GET  /diagnostics→ Detailed diagnostics                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Services Layer                               │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ CLIP Service     │  │ Pipeline Service │  │ Generation   │ │
│  │                  │  │                  │  │ Service      │ │
│  │ • Load CLIP     │  │ • Load SD models │  │ • Encode     │ │
│  │ • Image→Embed   │  │ • Cache pipelines│  │   prompts    │ │
│  │ • Singleton     │  │ • Memory mgmt    │  │ • Initialize │ │
│  │                  │  │ • Get pipeline   │  │   latents    │ │
│  │                  │  │                  │  │ • Denoise    │ │
│  │                  │  │                  │  │   loop       │ │
│  │                  │  │                  │  │ • Stream SSE │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Utils Layer                                 │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Image Utils      │  │ Noise Mask       │                    │
│  │                  │  │                  │                    │
│  │ • latent→image  │  │ • Circle         │                    │
│  │ • PIL→latent    │  │ • Square         │                    │
│  │ • Conversions    │  │ • Edges          │                    │
│  │                  │  │ • 9 patterns     │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Models & Config Layer                         │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Pydantic Schemas │  │ Configuration    │                    │
│  │                  │  │                  │                    │
│  │ • GenerateReq    │  │ • Constants      │                    │
│  │ • StreamData     │  │ • Env vars       │                    │
│  │ • Validation     │  │ • Model list     │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow

```
Client Request
      │
      ▼
[POST /generate with JSON body]
      │
      ▼
[routes.py: Validate & normalize params]
      │
      ▼
[generation_service.generate_stream()]
      │
      ├─── Use prompt_image? ──► [clip_service.image_to_embedding()]
      │                                      │
      │                                      ▼
      ├─────────────────────────► [Encode prompt/embeddings]
      │
      ▼
[Get pipeline from pipeline_service]
      │
      ▼
[Initialize latents]
      │
      ├─── img2img? ──► [pil_to_latent(init_image)]
      ├─── custom start? ──► [pil_to_latent(start_image) + noise]
      └─── txt2img? ──► [random noise]
      │
      ▼
[Denoising loop: steps iterations]
      │
      ├─── inject noise? ──► [create_noise_mask() + apply]
      │
      ├─── UNet prediction
      ├─── Scheduler step
      │
      ▼
[latent_to_image()] ──► [Encode to base64]
      │
      ▼
[Yield SSE: data: {step, image, progress, ...}]
      │
      ▼
[Stream to client in real-time]
      │
      ▼
[Done: yield "data: [DONE]"]
```

## Data Flow: Image-as-Prompt

```
User uploads image
      │
      ▼
[Base64 encoded in frontend]
      │
      ▼
[POST /generate with prompt_image field]
      │
      ▼
[clip_service.image_to_embedding()]
      │
      ├─► Decode base64
      ├─► CLIP processor
      ├─► CLIP.get_image_features() → [1, 768]
      └─► Repeat 77 times → [1, 77, 768]
      │
      ▼
[Use as prompt_embeds instead of text encoding]
      │
      ▼
[Continue normal diffusion process]
```

## Service Responsibilities

| Service              | Purpose                                  | Key Methods                          |
|---------------------|------------------------------------------|--------------------------------------|
| `CLIPService`       | CLIP model for image prompts            | `load()`, `image_to_embedding()`     |
| `PipelineService`   | Manage SD model loading/caching         | `load_pipeline()`, `get_pipeline()`  |
| `GenerationService` | Core generation logic & streaming       | `generate_stream()`, `_denoise_loop()`|

## File Size Comparison

| Old Structure      | Lines | New Structure                    | Lines |
|-------------------|-------|----------------------------------|-------|
| main.py           | ~900  | app/core/config.py              |    47 |
|                   |       | app/models/schemas.py            |    31 |
|                   |       | app/utils/image_utils.py         |    65 |
|                   |       | app/utils/noise_mask.py          |    86 |
|                   |       | app/services/clip_service.py     |    99 |
|                   |       | app/services/pipeline_service.py |   141 |
|                   |       | app/services/generation_service.py|  342 |
|                   |       | app/api/routes.py                |   167 |
|                   |       | app/main.py                      |    73 |
| **Total**         | **900** | **Total**                     | **1,051** |

*Note: Increased line count due to better structure, docstrings, and separation*
