# Architecture Documentation

## System Overview

This is a three-tier architecture demonstrating streaming diffusion model image generation.

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
│                    ViteJS + React + TS                      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   App.tsx    │  │  UI Components│  │  Tailwind +  │     │
│  │   (Main UI)  │  │   (shadcn/ui) │  │   shadcn/ui  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  Features:                                                  │
│  • Server-Sent Events (SSE) client                         │
│  • Real-time image updates                                 │
│  • Progress tracking                                       │
│  • TypeScript type safety                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      NODE.JS BACKEND                        │
│                    Express + Axios                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               server.js                              │  │
│  │  • CORS middleware                                   │  │
│  │  • SSE proxy to Python API                           │  │
│  │  • Request/Response forwarding                       │  │
│  │  • Error handling                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Responsibilities:                                          │
│  • API Gateway                                             │
│  • Stream proxying                                         │
│  • CORS handling                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/SSE
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON API SERVICE                       │
│                   FastAPI + Uvicorn                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    main.py                           │  │
│  │                                                      │  │
│  │  def create_mock_image():                           │  │
│  │    • Generate progressive images                    │  │
│  │    • Simulate denoising steps                       │  │
│  │    • Add noise → Clear progression                  │  │
│  │                                                      │  │
│  │  async def generate_image_stream():                 │  │
│  │    • Yield intermediate steps                       │  │
│  │    • Base64 encode images                           │  │
│  │    • Send SSE events                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Technologies:                                              │
│  • Pillow (PIL) for image generation                       │
│  • asyncio for async processing                            │
│  • SSE for streaming                                       │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Image Generation Request

```
User Input (Prompt)
    ↓
Frontend (App.tsx)
    ↓ EventSource connection
Backend (server.js)
    ↓ Axios stream
Python API (main.py)
    ↓ Start generation loop
```

### 2. Streaming Response

```
Python API
    ↓ Generate step 1 image
    ↓ SSE: data: {"step": 1, "image": "base64...", ...}
Backend
    ↓ Proxy stream
Frontend
    ↓ Update UI
    ↓ Display image
    ↓ Update progress bar
    
    ... repeat for each step ...
    
    ↓ Final step
    ↓ SSE: data: [DONE]
Frontend
    ↓ Close EventSource
    ↓ Mark complete
```

## Component Breakdown

### Frontend Components

#### `App.tsx`
- Main application component
- State management for prompt, image, progress
- EventSource handling for SSE
- UI composition

#### `components/ui/button.tsx`
- Reusable button component
- Variants: default, destructive, outline, ghost, link
- Sizes: default, sm, lg, icon
- Built with Radix UI Slot

#### `components/ui/card.tsx`
- Container components
- CardHeader, CardTitle, CardDescription
- CardContent, CardFooter
- Used for layout sections

#### `components/ui/input.tsx`
- Text input component
- Tailwind styling
- Accessible and responsive

#### `components/ui/progress.tsx`
- Progress bar component
- Radix UI Progress primitive
- Animated width transition

#### `lib/utils.ts`
- Utility functions
- `cn()` - className merging with tailwind-merge

### Backend Components

#### `server.js`
- Express server setup
- CORS middleware configuration
- `/api/generate` - SSE proxy endpoint
- `/api/health` - Health check with Python API status
- Stream handling and client disconnect management

### Python API Components

#### `main.py`
- FastAPI application
- `create_mock_image()` - Image generation logic
  - Progressive noise reduction
  - Color gradient generation
  - Geometric shapes
  - Text overlay
- `generate_image_stream()` - Async generator for SSE
- CORS middleware for cross-origin requests

## Technology Choices

### Why ViteJS?
- ⚡ Extremely fast HMR (Hot Module Replacement)
- 🎯 Native TypeScript support
- 📦 Optimized builds with Rollup
- 🔧 Minimal configuration

### Why TypeScript?
- 🛡️ Type safety
- 🔍 Better IDE support
- 📚 Self-documenting code
- 🐛 Catch errors early

### Why Tailwind CSS?
- 🚀 Rapid development
- 📱 Responsive by default
- 🎨 Consistent design system
- 🔧 Highly customizable

### Why shadcn/ui?
- ♿ Accessibility (Radix UI)
- 🎨 Customizable
- 📦 Copy-paste components
- 🔧 No package bloat

### Why Node.js Backend?
- 🔄 Easy proxying
- 🌐 CORS handling
- 🔌 Stream management
- 🚀 Fast and lightweight

### Why FastAPI?
- ⚡ Fast performance
- 📝 Auto API documentation
- 🔄 Async/await support
- 🎯 Type hints with Pydantic
- 🌊 Easy SSE implementation

## Streaming Implementation

### Server-Sent Events (SSE)

**Why SSE over WebSockets?**
- ✅ Simpler implementation
- ✅ Automatic reconnection
- ✅ HTTP-based (easier proxying)
- ✅ One-way communication (perfect for this use case)
- ✅ Built-in browser support

**SSE Format:**
```
data: {"step": 1, "total_steps": 20, "image": "base64...", "progress": 5}\n\n
data: {"step": 2, "total_steps": 20, "image": "base64...", "progress": 10}\n\n
...
data: [DONE]\n\n
```

**Frontend SSE Client:**
```typescript
const eventSource = new EventSource('/api/generate?prompt=...')
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data)
  // Update UI
}
```

**Backend SSE Proxy:**
```javascript
res.setHeader('Content-Type', 'text/event-stream')
response.data.on('data', (chunk) => res.write(chunk))
```

**Python SSE Server:**
```python
async def generate_image_stream():
    yield f"data: {json.dumps(data)}\n\n"
```

## Performance Optimizations

### Frontend
- ⚡ Vite's fast rebuild
- 🎯 Code splitting
- 📦 Tree shaking
- 🖼️ Lazy loading

### Backend
- 🔄 Stream proxying (no buffering)
- 🚀 Async handling
- 💾 Minimal memory usage

### Python API
- ⚡ Async image generation
- 📸 JPEG compression (quality 85)
- 🔄 Generator pattern (memory efficient)
- ⏱️ Controlled delay (0.3s per step)

## Security Considerations

### Current Implementation
- ✅ CORS enabled (development)
- ✅ Input validation (prompt length)
- ✅ Steps parameter bounds (5-50)

### Production Recommendations
- 🔒 Add authentication
- 🛡️ Rate limiting
- 🔐 Input sanitization
- 📊 Request logging
- 🚫 CORS restrictions
- 🔑 API keys

## Scalability

### Current Limits
- Single instance
- In-memory processing
- Synchronous image generation

### Future Improvements
- 📦 Redis for queue management
- 🔄 Worker pools
- 💾 S3 for image storage
- 📊 Load balancing
- 🎯 CDN for static assets

## Testing Strategy

### Frontend
```bash
# Unit tests
npm run test

# E2E tests
npm run test:e2e
```

### Backend
```bash
# Integration tests
npm run test

# Load testing
npm run test:load
```

### Python API
```bash
# Unit tests
pytest

# API tests
pytest tests/test_api.py
```

## Monitoring

### Metrics to Track
- 📊 Generation time per image
- 🔄 Concurrent generations
- 💾 Memory usage
- 🌐 API response times
- ❌ Error rates

### Recommended Tools
- Prometheus + Grafana
- DataDog
- New Relic
- Sentry (error tracking)

## Deployment

### Development
```bash
./setup.sh && ./start.sh
```

### Production

**Frontend:**
```bash
cd frontend
npm run build
# Deploy dist/ to Vercel/Netlify/S3+CloudFront
```

**Backend:**
```bash
cd backend
# Deploy to Heroku/Railway/AWS EC2
```

**Python API:**
```bash
cd python-api
# Deploy to AWS Lambda/Google Cloud Run/Railway
# Or: Docker container to any cloud
```

### Docker Deployment
```dockerfile
# Example Dockerfile for each service
# See individual README files
```

## Future Enhancements

### Short Term
- [ ] Save generated images
- [ ] Generation history
- [ ] More customization options
- [ ] Image-to-image mode

### Long Term
- [ ] Real Stable Diffusion integration
- [ ] Multi-user support
- [ ] Image gallery
- [ ] Social features (sharing)
- [ ] Advanced settings (CFG, sampling)
- [ ] Multiple models support
- [ ] API marketplace

---

**Built with modern best practices for scalability and maintainability**
