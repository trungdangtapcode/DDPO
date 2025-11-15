# Quick Reference Guide

## Start All Services (One-liner)

```bash
# Terminal 1: Python API (Port 8000)
cd python-api && source venv/bin/activate && python main.py

# Terminal 2: Backend (Port 3001)
cd backend && npm start

# Terminal 3: Frontend (Port 5174)
cd frontend && npm run dev
```

## URLs

- **Frontend**: http://localhost:5174
- **Backend**: http://localhost:3001
- **Python API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Available Models

| Model | Key | Optimized For |
|-------|-----|---------------|
| Aesthetic Quality | `aesthetic` | Visual appeal |
| Text Alignment | `alignment` | Prompt accuracy |
| Compressibility | `compressibility` | Smaller files |
| Incompressibility | `incompressibility` | Maximum detail |

## API Examples

### Generate with Model Selection
```bash
# Using curl
curl "http://localhost:8000/generate?prompt=sunset&steps=20&model=aesthetic"

# Using the UI
1. Open http://localhost:5174
2. Select model from dropdown
3. Enter prompt
4. Click Generate
```

### Check Loaded Models
```bash
curl http://localhost:8000/health
```

## Common Commands

### Python API
```bash
cd python-api
source venv/bin/activate
python main.py                    # Start server
pip install -r requirements.txt   # Install deps
```

### Backend
```bash
cd backend
npm start                         # Start server
npm install                       # Install deps
```

### Frontend
```bash
cd frontend
npm run dev                       # Start dev server
npm run build                     # Build production
npm install                       # Install deps
```

## Environment Variables

### python-api/.env
```env
DEVICE=cuda:3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### backend/.env
```env
PORT=3001
PYTHON_API_URL=http://localhost:8000
```

### frontend/.env
```env
VITE_API_URL=http://localhost:3001
```

## File Structure

```
dif/
├── frontend/           # ViteJS + React + TypeScript
│   ├── src/
│   │   ├── components/ui/
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── package.json
├── backend/            # Node.js + Express
│   ├── server.js
│   └── package.json
└── python-api/         # FastAPI + Stable Diffusion
    ├── main.py
    └── requirements.txt
```

## Memory Usage

- **Per Model**: ~2.5 GB VRAM
- **Total**: ~10 GB VRAM (4 models)
- **GPU**: CUDA device (cuda:3)

## Troubleshooting Quick Fixes

### Python API won't start
```bash
# Check CUDA device
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
df -h ~/.cache/huggingface/

# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

### Backend connection errors
```bash
# Verify Python API is running
curl http://localhost:8000/health

# Check port availability
lsof -i :3001
```

### Frontend not loading
```bash
# Check backend is running
curl http://localhost:3001/api/health

# Clear cache and rebuild
rm -rf node_modules dist
npm install
npm run dev
```

## Development Mode

### Auto-reload on changes
```bash
# Python API
cd python-api && uvicorn main:app --reload

# Backend
cd backend && npm run dev  # (requires nodemon)

# Frontend (always auto-reloads)
cd frontend && npm run dev
```

## Production Build

### Frontend
```bash
cd frontend
npm run build
npm run preview  # Test production build
```

### Deploy
```bash
# Copy built frontend to static server
cp -r frontend/dist /var/www/html/

# Run backend with PM2
pm2 start backend/server.js --name diffusion-backend

# Run Python API with supervisor/systemd
# See REAL_MODEL_GUIDE.md for details
```

## Testing Endpoints

```bash
# Test Python API directly
curl "http://localhost:8000/generate?prompt=test&steps=5&model=compressibility"

# Test through backend
curl "http://localhost:3001/api/generate?prompt=test&steps=5&model=aesthetic"

# Health checks
curl http://localhost:8000/health
curl http://localhost:3001/api/health
```

## Model Comparison Workflow

1. **Same Prompt**: Use identical text across models
2. **Same Steps**: Keep denoising steps consistent (e.g., 20)
3. **Compare**: Note differences in:
   - Visual quality (aesthetic)
   - Text matching (alignment)
   - File size (compressibility)
   - Detail level (incompressibility)

## Documentation Links

- **[README.md](README.md)** - Main documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Setup guide
- **[MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md)** - Model selection
- **[REAL_MODEL_GUIDE.md](REAL_MODEL_GUIDE.md)** - Stable Diffusion setup
- **[CONFIG.md](CONFIG.md)** - Configuration
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design

## Support

- Check GPU memory: `nvidia-smi`
- View Python logs: `tail -f logs/api.log`
- View Backend logs: `tail -f logs/backend.log`
- Browser console: F12 (Developer Tools)


## Start Commands

```bash
# All services at once
./start.sh

# Or individually:
cd python-api && source venv/bin/activate && python main.py  # Port 8000
cd backend && npm start                                        # Port 3001  
cd frontend && npm run dev                                     # Port 5173
```

## URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:3001 |
| Python API | http://localhost:8000 |

## Configuration Files

| File | Purpose |
|------|---------|
| `backend/.env` | PORT=3001, PYTHON_API_URL |
| `python-api/.env` | PORT=8000, HOST |
| `frontend/.env` | VITE_API_URL |

## Common Issues & Solutions

### Port 3001 in use
```bash
lsof -i:3001
kill -9 <PID>
# Or change PORT in backend/.env
```

### Can't connect to services
```bash
# Check all .env files exist
ls backend/.env python-api/.env frontend/.env

# Verify ports in config match
cat backend/.env  # Should say PORT=3001
```

### Dependencies missing
```bash
cd backend && npm install
cd python-api && pip install -r requirements.txt
cd frontend && npm install
```

## Key Files to Customize

- `frontend/src/App.tsx` - Main UI
- `python-api/main.py` - Image generation logic
- `backend/server.js` - API routing
- `frontend/tailwind.config.js` - Styling

## Documentation

- `README.md` - Full overview
- `QUICKSTART.md` - Get started fast
- `CONFIG.md` - Port & config details
- `ARCHITECTURE.md` - System design
- `SETUP_COMPLETE.md` - What changed

## Test the Stack

```bash
# 1. Start all services
./start.sh

# 2. Open browser
open http://localhost:5173

# 3. Enter prompt: "A beautiful sunset"
# 4. Click Generate
# 5. Watch the magic! ✨
```

---
**Port Change Summary**: Backend moved from `3000` → `3001` to avoid conflicts!
