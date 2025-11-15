# 🚀 Quick Reference Card

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
