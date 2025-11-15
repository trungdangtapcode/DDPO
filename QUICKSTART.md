# Quick Start Guide

## Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Start all services
./start.sh
```

Then open `http://localhost:5173` in your browser!

## Option 2: Manual Setup

### Terminal 1 - Python API
```bash
cd python-api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Terminal 2 - Node.js Backend
```bash
cd backend
npm install
npm start  # Runs on port 3001
```

### Terminal 3 - Frontend
```bash
cd frontend
npm install
npm run dev
```

### Access the App
Open `http://localhost:5173` in your browser

## 🎨 How to Use

1. Enter a text prompt (e.g., "A beautiful sunset over mountains")
2. Click "Generate"
3. Watch the diffusion process in real-time!
4. The image will progressively become clearer

## 🔧 Development Mode

### Frontend (with hot reload)
```bash
cd frontend
npm run dev
```

### Backend (with auto-restart)
```bash
cd backend
npm run dev
```

### Python API (with auto-reload)
```bash
cd python-api
source venv/bin/activate
uvicorn main:app --reload
```

## 📦 Build for Production

### Frontend
```bash
cd frontend
npm run build
npm run preview  # Preview production build
```

## 🐛 Common Issues

**Port already in use?**
```bash
# Backend now uses port 3001 (not 3000) to avoid conflicts
# You can change ports in .env files:

# python-api/.env
PORT=8000

# backend/.env
PORT=3001
PYTHON_API_URL=http://localhost:8000

# frontend/.env
VITE_API_URL=http://localhost:3001

# See CONFIG.md for detailed port configuration
```

**Dependencies not installing?**
```bash
# Clear caches and reinstall
cd frontend && rm -rf node_modules package-lock.json && npm install
cd ../backend && rm -rf node_modules package-lock.json && npm install
```

**Environment files missing?**
```bash
# Copy example files
cp python-api/.env.example python-api/.env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

## 🎯 Next Steps

- Customize the UI in `frontend/src/App.tsx`
- Modify image generation in `python-api/main.py`
- Add new API endpoints in `backend/server.js`
- Explore shadcn/ui components in `frontend/src/components/ui/`
