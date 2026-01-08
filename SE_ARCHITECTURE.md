# Diffusion Model - Text to Image Generation

A full-stack application for real-time streaming Stable Diffusion image generation with **4 DDPO-optimized models** and a modern tech stack.

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────────┐
│   ViteJS    │ ───> │   Node.js    │ ───> │   Python FastAPI    │
│  (Frontend) │ <─── │   (Backend)  │ <─── │ Stable Diffusion SD │
│  React + TS │      │   Express    │      │   4 DDPO Models     │
└─────────────┘      └──────────────┘      └─────────────────────┘
```

**Frontend**: ViteJS + React + TypeScript + Tailwind CSS + shadcn/ui  
**Backend**: Node.js + Express (Proxy/Gateway)  
**API Service**: Python + FastAPI + Stable Diffusion + DDPO Models

## ✨ Features

- 🎨 **Real Stable Diffusion** image generation with 4 optimized models
- 🎯 **Model Selection**: Choose between Aesthetic, Alignment, Compressibility, and Incompressibility
- 📊 Live progress tracking with step-by-step updates
- 🔄 Progressive image reveal (real diffusion denoising)
- 💅 Modern UI with Tailwind CSS and shadcn/ui components
- 🌊 Server-Sent Events (SSE) for real-time streaming
- 🎯 TypeScript for type safety
- 🚀 Fast development with Vite
- 🔧 CUDA-optimized with memory management

## 🤖 Available Models

1. **Aesthetic Quality** - Optimized for visual appeal
2. **Text Alignment** - Optimized for prompt accuracy
3. **Compressibility** - Optimized for smaller file sizes
4. **Incompressibility** - Optimized for maximum detail

All models are loaded simultaneously (~10GB VRAM total) for instant switching.

## 📁 Project Structure

```
dif/
├── frontend/                 # ViteJS + React + TypeScript
│   ├── src/
│   │   ├── components/ui/   # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── progress.tsx
│   │   ├── lib/
│   │   │   └── utils.ts     # Utility functions
│   │   ├── App.tsx          # Main app component
│   │   ├── main.tsx         # Entry point
│   │   └── index.css        # Tailwind styles
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── postcss.config.js
│
├── backend/                  # Node.js + Express
│   ├── server.js            # Express server with SSE proxy
│   └── package.json
│
├── python-api/              # FastAPI service
│   ├── main.py              # Mock diffusion API
│   └── requirements.txt
│
└── STREAM_DIFFUSION.md      # Technical documentation
```

## 🚀 Quick Start

### Prerequisites

- Node.js (v18+)
- Python (3.8+)
- npm or yarn

### 1. Setup Python API Service

```bash
cd python-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The Python API will run on `http://localhost:8000`

### 2. Setup Node.js Backend

```bash
cd backend
npm install
npm start
```

The Node.js backend will run on `http://localhost:3001`

### 3. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:5173`

### 4. Open the App

Visit `http://localhost:5173` in your browser!

## 🎯 Usage

1. **Select a Model**: Choose from 4 DDPO-optimized models (Aesthetic, Alignment, Compressibility, Incompressibility)
2. **Enter a Prompt**: Type your image description in the input field
3. **Click Generate**: Watch real Stable Diffusion in real-time
4. **Observe Progress**: See the image evolve through actual denoising steps
5. **View Result**: The final high-quality image appears when complete
6. **Compare Models**: Try the same prompt with different models to see variations!

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick setup guide
- **[MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md)** - Multi-model selection guide
- **[REAL_MODEL_GUIDE.md](REAL_MODEL_GUIDE.md)** - Real Stable Diffusion setup
- **[CONFIG.md](CONFIG.md)** - Configuration options
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture details

## 🔧 Development

### Frontend Development

```bash
cd frontend
npm run dev        # Start dev server
npm run build      # Build for production
npm run preview    # Preview production build
```

### Backend Development

```bash
cd backend
npm run dev        # Start with nodemon (auto-reload)
npm start          # Start normally
```

### Python API Development

```bash
cd python-api
uvicorn main:app --reload  # Auto-reload on changes
```

## 🎨 Tech Stack Details

### Frontend
- **Vite**: Lightning-fast build tool
- **React 18**: UI library
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS
- **shadcn/ui**: High-quality React components
- **Radix UI**: Accessible component primitives
- **Lucide React**: Beautiful icons

### Backend
- **Express**: Web framework
- **Axios**: HTTP client
- **CORS**: Cross-origin support

### Python API
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models
- **Stable Diffusion**: Text-to-image generation
- **DDPO Models**: 4 optimized variants (kvablack)

## 📡 API Endpoints

### Python API (Port 8000)

- `GET /` - API info and loaded models
- `GET /generate?prompt={prompt}&steps={steps}&model={model}` - Generate image (SSE stream)
  - `model`: aesthetic | alignment | compressibility | incompressibility
- `GET /health` - Health check with loaded models list

### Node.js Backend (Port 3001)

- `GET /` - Backend info
- `GET /api/generate?prompt={prompt}&steps={steps}&model={model}` - Proxy to Python API
- `GET /api/health` - Health check (includes Python API status)

## 🔄 How It Works

1. **Model Loading**: All 4 DDPO models load at Python API startup (~10GB VRAM)
2. **User Input**: User selects model and enters prompt in React frontend
3. **SSE Connection**: Frontend establishes Server-Sent Events connection
4. **Backend Proxy**: Node.js backend proxies request to Python API with model parameter
5. **Image Generation**: Python FastAPI uses selected Stable Diffusion model
6. **Streaming**: Each denoising step (every 3rd) is sent back through SSE stream
7. **Progressive Display**: Frontend updates image in real-time with JPEG base64
8. **Completion**: Final high-quality image is displayed with model info

## 🎭 Mock Diffusion Process

The mock diffusion simulates a real diffusion model by:

1. **Early Steps (0-30%)**: High noise, random colors
2. **Middle Steps (30-70%)**: Noise reduces, shapes emerge
3. **Late Steps (70-100%)**: Clear image with prompt text

Each step produces a progressively clearer image, mimicking the denoising process of real diffusion models like Stable Diffusion.

## 📦 Environment Variables

### Frontend (`.env`)
```bash
VITE_API_URL=http://localhost:3001
```

### Backend (`.env`)
```bash
PORT=3001
PYTHON_API_URL=http://localhost:8000
NODE_ENV=development
```

### Python API (`.env`)
```bash
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

**See `CONFIG.md` for detailed configuration guide.**

## 🐛 Troubleshooting

**Issue**: Frontend can't connect to backend  
**Solution**: Ensure backend is running on port 3000

**Issue**: Backend can't reach Python API  
**Solution**: Check Python API is running on port 8000

**Issue**: TypeScript errors in frontend  
**Solution**: Run `npm install` to ensure all dependencies are installed

**Issue**: Images not displaying  
**Solution**: Check browser console for CORS errors

## 📝 License

MIT License - Feel free to use this project for learning and development!

## 🤝 Contributing

This is a demonstration project. Feel free to fork and customize!

## 🌟 Features to Add

- [ ] Multiple image generation styles
- [ ] Save/download generated images
- [ ] Generation history
- [ ] Authentication
- [ ] Real Stable Diffusion integration
- [ ] Batch generation
- [ ] Advanced parameters (steps, guidance, etc.)

## 📚 Learn More

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn/ui](https://ui.shadcn.com/)

---

Built with ❤️ using ViteJS, Node.js, and Python
