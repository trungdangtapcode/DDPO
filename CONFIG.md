# Configuration Guide

This project uses environment variables for configuration. Each service has its own `.env` file.

## Port Configuration

### Default Ports
- **Python API**: `8000`
- **Node.js Backend**: `3001` (changed from 3000 to avoid conflicts)
- **Frontend**: `5173`

## Environment Files

### Python API (`python-api/.env`)

```bash
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

**Variables:**
- `HOST`: Server host address (0.0.0.0 for all interfaces)
- `PORT`: Port for Python FastAPI service
- `WORKERS`: Number of Uvicorn workers

### Node.js Backend (`backend/.env`)

```bash
PORT=3001
PYTHON_API_URL=http://localhost:8000
NODE_ENV=development
```

**Variables:**
- `PORT`: Port for Express server
- `PYTHON_API_URL`: URL of Python API service
- `NODE_ENV`: Environment (development/production)

### Frontend (`frontend/.env`)

```bash
VITE_API_URL=http://localhost:3001
```

**Variables:**
- `VITE_API_URL`: URL of Node.js backend (used by Vite proxy)

## How to Change Ports

### To change Python API port:

1. Edit `python-api/.env`:
```bash
PORT=8001  # Change to your desired port
```

2. Update `backend/.env`:
```bash
PYTHON_API_URL=http://localhost:8001
```

3. Restart Python API

### To change Backend port:

1. Edit `backend/.env`:
```bash
PORT=3002  # Change to your desired port
```

2. Update `frontend/.env`:
```bash
VITE_API_URL=http://localhost:3002
```

3. Update `frontend/vite.config.ts`:
```typescript
proxy: {
  '/api': {
    target: 'http://localhost:3002',
    changeOrigin: true,
  },
}
```

4. Restart backend and frontend

### To change Frontend port:

1. Edit `frontend/vite.config.ts`:
```typescript
server: {
  port: 5174,  // Change to your desired port
}
```

2. Restart frontend

## Production Configuration

### Python API

For production, use environment variables:

```bash
# .env.production
HOST=0.0.0.0
PORT=8000
WORKERS=4  # Increase workers for production
```

Run with:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Node.js Backend

```bash
# .env.production
PORT=3001
PYTHON_API_URL=https://your-python-api-domain.com
NODE_ENV=production
```

Run with:
```bash
NODE_ENV=production npm start
```

### Frontend

Build with production API URL:

```bash
# .env.production
VITE_API_URL=https://your-backend-domain.com
```

Build:
```bash
npm run build
```

## Docker Configuration

If using Docker, map ports in `docker-compose.yml`:

```yaml
version: '3.8'

services:
  python-api:
    build: ./python-api
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000

  backend:
    build: ./backend
    ports:
      - "3001:3001"
    environment:
      - PORT=3001
      - PYTHON_API_URL=http://python-api:8000

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://backend:3001
```

## Troubleshooting

### Port Already in Use

**Error**: `EADDRINUSE: address already in use`

**Solution**:
```bash
# Find process using the port
lsof -i:3001  # Replace with your port

# Kill the process
kill -9 <PID>

# Or change port in .env file
```

### Can't Connect Between Services

**Check**:
1. All services are running
2. Ports match in configuration files
3. Firewall allows the ports
4. No proxy/VPN blocking local connections

### Environment Variables Not Loading

**Solution**:
```bash
# Make sure .env files exist
ls python-api/.env
ls backend/.env
ls frontend/.env

# If missing, copy from examples
cp python-api/.env.example python-api/.env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

## Security Notes

### Development
- `.env` files are in `.gitignore`
- Use `.env.example` for reference
- Never commit actual `.env` files

### Production
- Use environment variables from hosting platform
- Don't use `.env` files in production
- Use secrets management (AWS Secrets Manager, etc.)

## Quick Reference

| Service | Config File | Port | URL |
|---------|-------------|------|-----|
| Python API | `python-api/.env` | 8000 | http://localhost:8000 |
| Backend | `backend/.env` | 3001 | http://localhost:3001 |
| Frontend | `frontend/.env` | 5173 | http://localhost:5173 |

## Example: Custom Port Setup

If you need all custom ports (e.g., avoiding conflicts):

**1. Python API on 8888:**
```bash
# python-api/.env
PORT=8888
```

**2. Backend on 4000:**
```bash
# backend/.env
PORT=4000
PYTHON_API_URL=http://localhost:8888
```

**3. Frontend on 3000:**
```bash
# frontend/.env
VITE_API_URL=http://localhost:4000
```

```typescript
// frontend/vite.config.ts
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:4000',
      changeOrigin: true,
    },
  },
}
```

Then restart all services!
