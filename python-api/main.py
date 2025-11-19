"""
Entry point for the Stable Diffusion API server.

Run with: python main.py
Or with uvicorn: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import uvicorn
from app.core.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False  # Set to True for development
    )
