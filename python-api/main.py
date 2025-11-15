from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import io
import json
import asyncio
import random
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_mock_image(step: int, total_steps: int, prompt: str, width: int = 512, height: int = 512):
    """
    Create a mock image that simulates diffusion process
    Early steps: more noise/blur
    Later steps: clearer image with text
    """
    # Create base image with gradient
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Progress from noisy to clear
    progress = step / total_steps
    
    # Add colored gradient based on progress
    for y in range(height):
        # Color transitions from gray (noisy) to colorful (clear)
        r = int(100 + (155 * progress))
        g = int(120 + (100 * progress * (y / height)))
        b = int(150 + (105 * progress * (1 - y / height)))
        
        # Add noise for early steps
        if progress < 0.7:
            noise_factor = int((1 - progress) * 50)
            r += random.randint(-noise_factor, noise_factor)
            g += random.randint(-noise_factor, noise_factor)
            b += random.randint(-noise_factor, noise_factor)
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add geometric shapes to simulate "content"
    if progress > 0.3:
        # Draw some shapes that become clearer
        opacity = int(progress * 255)
        
        # Circle
        circle_size = int(progress * 100)
        circle_pos = (width // 2 - circle_size // 2, height // 2 - circle_size // 2)
        draw.ellipse(
            [circle_pos, (circle_pos[0] + circle_size, circle_pos[1] + circle_size)],
            outline=(255, 200, 0),
            width=3
        )
        
        # Rectangles
        if progress > 0.5:
            draw.rectangle(
                [50, 50, 150, 150],
                outline=(0, 200, 255),
                width=2
            )
    
    # Add text overlay showing progress and prompt
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add prompt text (becomes clearer over time)
    if progress > 0.4:
        text_alpha = int((progress - 0.4) / 0.6 * 255)
        prompt_display = prompt[:30] + "..." if len(prompt) > 30 else prompt
        draw.text((20, height - 80), f"Prompt: {prompt_display}", fill=(0, 0, 0), font=font)
    
    # Always show step counter
    draw.text((20, height - 40), f"Step {step}/{total_steps}", fill=(0, 0, 0), font=font)
    
    return img

async def generate_image_stream(prompt: str, steps: int = 20):
    """
    Generator that yields intermediate diffusion steps
    """
    for step in range(1, steps + 1):
        # Simulate processing time
        await asyncio.sleep(0.3)
        
        # Generate mock image for this step
        image = create_mock_image(step, steps, prompt)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85 if step < steps else 95)
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Yield as Server-Sent Event
        data = {
            'step': step,
            'total_steps': steps,
            'image': image_base64,
            'progress': (step / steps) * 100,
            'done': step == steps
        }
        
        yield f"data: {json.dumps(data)}\n\n"
    
    # Send completion signal
    yield "data: [DONE]\n\n"

@app.get("/")
async def root():
    return {"message": "Mock Diffusion API", "status": "running"}

@app.get("/generate")
async def generate(prompt: str, steps: int = 20):
    """
    Generate image with streaming intermediate steps
    """
    if not prompt:
        return {"error": "Prompt is required"}
    
    if steps < 5 or steps > 50:
        steps = 20
    
    return StreamingResponse(
        generate_image_stream(prompt, steps),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
