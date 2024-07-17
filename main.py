from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import requests
import cv2
import io
import numpy as np

app = FastAPI()

# Define CORS origins
origins = [
    "https://cine-sense.netlify.app"
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

def create_sketch(image_bytes: bytes) -> bytes:
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Image cannot be loaded.")
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    invert = cv2.bitwise_not(gray)
    
    # Apply Gaussian blur to the inverted image
    blur = cv2.GaussianBlur(invert, (111, 111), 0)
    
    # Invert the blurred image
    invert = cv2.bitwise_not(blur)
    
    # Create the sketch by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray, invert, scale=256)
    
    # Convert the sketch image to a byte object
    _, buffer = cv2.imencode('.jpg', sketch)
    byte_object = io.BytesIO(buffer).getvalue()
    
    return byte_object

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

@app.get("/create_sketch/")
async def create_sketch_endpoint(image_url: str = Query(..., description="URL of the image")):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        
        # Process the image bytes
        image_bytes = response.content
        sketch_bytes = create_sketch(image_bytes)
        
        return Response(content=sketch_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
