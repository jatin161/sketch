from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from oauth2client.service_account import ServiceAccountCredentials
import cv2
import io

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


def create_sketch(image_path):

    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Image not found or cannot be loaded. Please check the file path.")
    
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
    _, buffer = cv2.imencode('.jpg', sketch)  # You can change the file extension if needed
    byte_object = io.BytesIO(buffer).getvalue()
    
    return byte_object



@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

@app.head("/")
async def head_root():
    return

class requestpath(BaseModel):
    path: str
   


@app.get("/create_sketch/{requestpath}")
async def create_sketch(requestpath: requestpath):
    path = requestpath.path
 
    
    if create_sketch(path):
        return {"success": True}
    else:
        raise HTTPException(status_code=401, detail="Unauthorized")