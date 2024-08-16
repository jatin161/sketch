from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import io
import numpy as np

app = FastAPI()

# Define CORS origins
origins = [
    "https://cine-sense.netlify.app", "http://localhost:3000/"
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Basic sketch creation function
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

# Advanced sketch creation with effects
def create_sketch_with_effects(image_bytes: bytes,
                               brightness: float,
                               sepia: bool,
                               sepia_intensity: float,
                               vignette: bool,
                               vignette_intensity: float,
                               sharpen: bool,
                               sharpen_intensity: float,
                               sketch_style: str,
                               border: bool,
                               border_size: int,
                               border_color: tuple,
                               frame: bool,
                               frame_type: str) -> bytes:
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image cannot be loaded.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    invert = cv2.bitwise_not(gray)

    # Adjust blur intensity based on sketch style
    blur_intensity = {
        'detailed': (111, 111),
        'rough': (61, 61),
        'abstract': (151, 151)
    }.get(sketch_style, (111, 111))

    # Apply Gaussian blur to the inverted image
    blur = cv2.GaussianBlur(invert, blur_intensity, 0)

    # Invert the blurred image
    invert = cv2.bitwise_not(blur)

    # Create the sketch by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray, invert, scale=256)

    # Adjust the brightness of the sketch
    sketch = cv2.convertScaleAbs(sketch, alpha=brightness, beta=0)

    # Convert grayscale sketch to BGR for further processing
    sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # Apply additional filters if selected
    if sepia:
        sketch_bgr = apply_sepia(sketch_bgr, intensity=sepia_intensity)
    if vignette:
        sketch_bgr = apply_vignette(sketch_bgr, intensity=vignette_intensity)
    if sharpen:
        sketch_bgr = apply_sharpen(sketch_bgr, intensity=sharpen_intensity)

    # Add border if selected
    if border:
        sketch_bgr = add_border(sketch_bgr, border_size=border_size, color=border_color)

    # Add frame if selected
    if frame:
        sketch_bgr = add_frame(sketch_bgr, frame_type=frame_type)

    # Encode the processed image back to bytes
    _, buffer = cv2.imencode('.jpg', sketch_bgr)
    return io.BytesIO(buffer).getvalue()



def apply_sepia(image, intensity=1.0):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter * intensity)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image

def apply_vignette(image, intensity=1.0):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette_image = np.copy(image)
    for i in range(3):
        vignette_image[:, :, i] = vignette_image[:, :, i] * mask * intensity
    return vignette_image

def apply_sharpen(image, intensity=1.0):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]]) * intensity
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def add_border(image, border_size=10, color=(0, 0, 0)):
    bordered_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT, value=color)
    return bordered_image

def add_frame(image, frame_type='classic'):
    if frame_type == 'classic':
        frame_color = (165, 42, 42)
    elif frame_type == 'modern':
        frame_color = (255, 255, 255)
    elif frame_type == 'vintage':
        frame_color = (105, 105, 105)
    else:
        frame_color = (0, 0, 0)

    frame_size = int(min(image.shape[:2]) * 0.1)
    framed_image = add_border(image, border_size=frame_size, color=frame_color)
    return framed_image

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/create_sketch_basic/")
async def create_sketch_basic_endpoint(file: UploadFile = File(...)):
    try:
        # Read the image bytes from the uploaded file
        image_bytes = await file.read()

        # Process the image bytes to create a basic sketch
        sketch_bytes = create_sketch(image_bytes)

        return Response(content=sketch_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/create_sketch_with_effects/")
async def create_sketch_with_effects_endpoint(file: UploadFile = File(...),
                                              brightness: float,
                                              sepia: bool, sepia_intensity: float,
                                              vignette: bool, vignette_intensity: float,
                                              sharpen: bool, sharpen_intensity: float,
                                              sketch_style: str,
                                              border: bool, border_size: int, border_color: str,
                                              frame: bool, frame_type: str):
    try:
        # Convert the string border_color to a tuple
        border_color_tuple = tuple(map(int, border_color.split(',')))

        # Read the image bytes from the uploaded file
        image_bytes = await file.read()

        # Process the image bytes to create a sketch with effects
        sketch_bytes = create_sketch_with_effects(image_bytes, brightness=brightness, sepia=sepia,
                                                  sepia_intensity=sepia_intensity, vignette=vignette,
                                                  vignette_intensity=vignette_intensity, sharpen=sharpen,
                                                  sharpen_intensity=sharpen_intensity, sketch_style=sketch_style,
                                                  border=border, border_size=border_size, border_color=border_color_tuple,
                                                  frame=frame, frame_type=frame_type)

        return Response(content=sketch_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

