# main.py
from fastapi import FastAPI, Path, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io
import cv2


from airbus_ship_detection import configs
from airbus_ship_detection.inference import load_model, predict_mask
# from airbus_ship_detection.processing import mask_to_png_bytes, draw_contours_on_image

app = FastAPI(
    title="Ship Detection API",
    version="1.0",
    description="""
API for detecting ships in satellite images.

Endpoints:
- `/image/mask`: returns binary ship mask (PNG)
- `/image/contour`: returns original image with yellow contours
"""
)

# Load model once when the app starts
model_name = "UNET_CUSTOM"
fold = 5

model = load_model(model_name=model_name, run_id=fold)

def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert binary mask (0/1) to PNG bytes."""
    mask_img = Image.fromarray((mask * 255).astype('uint8'))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def draw_contours_on_image(image_np: np.ndarray, mask: np.ndarray) -> bytes:
    """Draw bright yellow contours on the input image."""
    contours, _ = cv2.findContours((mask * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image_np.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), thickness=2)  # BGR: yellow
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

@app.get("/")
async def root():
    return {
        "Name": "Airbus Ship Detection",
        "description": "This is a ship detection model for satellite images.",
    }

@app.post("/image/mask", summary="Return binary ship mask")
async def get_ship_mask(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        mask = predict_mask(model, image_np)
        png_bytes = mask_to_png_bytes(mask)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image/contour", summary="Return image with ship contours")
async def get_ship_contour(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = predict_mask(model, image_np)
        png_bytes = draw_contours_on_image(image_np, mask)
        return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))