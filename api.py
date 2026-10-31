
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from detector import detect_objects

app = FastAPI(
    title="Object Detector API",
    description="API для распознавания объектов на изображениях",
    version="1.0.0",
)

MAX_IMAGE_SIZE_MB = 15


@app.get("/")
def root():
    return {
        "message": "Object Detector API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.70, ge=0.05, le=0.99),
):
    content = await file.read()

    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 15 MB.",
        )

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    detections = detect_objects(image, threshold)

    return {
        "filename": file.filename,
        "total_objects": len(detections),
        "detections": detections,
    }
