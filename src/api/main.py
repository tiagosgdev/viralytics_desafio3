"""
src/api/main.py
───────────────
FastAPI application.

Endpoints:
  GET  /                    → HTML dashboard (serves frontend/index.html)
  GET  /health              → health check
  GET  /api/detect/image    → single image upload detection
  WS   /ws/camera           → real-time WebSocket camera stream

Run:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import DetectionResponse, HealthResponse, RecommendationItem
from src.detection.camera import CameraStream
from src.detection.detector import FashionDetector
from src.recommendations.engine import RecommendationEngine

# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "FashionSense API",
    description = "Real-time clothing detection & recommendation system",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
STATIC_DIR   = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Shared singletons (loaded once at startup) ─────────────────────────────

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS", "models/weights/yolov8s_fashion/weights/best.pt")

detector    : FashionDetector    = None
recommender : RecommendationEngine = None
camera      : CameraStream       = None


@app.on_event("startup")
async def startup():
    global detector, recommender, camera

    # Fallback to base YOLOv8 if fine-tuned weights not found yet
    weights = WEIGHTS_PATH if Path(WEIGHTS_PATH).exists() else "yolov8s.pt"
    print(f"\n🚀  Loading model from: {weights}")

    detector    = FashionDetector(weights=weights, conf_thres=0.40)
    recommender = RecommendationEngine(top_k=5)
    camera      = CameraStream(detector, recommender, source=0)

    print("✅  API ready\n")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>FashionSense API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=detector is not None)


@app.post("/api/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs detection, returns detections + recommendations.
    """
    contents = await file.read()
    arr   = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = detector.detect(frame)
    cats   = list({d.class_name for d in result.detections})
    recs   = recommender.recommend(cats)

    # Encode annotated frame
    annotated = detector.draw(frame, result)
    _, buf    = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_frame = base64.b64encode(buf).decode("utf-8")

    return DetectionResponse(
        detections = [
            {
                "class_id":   d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox":       d.bbox,
            }
            for d in result.detections
        ],
        recommendations = recs,
        inference_ms    = round(result.inference_ms, 1),
        annotated_frame = b64_frame,
    )


@app.websocket("/ws/camera")
async def websocket_camera(ws: WebSocket):
    """
    WebSocket endpoint — streams annotated frames + detections to the frontend.
    Client receives JSON: { frame, detections, recommendations, fps }
    """
    await ws.accept()
    print("🔌  WebSocket client connected")

    try:
        async for payload in camera.frame_generator():
            await ws.send_text(payload)
    except WebSocketDisconnect:
        print("🔌  WebSocket client disconnected")
    except Exception as e:
        print(f"⚠️  WebSocket error: {e}")
    finally:
        camera.stop()
