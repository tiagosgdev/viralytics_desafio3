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

# Project root = three levels up from src/api/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Priority order for weights:
#   1. MODEL_WEIGHTS env var (set in .env or docker-compose)
#   2. yolov8n_fashion  (nano — trained on this machine)
#   3. yolov8s_fashion  (small — trained on Colab/GPU)
#   4. Base yolov8n.pt  (fallback, no fashion fine-tuning)
def _find_weights() -> str:
    env = os.getenv("MODEL_WEIGHTS")
    if env:
        return env
    candidates = [
        "yolov8n_fashion/weights/best.pt",
        "yolov8s_fashion/weights/best.pt",
        "yolov8m_fashion/weights/best.pt",
    ]
    weights_dir = PROJECT_ROOT / "models" / "weights"
    for c in candidates:
        full = weights_dir / c
        if full.exists():
            print(f"Found trained weights: {full}")
            return str(full)
    print("⚠️  No fine-tuned weights found, using base yolov8n.pt")
    return "yolov8n.pt"   # base fallback

WEIGHTS_PATH = _find_weights()

detector    : FashionDetector    = None
recommender : RecommendationEngine = None
camera      : CameraStream       = None


@app.on_event("startup")
async def startup():
    global detector, recommender, camera

    # Fallback to base YOLOv8 if fine-tuned weights not found yet
    weights = WEIGHTS_PATH  # already resolved by _find_weights()
    print(f"\n🚀  Loading model from: {weights}")

    detector    = FashionDetector(weights=weights, conf_thres=0.60)
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



@app.get("/api/conf")
async def get_conf():
    """Return the current confidence threshold."""
    return {"conf_thres": round(detector.conf_thres, 2)}


@app.post("/api/conf/{value}")
async def set_conf(value: float):
    """
    Update the detection confidence threshold live (0.01 – 0.99).
    Affects both the live camera stream and result filtering.
    """
    from fastapi import HTTPException
    if not (0.01 <= value <= 0.99):
        raise HTTPException(status_code=400, detail="conf must be between 0.01 and 0.99")
    detector.conf_thres = value
    return {"conf_thres": round(detector.conf_thres, 2)}


@app.websocket("/ws/camera")
async def websocket_camera(ws: WebSocket):
    """
    WebSocket endpoint — runs the 3-phase session loop.

    Server → Client message types:
      { type: "frame",   phase: "capturing"|"analysing", frame: <b64>, countdown: int }
      { type: "results", detections: [...], recommendations: [...] }
      { type: "error",   message: str }

    Client → Server commands:
      { cmd: "retry" }      — retake the 3-second scan
      { cmd: "more_recs" }  — keep outfit, get new recommendations
    """
    await ws.accept()
    print("🔌  WebSocket client connected")

    async def send(text: str):
        try:
            await ws.send_text(text)
        except Exception:
            pass

    async def receive():
        try:
            return await ws.receive_text()
        except WebSocketDisconnect:
            return None

    try:
        await camera.run_session(send, receive)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"⚠️  WebSocket error: {e}")
    finally:
        print("🔌  WebSocket client disconnected")