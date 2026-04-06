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
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DetectionResponse,
    HealthResponse,
    SessionResponse,
    SessionStartRequest,
)
from src.api.search_service import UnifiedSearchService
from src.detection.camera import CameraStream
from src.detection.detector import BaseDetector, FashionDetector
from src.detection.yolo_world import YOLOWorldDetector
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

# Priority order for weights (largest/best model first):
#   1. MODEL_WEIGHTS env var (set in .env or docker-compose)
#   2. yolov8l_fashion  (large — best accuracy)
#   3. yolov8m_fashion  (medium)
#   4. yolov8s_fashion  (small)
#   5. yolov8n_fashion  (nano — fastest)
#   6. Base yolov8n.pt  (fallback, no fashion fine-tuning)
def _find_weights() -> str:
    env = os.getenv("MODEL_WEIGHTS")
    if env:
        full = PROJECT_ROOT / "models" / "weights" / env / "weights" / "best.pt"
        return str(full)
    candidates = [
        "yolov8l_fashion/weights/best.pt",
        "yolov8m_fashion/weights/best.pt",
        "yolov8s_fashion/weights/best.pt",
        "yolov8n_fashion/weights/best.pt",
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

detector    : BaseDetector        = None
recommender : RecommendationEngine = None
camera      : CameraStream       = None
whisper_model = None
search_service: UnifiedSearchService = None


DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolov8").lower()


@app.on_event("startup")
async def startup():
    global detector, recommender, camera, whisper_model, search_service

    if DETECTOR_BACKEND == "yolo_world":
        print("\n🚀  Starting with YOLO-World zero-shot detector")
        detector = YOLOWorldDetector(conf_thres=0.15)
    else:
        weights = WEIGHTS_PATH
        print(f"\n🚀  Loading model from: {weights}")
        detector = FashionDetector(weights=weights, conf_thres=0.60)

    recommender = RecommendationEngine(top_k=5)
    search_service = UnifiedSearchService(recommender)
    camera      = CameraStream(detector, recommender, source=0)

    # Load Whisper for voice transcription in a background thread
    # (can be slow on first run — downloads model weights)
    async def _load_whisper():
        global whisper_model
        try:
            from faster_whisper import WhisperModel
            loop = asyncio.get_event_loop()
            print("🎙️  Loading Whisper model (may download on first run)...")
            whisper_model = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: WhisperModel("base", device="cpu", compute_type="int8")
                ),
                timeout=60,
            )
            print("🎙️  Whisper speech-to-text loaded")
        except asyncio.TimeoutError:
            print("⚠️  Whisper loading timed out after 60s — voice input disabled")
        except Exception as e:
            print(f"⚠️  Whisper not available: {e}")

    asyncio.create_task(_load_whisper())

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


@app.post("/api/session/start", response_model=SessionResponse)
async def start_session(payload: SessionStartRequest):
    session = search_service.create_session(
        detected_categories=payload.detected_categories,
        recommendations=payload.recommendations,
    )
    return SessionResponse(
        session_id=session.id,
        mode=session.mode,
        detected_categories=session.detected_categories,
        seed_categories=session.seed_categories,
        active_filters=session.active_filters,
        results=session.last_results,
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    from fastapi import HTTPException

    session = search_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.id,
        mode=session.mode,
        detected_categories=session.detected_categories,
        seed_categories=session.seed_categories,
        active_filters=session.active_filters,
        results=session.last_results,
    )


async def _detect_image_impl(file: UploadFile) -> DetectionResponse:
    """Shared implementation for image/mobile scan uploads."""
    contents = await file.read()
    arr   = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Could not decode image")

    result = detector.detect(frame)
    cats   = list({d.class_name for d in result.detections})
    recs   = recommender.recommend(cats)
    session = search_service.create_session(cats, recs)

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
        session_id      = session.id,
    )


@app.post("/api/search/conversation")
async def search_conversation(payload: ConversationRequest):
    """
    HTTP conversation step for the LNIAGIA search flow.

    The frontend should send `state` from the previous response to keep context.
    """
    try:
        from LNIAGIA.search_app import run_conversation_model
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import conversation search module: {exc}",
        ) from exc

    result = await run_in_threadpool(
        run_conversation_model,
        detected_type=payload.detected_type,
        user_input=payload.message,
        conversation_state=payload.state,
        strict=payload.strict,
    )

    if not result.get("ok", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Conversation search failed."),
        )

    return result


@app.post("/api/search/conversation")
async def search_conversation(payload: ConversationRequest):
    """
    HTTP conversation step for the LNIAGIA search flow.

    The frontend should send `state` from the previous response to keep context.
    """
    try:
        from LNIAGIA.search_app import run_conversation_model
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not import conversation search module: {exc}",
        ) from exc

    result = await run_in_threadpool(
        run_conversation_model,
        detected_type=payload.detected_type,
        user_input=payload.message,
        conversation_state=payload.state,
        strict=payload.strict,
    )

    if not result.get("ok", False):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Conversation search failed."),
        )

    return result
@app.post("/api/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs detection, returns detections + recommendations.
    """
    return await _detect_image_impl(file)


@app.post("/api/mobile/scan", response_model=DetectionResponse)
async def mobile_scan(file: UploadFile = File(...)):
    """
    Mobile-friendly alias for image scan uploads from native clients.
    """
    return await _detect_image_impl(file)



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


# Common Whisper hallucinations on silence/noise
_WHISPER_JUNK = {
    "you", "thank you", "thanks", "thanks for watching",
    "thank you for watching", "bye", "goodbye", "hello",
    "the end", "subtitle", "subtitles", "music",
    "applause", "laughter", "", "...",
}


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Accept an audio file (webm/opus from browser) and return transcribed text."""
    from fastapi import HTTPException

    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    data = await audio.read()
    print(f"🎙️  Received audio: {len(data)} bytes, type: {audio.content_type}")

    # Write browser webm to temp file, then convert to WAV for Whisper
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(data)
        webm_path = tmp.name

    wav_path = webm_path.replace(".webm", ".wav")
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(f"🎙️  ffmpeg stderr: {result.stderr}")
    wav_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
    print(f"🎙️  Converted to WAV: {wav_size} bytes")
    os.unlink(webm_path)

    try:
        segments, info = whisper_model.transcribe(wav_path, beam_size=5, language="en")
        print(f"🎙️  Language: {info.language} ({info.language_probability:.0%}), duration: {info.duration:.1f}s")

        parts = []
        for seg in segments:
            print(f"🎙️  Segment [{seg.start:.1f}s-{seg.end:.1f}s] "
                  f"no_speech={seg.no_speech_prob:.2f}: {seg.text!r}")
            parts.append(seg.text.strip())

        text = " ".join(parts).strip()

        # Filter known hallucinations
        if text.lower().rstrip(".!?, ") in _WHISPER_JUNK:
            print(f"🎙️  Filtered as hallucination: {text!r}")
            text = ""

        print(f"🎙️  Final text: {text!r}")
    finally:
        os.unlink(wav_path)

    return {"text": text}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    from fastapi import HTTPException

    session_id = payload.session_id
    if not session_id:
        session = search_service.create_session(
            detected_categories=payload.detected_categories,
            recommendations=payload.recommendations,
        )
        session_id = session.id

    try:
        result = search_service.refine(
            session_id=session_id,
            message=payload.message,
            history=[msg.model_dump() for msg in payload.history],
            replace_vision=payload.replace_vision,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    return ChatResponse(**result)


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
