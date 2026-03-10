"""
src/detection/camera.py
───────────────────────
Real-time camera pipeline.

Two modes:
  1. Standalone OpenCV window  → CameraStream.run_local()
  2. Generator for FastAPI WS  → CameraStream.frame_generator()
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Generator, Optional

import cv2
import numpy as np

from src.detection.detector import FashionDetector, DetectionResult
from src.recommendations.engine import RecommendationEngine


class CameraStream:
    """
    Manages a camera capture loop and exposes frames + detections
    for either local display or WebSocket streaming.
    """

    def __init__(
        self,
        detector:   FashionDetector,
        recommender: RecommendationEngine,
        source:     int | str = 0,      # 0 = default webcam; or RTSP url
        width:      int = 1280,
        height:     int = 720,
    ):
        self.detector    = detector
        self.recommender = recommender
        self.source      = source
        self.width       = width
        self.height      = height

        self._cap:    Optional[cv2.VideoCapture] = None
        self._latest_result: Optional[DetectionResult] = None
        self._lock    = threading.Lock()
        self._running = False

    # ── Public ─────────────────────────────────────────────────────────────

    def run_local(self, window_name: str = "FashionSense — press Q to quit"):
        """Blocking loop — shows annotated feed in an OpenCV window."""
        self._open()
        self._running = True
        print(f"\n📷  Camera started (source={self.source})")
        print("   Press  Q  to quit\n")

        try:
            while self._running:
                frame = self._read_frame()
                if frame is None:
                    continue

                result    = self.detector.detect(frame)
                annotated = self.detector.draw(frame, result)

                # Sidebar with recommendations
                annotated = self._draw_recommendations(annotated, result)

                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._close()
            cv2.destroyAllWindows()

    async def frame_generator(self) -> Generator[str, None, None]:
        """
        Async generator for FastAPI WebSocket.
        Yields JSON strings: { "frame": <base64 jpeg>, "detections": [...], "recommendations": [...] }
        """
        self._open()
        self._running = True
        try:
            while self._running:
                frame = self._read_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                result    = self.detector.detect(frame)
                annotated = self.detector.draw(frame, result)

                # Encode frame to JPEG → base64
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64    = base64.b64encode(buf).decode("utf-8")

                # Build payload
                cats  = list({d.class_name for d in result.detections})
                recs  = self.recommender.recommend(cats)

                payload = json.dumps({
                    "frame": b64,
                    "detections": [
                        {
                            "class_id":   d.class_id,
                            "class_name": d.class_name,
                            "confidence": round(d.confidence, 3),
                            "bbox":       d.bbox,
                        }
                        for d in result.detections
                    ],
                    "recommendations": recs,
                    "inference_ms": round(result.inference_ms, 1),
                    "fps": round(1000 / max(result.inference_ms, 1), 1),
                })

                yield payload
                await asyncio.sleep(0)   # yield control to event loop
        finally:
            self._close()

    def stop(self):
        self._running = False

    # ── Private ────────────────────────────────────────────────────────────

    def _open(self):
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")

    def _close(self):
        if self._cap:
            self._cap.release()

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def _draw_recommendations(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Adds a right-hand sidebar with recommendations."""
        H, W = frame.shape[:2]
        sidebar_w = 320
        canvas    = np.zeros((H, W + sidebar_w, 3), dtype=np.uint8)
        canvas[:H, :W] = frame

        # Sidebar background
        canvas[:, W:] = (18, 18, 28)

        y = 30
        cv2.putText(canvas, "DETECTED", (W + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 200), 1)
        y += 24

        cats = list({d.class_name for d in result.detections})
        for cat in cats:
            label = cat.replace("_", " ").title()
            cv2.putText(canvas, f"  • {label}", (W + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1)
            y += 20

        y += 16
        cv2.putText(canvas, "SUGGESTIONS", (W + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 200), 1)
        y += 24

        recs = self.recommender.recommend(cats)
        for rec in recs[:6]:
            name  = rec["name"][:28]
            price = rec.get("price", "")
            cv2.putText(canvas, f"  {name}", (W + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 180, 80), 1)
            y += 16
            if price:
                cv2.putText(canvas, f"     {price}", (W + 16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 180), 1)
                y += 16
            y += 4

        return canvas
