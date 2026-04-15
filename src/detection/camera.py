"""
src/detection/camera.py
───────────────────────
Real-time camera pipeline with a 3-state UX flow:

  READY      → camera connection is open, waiting for a client start/retry command
  CAPTURING  → streams raw frames for CAPTURE_SECONDS (default 5s)
               accumulates detections across all frames
  ANALYSING  → briefly shown while recommendations are computed
  RESULTS    → sends final detections + recommendations, then waits
               for a client command:
                 { "cmd": "start" }      → begin the first capture cycle
                 { "cmd": "retry" }      → restart capture
                 { "cmd": "more_recs" }  → keep same detections, get new recs

WebSocket message types sent to client:
  { "type": "ready",   "message": "camera_ready" }
  { "type": "frame",   "frame": <b64>, "countdown": int, "phase": "capturing" }
  { "type": "frame",   "frame": <b64>, "countdown": 0,   "phase": "analysing" }
  { "type": "results", "detections": [...], "recommendations": [...] }
  { "type": "error",   "message": "..." }
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src.detection.detector import BaseDetector, Detection, DetectionResult
from src.recommendations.engine import RecommendationEngine


# ── Constants ──────────────────────────────────────────────────────────────
CAPTURE_SECONDS = 5      # how long to scan the user
FRAME_INTERVAL  = 0.08   # ~12 fps — feels live, light on CPU


class CameraStream:
    """
    Manages a camera capture loop with a controlled UX state machine.
    One instance is shared across WebSocket connections via main.py.
    """

    def __init__(
        self,
        detector:    BaseDetector,
        recommender: RecommendationEngine,
        recommendation_resolver: Optional[Callable[[List[str]], List[dict]]] = None,
        source:      int | str = 0,
        width:       int = 1280,
        height:      int = 720,
    ):
        self.detector    = detector
        self.recommender = recommender
        self.recommendation_resolver = recommendation_resolver
        self.source      = source
        self.width       = width
        self.height      = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_warm = False

    # ── Public API ─────────────────────────────────────────────────────────

    async def run_session(self, send, receive):
        """
        Full session loop for one WebSocket connection.

        send    : async callable — sends a JSON string to the client
        receive : async callable — returns next raw client message string
        """
        self._open()
        try:
            await send(json.dumps({"type": "ready", "message": "camera_ready"}))
            started = False
            while True:
                if not started:
                    while True:
                        await self._send_preview_frame(send)
                        cmd = await self._wait_for_command(receive, timeout=FRAME_INTERVAL)
                        if cmd == "start":
                            started = True
                            break
                        if cmd == "disconnect":
                            return

                # Phase 1 — capture & accumulate detections
                accumulated, last_frame = await self._phase_capture(send)

                # Phase 2 — brief "analysing" pause
                await self._phase_analyse(send)

                # Phase 3 — filter accumulated by current conf threshold, send results
                threshold = self.detector.conf_thres
                filtered  = {
                    cat: conf for cat, conf in accumulated.items()
                    if conf >= threshold
                }
                dominant_cats = self._dominant_categories(filtered)
                recs = self._resolve_recommendations(dominant_cats)
                final_detections, final_b64 = self._build_final_results_frame(last_frame, filtered)

                await send(json.dumps({
                    "type": "results",
                    "detections": [
                        {
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": round(det.confidence, 3),
                            "bbox": det.bbox,
                        }
                        for det in final_detections
                    ],
                    "dominant":        dominant_cats,
                    "recommendations": recs,
                    "annotated_frame": final_b64,
                }))

                # Wait for user action
                cmd = await self._wait_for_command(receive)

                if cmd == "retry":
                    continue                  # restart full capture cycle

                elif cmd == "more_recs":
                    # Keep same outfit detections, cycle through new recs
                    while True:
                        new_recs = self._resolve_recommendations(dominant_cats)
                        await send(json.dumps({
                            "type": "results",
                            "detections": [
                                {"class_name": cat, "confidence": round(conf, 3)}
                                for cat, conf in filtered.items()
                            ],
                            "dominant":        dominant_cats,
                            "recommendations": new_recs,
                        }))
                        inner_cmd = await self._wait_for_command(receive)
                        if inner_cmd == "retry":
                            break           # break inner → restart capture
                        elif inner_cmd == "more_recs":
                            continue        # get yet more recs
                        else:
                            return          # disconnected
                    continue                # restart capture after retry

                else:
                    return                  # disconnected / unknown

        except Exception as e:
            try:
                await send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass

    def run_local(self):
        """Blocking OpenCV window for local testing without WebSocket."""
        self._open()
        print("Camera open — press Q to quit")
        try:
            while True:
                frame = self._read_frame()
                if frame is None:
                    continue
                result    = self.detector.detect(frame)
                annotated = self.detector.draw(frame, result)
                cv2.imshow("FashionSense", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._close()
            cv2.destroyAllWindows()

    # ── Phases ─────────────────────────────────────────────────────────────

    def warmup(self, frames: int = 8):
        """Open the camera early and discard a few frames to reduce first-scan latency."""
        self._open()
        for _ in range(max(frames, 1)):
            frame = self._read_frame()
            if frame is not None:
                self._is_warm = True

    def shutdown(self):
        self._close()

    async def _phase_capture(self, send) -> tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Stream annotated frames for CAPTURE_SECONDS.
        Returns averaged confidence per detected class plus the last raw frame.
        """
        t_start     = time.perf_counter()
        conf_totals: Dict[str, float] = {}
        conf_counts: Dict[str, int]   = {}
        last_frame: Optional[np.ndarray] = None

        while True:
            elapsed   = time.perf_counter() - t_start
            remaining = max(0.0, CAPTURE_SECONDS - elapsed)

            frame = self._read_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            last_frame = frame.copy()

            result    = self.detector.detect(frame)
            annotated = self._draw_capture_overlay(frame, result, int(remaining) + 1)

            # Accumulate
            for det in result.detections:
                conf_totals[det.class_name] = conf_totals.get(det.class_name, 0.0) + det.confidence
                conf_counts[det.class_name] = conf_counts.get(det.class_name, 0)   + 1

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64    = base64.b64encode(buf).decode("utf-8")

            is_last = elapsed >= CAPTURE_SECONDS
            await send(json.dumps({
                "type":      "frame",
                "phase":     "capturing",
                "frame":     b64,
                "countdown": int(remaining) + 1,
                "fps":       round(1000 / max(result.inference_ms, 1), 1),
                "flash":     is_last,   # frontend triggers camera flash on this frame
            }))

            if is_last:
                break

            await asyncio.sleep(FRAME_INTERVAL)

        return (
            {
                cat: conf_totals[cat] / conf_counts[cat]
                for cat in conf_totals
            },
            last_frame,
        )

    async def _phase_analyse(self, send):
        """Flash an 'Analysing' overlay while recommendations are generated."""
        frame = self._read_frame()
        if frame is not None:
            h, w  = frame.shape[:2]
            dark  = frame.copy()
            cv2.rectangle(dark, (0, 0), (w, h), (10, 10, 20), -1)
            out   = cv2.addWeighted(dark, 0.55, frame, 0.45, 0)

            label = "ANALYSING YOUR OUTFIT..."
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(out, label, ((w - tw) // 2, (h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (232, 255, 71), 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64    = base64.b64encode(buf).decode("utf-8")
            await send(json.dumps({"type": "frame", "phase": "analysing", "frame": b64, "countdown": 0}))

        await asyncio.sleep(0.5)

    async def _send_preview_frame(self, send):
        """Stream a raw preview frame while the app is waiting for the user to start scanning."""
        frame = self._read_frame()
        if frame is None:
            await asyncio.sleep(0.01)
            return

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf).decode("utf-8")
        await send(json.dumps({
            "type": "frame",
            "phase": "preview",
            "frame": b64,
            "countdown": 0,
            "flash": False,
        }))

    def _build_final_results_frame(
        self,
        frame: Optional[np.ndarray],
        filtered: Dict[str, float],
    ) -> tuple[List[Detection], Optional[str]]:
        if frame is None:
            return [], None

        result = self.detector.detect(frame)
        kept = [
            det for det in result.detections
            if det.class_name in filtered and det.confidence >= self.detector.conf_thres
        ]
        if not kept:
            return [], None

        annotated = self.detector.draw(
            frame,
            DetectionResult(
                detections=kept,
                frame_shape=frame.shape,
                inference_ms=result.inference_ms,
                timestamp=result.timestamp,
            ),
        )
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return kept, base64.b64encode(buf).decode("utf-8")

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _wait_for_command(self, receive, timeout: Optional[float] = 120) -> str:
        """Wait for a client command and return its cmd value."""
        try:
            if timeout is None:
                msg = await receive()
            else:
                msg = await asyncio.wait_for(receive(), timeout=timeout)
            if msg is None:
                return "disconnect"
            data = json.loads(msg) if isinstance(msg, str) else msg
            return data.get("cmd", "disconnect")
        except asyncio.TimeoutError:
            return "timeout"
        except Exception:
            return "disconnect"

    def _dominant_categories(self, accumulated: Dict[str, float]) -> List[str]:
        """Top 5 categories by average confidence."""
        return [
            cat for cat, _ in
            sorted(accumulated.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _resolve_recommendations(self, categories: List[str]) -> List[dict]:
        if self.recommendation_resolver is not None:
            try:
                resolved = self.recommendation_resolver(categories)
                if isinstance(resolved, list):
                    return resolved
            except Exception as exc:
                print(f"Warning: DB recommendation resolver failed: {exc}")
        return self.recommender.recommend(categories)

    def _draw_capture_overlay(
        self, frame: np.ndarray, result: DetectionResult, countdown: int
    ) -> np.ndarray:
        out  = self.detector.draw(frame, result)
        h, w = out.shape[:2]

        # Countdown circle — top right
        cx, cy, r = w - 60, 60, 44
        cv2.circle(out, (cx, cy), r, (20, 20, 30), -1)
        cv2.circle(out, (cx, cy), r, (232, 255, 71), 3)
        cnt_str = str(countdown)
        (tw, th), _ = cv2.getTextSize(cnt_str, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
        cv2.putText(out, cnt_str, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (232, 255, 71), 3, cv2.LINE_AA)

        # Bottom label
        label = "SCANNING YOUR OUTFIT"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(out, label, ((w - lw) // 2, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (232, 255, 71), 2, cv2.LINE_AA)
        return out

    def _open(self):
        if self._cap is not None and self._cap.isOpened():
            return

        backend = cv2.CAP_ANY
        if os.name == "nt" and isinstance(self.source, int):
            backend = cv2.CAP_DSHOW

        self._cap = cv2.VideoCapture(self.source, backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Cannot open camera source: {self.source}")

    def _close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            self._is_warm = False

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None
