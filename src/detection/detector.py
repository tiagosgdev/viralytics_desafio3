"""
src/detection/detector.py
─────────────────────────
Detection base class and YOLOv8 inference wrapper.
Returns structured DetectionResult objects used throughout the app.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ── Category definitions ───────────────────────────────────────────────────

CATEGORY_NAMES = {
    0: "short_sleeve_top",
    1: "long_sleeve_top",
    2: "short_sleeve_outwear",
    3: "long_sleeve_outwear",
    4: "vest",
    5: "sling",
    6: "shorts",
    7: "trousers",
    8: "skirt",
    9: "short_sleeve_dress",
    10: "long_sleeve_dress",
    11: "vest_dress",
    12: "sling_dress",
}

CATEGORY_COLORS = {
    0:  (255, 100, 100),   # short_sleeve_top    — coral
    1:  (255, 160,  60),   # long_sleeve_top     — amber
    2:  ( 80, 200, 120),   # short_sleeve_outwear— green
    3:  ( 40, 180, 200),   # long_sleeve_outwear — teal
    4:  (180,  80, 255),   # vest                — violet
    5:  (255,  80, 180),   # sling               — pink
    6:  (100, 180, 255),   # shorts              — sky
    7:  ( 60,  80, 200),   # trousers            — blue
    8:  (220, 180,  40),   # skirt               — gold
    9:  (255, 120,  80),   # short_sleeve_dress  — orange
    10: (140,  60, 200),   # long_sleeve_dress   — purple
    11: (200, 100, 150),   # vest_dress          — mauve
    12: (100, 200, 180),   # sling_dress         — mint
}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id:    int
    class_name:  str
    confidence:  float
    bbox:        List[int]          # [x1, y1, x2, y2] in pixels
    color:       tuple = field(default_factory=lambda: (0, 255, 0))


@dataclass
class DetectionResult:
    detections:     List[Detection]
    frame_shape:    tuple           # (H, W, C)
    inference_ms:   float
    timestamp:      float


# ── Base detector ──────────────────────────────────────────────────────

class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    Subclasses must implement ``detect()``.
    """

    conf_thres: float
    iou_thres:  float

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a BGR frame. Returns DetectionResult."""

    def draw(
        self,
        frame:  np.ndarray,
        result: DetectionResult,
        show_conf: bool = True,
    ) -> np.ndarray:
        """Returns a copy of frame with bounding boxes drawn."""
        out = frame.copy()

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = det.color

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{det.class_name}"
            if show_conf:
                label += f"  {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(
                out, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # FPS / inference time overlay
        fps_text = f"{1000/result.inference_ms:.1f} FPS  ({result.inference_ms:.0f} ms)"
        cv2.putText(out, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)

        return out


# ── YOLOv8 detector ───────────────────────────────────────────────────

class FashionDetector(BaseDetector):
    """
    Wraps YOLOv8 for clothing detection.

    Example
    -------
    detector = FashionDetector("models/weights/.../best.pt")
    result   = detector.detect(frame)          # frame = BGR numpy array
    annotated = detector.draw(frame, result)
    """

    def __init__(
        self,
        weights:    str   = "yolov8s.pt",   # swap for your best.pt after training
        conf_thres: float = 0.60,
        iou_thres:  float = 0.45,
        device:     str   = "",             # "" = auto
        imgsz:      int   = 640,
    ):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self.imgsz      = imgsz

        print(f"🔍  Loading model: {weights}")
        self.model = YOLO(weights)
        if device:
            self.model.to(device)
        print("✅  Model ready")

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a BGR frame. Returns DetectionResult."""
        t0 = time.perf_counter()

        results = self.model.predict(
            source  = frame,
            conf    = self.conf_thres,
            iou     = self.iou_thres,
            imgsz   = self.imgsz,
            verbose = False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        detections   = self._parse(results)

        return DetectionResult(
            detections   = detections,
            frame_shape  = frame.shape,
            inference_ms = inference_ms,
            timestamp    = time.time(),
        )

    # ── Private ────────────────────────────────────────────────────────────

    def _parse(self, results) -> List[Detection]:
        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls.item())
                detections.append(Detection(
                    class_id   = class_id,
                    class_name = CATEGORY_NAMES.get(class_id, f"class_{class_id}"),
                    confidence = float(box.conf.item()),
                    bbox       = [int(v) for v in box.xyxy[0].tolist()],
                    color      = CATEGORY_COLORS.get(class_id, (0, 255, 0)),
                ))
        return detections