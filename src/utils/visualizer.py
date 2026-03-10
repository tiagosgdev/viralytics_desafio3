"""
src/utils/visualizer.py
───────────────────────
Utility functions for drawing and visualization beyond the core detector.
"""

import cv2
import numpy as np
from typing import List
from src.detection.detector import DetectionResult, CATEGORY_COLORS


def draw_confidence_histogram(result: DetectionResult) -> np.ndarray:
    """Returns a small BGR image showing per-class confidence bars."""
    W, H = 300, 200
    img = np.zeros((H, W, 3), dtype=np.uint8) + 20

    if not result.detections:
        cv2.putText(img, "No detections", (10, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        return img

    dets = sorted(result.detections, key=lambda d: d.confidence, reverse=True)[:6]
    bar_h = (H - 20) // len(dets)

    for i, det in enumerate(dets):
        y     = 10 + i * bar_h
        bar_w = int((W - 100) * det.confidence)
        color = CATEGORY_COLORS.get(det.class_id, (0,200,0))

        cv2.rectangle(img, (80, y+2), (80 + bar_w, y + bar_h - 4), color, -1)
        label = det.class_name[:14]
        cv2.putText(img, label, (2, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        cv2.putText(img, f"{det.confidence:.0%}", (82 + bar_w, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

    return img


def annotate_for_display(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Full annotation pipeline with semi-transparent boxes."""
    overlay = frame.copy()

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        color = det.color

        # Semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend
    out = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

    # Draw borders and labels on blended image
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        color = det.color

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name.replace('_',' ')}  {det.confidence:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    return out
