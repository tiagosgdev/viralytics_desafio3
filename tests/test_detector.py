"""
tests/test_detector.py
──────────────────────
Unit tests for FashionDetector (uses base YOLOv8 weights, no fine-tuned model needed).
"""

import numpy as np
import pytest
from src.detection.detector import FashionDetector, Detection, DetectionResult


@pytest.fixture(scope="module")
def detector():
    """Load detector once for all tests in this module."""
    return FashionDetector(weights="yolov8n.pt", conf_thres=0.25)


def make_blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detector_returns_result(detector):
    frame  = make_blank_frame()
    result = detector.detect(frame)
    assert isinstance(result, DetectionResult)


def test_result_has_inference_time(detector):
    frame  = make_blank_frame()
    result = detector.detect(frame)
    assert result.inference_ms > 0


def test_draw_returns_same_shape(detector):
    frame  = make_blank_frame()
    result = detector.detect(frame)
    drawn  = detector.draw(frame, result)
    assert drawn.shape == frame.shape


def test_detections_are_list(detector):
    frame  = make_blank_frame()
    result = detector.detect(frame)
    assert isinstance(result.detections, list)


def test_bbox_within_frame(detector):
    """All bounding boxes should be within frame boundaries."""
    frame  = make_blank_frame(480, 640)
    result = detector.detect(frame)
    H, W   = frame.shape[:2]
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 < W
        assert 0 <= y1 < H
        assert x1 < x2
        assert y1 < y2


def test_confidence_in_range(detector):
    frame  = make_blank_frame()
    result = detector.detect(frame)
    for det in result.detections:
        assert 0.0 <= det.confidence <= 1.0
