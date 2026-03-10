"""
src/api/schemas.py
──────────────────
Pydantic request / response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


class DetectionItem(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[int]   # [x1, y1, x2, y2]


class RecommendationItem(BaseModel):
    id:        str
    name:      str
    category:  str
    price:     str
    image_url: Optional[str] = None
    reason:    str
    score:     float


class DetectionResponse(BaseModel):
    detections:      List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    inference_ms:    float
    annotated_frame: Optional[str] = None   # base64 JPEG
