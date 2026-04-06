"""
src/api/schemas.py
──────────────────
Pydantic request / response models.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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
    session_id:      Optional[str] = None


class SessionStartRequest(BaseModel):
    detected_categories: List[str]
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class SessionResponse(BaseModel):
    session_id: str
    mode: str
    detected_categories: List[str]
    seed_categories: List[str]
    active_filters: Dict[str, Any]
    results: List[Dict[str, Any]]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = None
    replace_vision: Optional[bool] = None
    detected_categories: List[str] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    mode: str
    active_filters: Dict[str, Any]
    results: List[Dict[str, Any]]
    strict: bool = False
    warning: Optional[str] = None
