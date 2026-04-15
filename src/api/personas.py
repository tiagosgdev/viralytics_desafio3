from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaConfig:
    key: str
    label: str
    vision_backend: str
    text_backend: str


PERSONA_CONFIGS = {
    "cruella": PersonaConfig(
        key="cruella",
        label="Cruella",
        vision_backend="yolo",
        text_backend="llm",
    ),
    "edna": PersonaConfig(
        key="edna",
        label="Edna",
        vision_backend="fashionnet",
        text_backend="custom",
    ),
}


def normalize_persona(value: str | None) -> str:
    if not value:
        return "cruella"
    key = str(value).strip().lower()
    return key if key in PERSONA_CONFIGS else "cruella"
