"""
src/recommendations/catalogue.py
--------------------------------
JSON-backed store catalogue loader.

This replaces the older hardcoded mock Python list so stores can swap in
their own JSON export later, or use the same column structure from Excel/CSV.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOGUE_PATH = PROJECT_ROOT / "data" / "mock_store_catalogue_template.json"

CORE_METADATA_FIELDS = (
    "color",
    "style",
    "pattern",
    "material",
    "fit",
    "gender",
    "age_group",
    "season",
    "occasion",
    "neckline",
    "collar",
    "sleeve_style",
    "hem_style",
    "closure",
    "hood",
    "insulation",
    "waterproof",
    "outwear_pockets",
    "waist",
    "waist_style",
    "rise",
    "length",
    "leg_style",
    "bottom_pockets",
    "dress_style",
)


@dataclass
class CatalogueItem:
    id: str
    name: str
    category: str
    price: str
    image_url: str | None = None
    brand: str | None = None
    description: str | None = None
    product_url: str | None = None
    sku: str | None = None
    currency: str = "EUR"
    stock_status: str | None = None
    inventory_count: int | None = None
    sizes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], default_currency: str = "EUR") -> "CatalogueItem":
        item_type = str(raw.get("type") or raw.get("category") or "").strip()
        if not item_type:
            raise ValueError(f"Catalogue item {raw.get('id', '<missing>')} is missing type/category")

        raw_price = raw.get("price")
        currency = str(raw.get("currency") or default_currency)
        price = _format_price(raw_price, currency)

        metadata = {
            field_name: raw.get(field_name)
            for field_name in CORE_METADATA_FIELDS
            if raw.get(field_name) is not None
        }

        return cls(
            id=str(raw.get("id") or raw.get("sku") or ""),
            name=str(raw.get("name") or raw.get("title") or item_type.replace("_", " ").title()),
            category=item_type,
            price=price,
            image_url=_string_or_none(raw.get("image_url")),
            brand=_string_or_none(raw.get("brand")),
            description=_string_or_none(raw.get("description")),
            product_url=_string_or_none(raw.get("product_url")),
            sku=_string_or_none(raw.get("sku")),
            currency=currency,
            stock_status=_string_or_none(raw.get("stock_status")),
            inventory_count=_int_or_none(raw.get("inventory_count")),
            sizes=_string_list(raw.get("sizes")),
            tags=_string_list(raw.get("tags")),
            metadata=metadata,
        )

    def to_recommendation(self, reason: str, score: float) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "image_url": self.image_url,
            "reason": reason,
            "score": round(score, 2),
            "brand": self.brand,
            "description": self.description,
            "product_url": self.product_url,
            "sku": self.sku,
            "stock_status": self.stock_status,
            "inventory_count": self.inventory_count,
            "sizes": self.sizes,
            "tags": self.tags,
            "metadata": self.metadata,
        }
        return payload


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_price(value: Any, currency: str) -> str:
    if isinstance(value, (int, float)):
        return f"{currency} {value:.2f}"
    text = _string_or_none(value)
    return text or f"{currency} 0.00"


def get_catalogue_path() -> Path:
    configured = os.getenv("STORE_CATALOGUE_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CATALOGUE_PATH


def load_catalogue(path: Path | None = None) -> list[CatalogueItem]:
    catalogue_path = path or get_catalogue_path()
    if not catalogue_path.exists():
        raise FileNotFoundError(
            f"Catalogue JSON not found at {catalogue_path}. "
            "Set STORE_CATALOGUE_PATH or create the default file."
        )

    with catalogue_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    default_currency = str(data.get("currency") or "EUR")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ValueError(f"Catalogue file {catalogue_path} is missing an 'items' array.")

    items: list[CatalogueItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        items.append(CatalogueItem.from_dict(raw, default_currency=default_currency))

    return items


CATALOGUE = load_catalogue()
