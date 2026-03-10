"""
src/recommendations/catalogue.py
─────────────────────────────────
Mock store catalogue.
In production, replace with a real DB query / product API call.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CatalogueItem:
    id:        str
    name:      str
    category:  str
    price:     str
    image_url: Optional[str] = None


CATALOGUE = [
    # ── Tops ──────────────────────────────────────────────────────────────
    CatalogueItem("t001", "Classic White Tee",           "short_sleeve_top",      "€19.99"),
    CatalogueItem("t002", "Striped Linen Tee",           "short_sleeve_top",      "€24.99"),
    CatalogueItem("t003", "Relaxed Fit Tee",             "short_sleeve_top",      "€17.99"),
    CatalogueItem("t004", "Oxford Button-Down",          "long_sleeve_top",       "€39.99"),
    CatalogueItem("t005", "Knit Ribbed Top",             "long_sleeve_top",       "€29.99"),
    CatalogueItem("t006", "Floral Wrap Blouse",          "long_sleeve_top",       "€34.99"),
    CatalogueItem("t007", "Minimal Turtleneck",          "long_sleeve_top",       "€32.99"),
    CatalogueItem("t008", "Oversized Graphic Tee",       "short_sleeve_top",      "€22.99"),

    # ── Outerwear ─────────────────────────────────────────────────────────
    CatalogueItem("o001", "Trench Coat",                 "long_sleeve_outwear",   "€89.99"),
    CatalogueItem("o002", "Denim Jacket",                "short_sleeve_outwear",  "€59.99"),
    CatalogueItem("o003", "Wool Blazer",                 "long_sleeve_outwear",   "€119.99"),
    CatalogueItem("o004", "Bomber Jacket",               "short_sleeve_outwear",  "€79.99"),
    CatalogueItem("o005", "Oversized Parka",             "long_sleeve_outwear",   "€149.99"),
    CatalogueItem("o006", "Cropped Leather Jacket",      "short_sleeve_outwear",  "€199.99"),

    # ── Vests / Slings ────────────────────────────────────────────────────
    CatalogueItem("v001", "Padded Gilet",                "vest",                  "€49.99"),
    CatalogueItem("v002", "Tailored Waistcoat",          "vest",                  "€44.99"),
    CatalogueItem("s001", "Silk Slip Cami",              "sling",                 "€27.99"),
    CatalogueItem("s002", "Ruched Satin Cami",           "sling",                 "€31.99"),

    # ── Bottoms ───────────────────────────────────────────────────────────
    CatalogueItem("b001", "Slim Chinos",                 "trousers",              "€49.99"),
    CatalogueItem("b002", "Wide-leg Linen Trousers",     "trousers",              "€54.99"),
    CatalogueItem("b003", "High-waist Straight Jeans",   "trousers",              "€64.99"),
    CatalogueItem("b004", "Cargo Trousers",              "trousers",              "€57.99"),
    CatalogueItem("b005", "Tailored Shorts",             "shorts",                "€34.99"),
    CatalogueItem("b006", "Denim Shorts",                "shorts",                "€39.99"),
    CatalogueItem("b007", "Bermuda Shorts",              "shorts",                "€29.99"),
    CatalogueItem("b008", "Midi Floral Skirt",           "skirt",                 "€42.99"),
    CatalogueItem("b009", "Pleated Mini Skirt",          "skirt",                 "€37.99"),
    CatalogueItem("b010", "Satin Bias-Cut Skirt",        "skirt",                 "€55.99"),

    # ── Dresses ───────────────────────────────────────────────────────────
    CatalogueItem("d001", "Sundress",                    "short_sleeve_dress",    "€49.99"),
    CatalogueItem("d002", "Wrap Dress",                  "short_sleeve_dress",    "€59.99"),
    CatalogueItem("d003", "Shirt Dress",                 "long_sleeve_dress",     "€67.99"),
    CatalogueItem("d004", "Knit Midi Dress",             "long_sleeve_dress",     "€72.99"),
    CatalogueItem("d005", "Smocked Maxi Dress",          "long_sleeve_dress",     "€84.99"),
    CatalogueItem("d006", "Pinafore Dress",              "vest_dress",            "€52.99"),
    CatalogueItem("d007", "Denim Pinafore",              "vest_dress",            "€58.99"),
    CatalogueItem("d008", "Slip Dress",                  "sling_dress",           "€61.99"),
    CatalogueItem("d009", "Asymmetric Sling Dress",      "sling_dress",           "€74.99"),
]
