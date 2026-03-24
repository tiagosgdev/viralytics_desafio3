# ════════════════════════════════════════════════════════════
# description_generator.py
# Converts JSON clothing items to natural language descriptions
# ════════════════════════════════════════════════════════════

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json
import sys

# Allow importing models.py from the parent DB/ directory
_DB_DIR = Path(__file__).resolve().parent.parent
if str(_DB_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_DIR))

from models import METADATA_EXCLUDE_FIELDS
from nl_mappings import (
    TYPE_NAMES,
    COLOR_DESCRIPTIONS,
    STYLE_DESCRIPTIONS,
    PATTERN_DESCRIPTIONS,
    MATERIAL_DESCRIPTIONS,
    FIT_DESCRIPTIONS,
    SEASON_DESCRIPTIONS,
    OCCASION_DESCRIPTIONS,
    GENDER_DESCRIPTIONS,
    AGE_GROUP_DESCRIPTIONS,
    NECKLINE_DESCRIPTIONS,
    COLLAR_DESCRIPTIONS,
    SLEEVE_STYLE_DESCRIPTIONS,
    HEM_STYLE_DESCRIPTIONS,
    CLOSURE_DESCRIPTIONS,
    HOOD_DESCRIPTIONS,
    INSULATION_DESCRIPTIONS,
    WATERPROOF_DESCRIPTIONS,
    OUTWEAR_POCKET_DESCRIPTIONS,
    WAIST_SIZE_DESCRIPTIONS,
    WAIST_STYLE_DESCRIPTIONS,
    RISE_DESCRIPTIONS,
    LENGTH_DESCRIPTIONS,
    LEG_STYLE_DESCRIPTIONS,
    BOTTOM_POCKET_DESCRIPTIONS,
    DRESS_STYLE_DESCRIPTIONS,
)


class ClothingDescriptionGenerator:
    """
    Generates natural language descriptions from clothing item JSON.

    These descriptions are optimized for embedding models to understand:
    - Semantic meaning of attributes
    - Relationships between fields
    - Natural language synonyms
    """

    def __init__(self, include_synonyms: bool = True, verbose: bool = False):
        """
        Args:
            include_synonyms: Include synonym phrases in parentheses (improves embedding quality).
            verbose: Print debug information while generating.
        """
        self.include_synonyms = include_synonyms
        self.verbose = verbose

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def generate(self, item: Dict) -> str:
        """Generate a natural language description for a single clothing item."""
        parts: List[str] = []

        # ═══ MAIN DESCRIPTION ═══
        type_desc = self._lookup(TYPE_NAMES, item.get("type", ""), fallback_clean=True)
        color_desc = self._lookup(COLOR_DESCRIPTIONS, item.get("color", ""))

        parts.append(f"This is a {color_desc} {type_desc}.")

        # ═══ STYLE & PATTERN ═══
        style_desc = self._lookup(STYLE_DESCRIPTIONS, item.get("style", ""))
        pattern_desc = self._lookup(PATTERN_DESCRIPTIONS, item.get("pattern", ""))

        if style_desc and pattern_desc:
            parts.append(f"The style is {style_desc} with a {pattern_desc} pattern.")
        elif style_desc:
            parts.append(f"The style is {style_desc}.")
        elif pattern_desc:
            parts.append(f"The pattern is {pattern_desc}.")

        # ═══ MATERIAL & FIT ═══
        material_desc = self._lookup(MATERIAL_DESCRIPTIONS, item.get("material", ""))
        fit_desc = self._lookup(FIT_DESCRIPTIONS, item.get("fit", ""))

        if material_desc and fit_desc:
            parts.append(f"Made of {material_desc} with a {fit_desc}.")
        elif material_desc:
            parts.append(f"Made of {material_desc}.")
        elif fit_desc:
            parts.append(f"The fit is {fit_desc}.")

        # ═══ TARGET AUDIENCE ═══
        gender = item.get("gender", "")
        age_group = item.get("age_group", "")
        audience = self._build_audience(gender, age_group)
        if audience:
            parts.append(f"Designed for {audience}.")

        # ═══ SEASON & OCCASION ═══
        season_desc = self._lookup(SEASON_DESCRIPTIONS, item.get("season", ""))
        occasion_desc = self._lookup(OCCASION_DESCRIPTIONS, item.get("occasion", ""))

        if season_desc and occasion_desc:
            parts.append(f"Suitable for {season_desc} and {occasion_desc}.")
        elif season_desc:
            parts.append(f"Suitable for {season_desc}.")
        elif occasion_desc:
            parts.append(f"Suitable for {occasion_desc}.")

        # ═══ BRAND & PRICE ═══
        brand = item.get("brand", "")
        price = item.get("price", 0)

        if brand and price:
            price_tier = self._price_tier(price)
            parts.append(f"Brand: {brand}. Price: €{price:.2f} ({price_tier}).")
        elif brand:
            parts.append(f"Brand: {brand}.")

        # ═══ TYPE-SPECIFIC DETAILS ═══
        type_specific = self._get_type_specific_details(item)
        if type_specific:
            parts.append(type_specific)

        description = " ".join(parts)

        if self.verbose:
            print(f"Generated ({len(description)} chars): {description[:100]}...")

        return description

    def generate_batch(self, items: List[Dict]) -> List[str]:
        """Generate descriptions for a list of items."""
        return [self.generate(item) for item in items]

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _lookup(self, mapping: Dict[str, str], value: str,
                fallback_clean: bool = False) -> str:
        """Look up a value in a mapping dict, optionally with synonyms."""
        if not value:
            return ""
        if self.include_synonyms:
            result = mapping.get(value)
            if result:
                return result
        # Fallback: return the raw value, cleaned if requested
        return value.replace("_", " ") if fallback_clean else value

    def _build_audience(self, gender: str, age_group: str) -> str:
        """Build a human-readable audience string from gender + age_group."""
        gender_desc = self._lookup(GENDER_DESCRIPTIONS, gender) if gender else ""
        if not age_group:
            return gender_desc

        # age_group can be comma-separated ("teenager, young adult")
        age_parts = [a.strip() for a in age_group.split(",")]
        if self.include_synonyms:
            age_descs = [AGE_GROUP_DESCRIPTIONS.get(a, a) for a in age_parts]
        else:
            age_descs = age_parts

        age_str = ", ".join(age_descs)
        if gender_desc:
            return f"{gender_desc} — {age_str}"
        return age_str

    @staticmethod
    def _price_tier(price: float) -> str:
        if price > 200:
            return "premium, high-end, luxury"
        if price > 80:
            return "mid-range"
        if price > 30:
            return "affordable"
        return "budget-friendly, cheap, inexpensive"

    def _get_type_specific_details(self, item: Dict) -> Optional[str]:
        """Build a natural-language features sentence from type-specific fields."""
        details: List[str] = []

        # ═══ TOPS & DRESSES ═══
        if "neckline" in item:
            details.append(self._lookup(NECKLINE_DESCRIPTIONS, item["neckline"]))

        if "collar" in item and item["collar"] != "none":
            details.append(self._lookup(COLLAR_DESCRIPTIONS, item["collar"]))

        if "sleeve_style" in item:
            details.append(self._lookup(SLEEVE_STYLE_DESCRIPTIONS, item["sleeve_style"]))

        if "hem_style" in item:
            details.append(self._lookup(HEM_STYLE_DESCRIPTIONS, item["hem_style"]))

        # ═══ DRESSES ═══
        if "dress_style" in item:
            details.append(self._lookup(DRESS_STYLE_DESCRIPTIONS, item["dress_style"]))

        if "length" in item:
            details.append(self._lookup(LENGTH_DESCRIPTIONS, item["length"]))

        if "waist_style" in item:
            details.append(self._lookup(WAIST_STYLE_DESCRIPTIONS, item["waist_style"]))

        # ═══ OUTWEAR ═══
        if "closure" in item:
            details.append(self._lookup(CLOSURE_DESCRIPTIONS, item["closure"]))

        if "hood" in item and item["hood"] != "none":
            details.append(self._lookup(HOOD_DESCRIPTIONS, item["hood"]))

        if "insulation" in item and item["insulation"] != "none":
            details.append(self._lookup(INSULATION_DESCRIPTIONS, item["insulation"]))

        if "waterproof" in item and item["waterproof"] != "none":
            details.append(self._lookup(WATERPROOF_DESCRIPTIONS, item["waterproof"]))

        # Handle both "outwear_pockets" and "pockets" field names
        pockets = item.get("outwear_pockets", item.get("pockets", "none"))
        if pockets != "none":
            details.append(self._lookup(OUTWEAR_POCKET_DESCRIPTIONS, pockets))

        # ═══ BOTTOMS ═══
        if "waist" in item:
            details.append(f"waist size {self._lookup(WAIST_SIZE_DESCRIPTIONS, item['waist'])}")

        if "rise" in item:
            details.append(self._lookup(RISE_DESCRIPTIONS, item["rise"]))

        if "leg_style" in item:
            details.append(self._lookup(LEG_STYLE_DESCRIPTIONS, item["leg_style"]))

        if "bottom_pockets" in item and item["bottom_pockets"] != "none":
            details.append(self._lookup(BOTTOM_POCKET_DESCRIPTIONS, item["bottom_pockets"]))

        # Filter out empty strings
        details = [d for d in details if d]

        if details:
            return "Features: " + ", ".join(details) + "."
        return None


# ════════════════════════════════════════════════════════════
# FILE I/O: Load source JSON, generate descriptions, save output
# ════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_SOURCES_DIR = _SCRIPT_DIR.parent / "SQLLite" / "DataSources"
_OUTPUT_DIR = _SCRIPT_DIR / "NL_Items_Descriptions"

# Fields excluded from the Qdrant payload — defined in DB/models.py (METADATA_EXCLUDE_FIELDS).
# Everything else in the raw item JSON is stored as metadata for filtering.

def list_available_sources() -> List[Path]:
    """List all JSON files in the DataSources folder."""
    if not _DATA_SOURCES_DIR.exists():
        print(f"DataSources directory not found: {_DATA_SOURCES_DIR}")
        return []
    return sorted(_DATA_SOURCES_DIR.glob("*.json"))


def select_source_interactive() -> Optional[Path]:
    """Prompt the user to pick a JSON source file."""
    sources = list_available_sources()
    if not sources:
        print("No JSON files found in DataSources.")
        return None

    print("\nAvailable data sources:")
    print("-" * 50)
    for idx, src in enumerate(sources, 1):
        size_kb = src.stat().st_size / 1024
        print(f"  [{idx}] {src.name}  ({size_kb:.1f} KB)")
    print("-" * 50)

    while True:
        choice = input(f"Select a file (1-{len(sources)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(sources):
            return sources[int(choice) - 1]
        print("Invalid choice. Try again.")


def generate_and_save(
    source_path: Path,
    include_synonyms: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Load items from a JSON file, generate NL descriptions, and save the output.

    Each output record has the shape:
        {
            "item_id":    <int>,
            "description": "<natural language text>",
            "metadata":   { type, color, pattern, season, occasion, brand, price }
        }

    Returns:
        Path to the saved output file.
    """
    with open(source_path, "r", encoding="utf-8") as f:
        items: List[Dict] = json.load(f)

    print(f"\nLoaded {len(items)} items from {source_path.name}")

    generator = ClothingDescriptionGenerator(
        include_synonyms=include_synonyms, verbose=verbose
    )

    output_items = []
    for item in items:
        output_items.append({
            "item_id": item.get("id"),
            "description": generator.generate(item),
            "metadata": {k: v for k, v in item.items() if k not in METADATA_EXCLUDE_FIELDS},
        })

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _OUTPUT_DIR / f"{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_items, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output_items)} descriptions to {output_path.name}")
    return output_path


# ════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    selected = select_source_interactive()
    if selected:
        out = generate_and_save(selected, include_synonyms=True, verbose=True)
        print(f"\nOutput saved to:\n  {out}")
