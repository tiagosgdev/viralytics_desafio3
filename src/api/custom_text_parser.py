from __future__ import annotations

import re
from copy import deepcopy

try:
    from LNIAGIA.DB.vector.nl_mappings import ALL_MAPPINGS
except Exception:  # pragma: no cover - optional integration stack
    ALL_MAPPINGS = {}


NEGATION_HINTS = ("no ", "not ", "without ", "avoid ", "don't want ", "do not want ", "exclude ")

TYPE_SYNONYMS = {
    "short_sleeve_top": ("t-shirt", "tee", "short sleeve top"),
    "long_sleeve_top": ("shirt", "blouse", "sweater", "jumper", "long sleeve top"),
    "long_sleeve_outwear": ("jacket", "coat", "blazer", "hoodie", "cardigan", "outerwear"),
    "vest": ("tank top", "camisole", "sleeveless top", "vest top"),
    "shorts": ("shorts",),
    "trousers": ("pants", "trousers", "jeans", "slacks", "chinos", "leggings"),
    "skirt": ("skirt",),
    "short_sleeve_dress": ("short sleeve dress",),
    "long_sleeve_dress": ("long sleeve dress",),
    "vest_dress": ("sleeveless dress", "vest dress"),
    "sling_dress": ("slip dress", "strappy dress", "spaghetti strap dress", "sling dress"),
}

GENERIC_TYPE_GROUPS = {
    "top": ["short_sleeve_top", "long_sleeve_top", "vest"],
    "tops": ["short_sleeve_top", "long_sleeve_top", "vest"],
    "dress": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
    "dresses": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
    "bottom": ["shorts", "trousers", "skirt"],
    "bottoms": ["shorts", "trousers", "skirt"],
}


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


def _is_negated(text: str, phrase: str) -> bool:
    for hint in NEGATION_HINTS:
        if hint + phrase in text:
            return True
    return False


def _add_value(filters: dict, section: str, field: str, value: str) -> None:
    filters.setdefault(section, {}).setdefault(field, [])
    if value not in filters[section][field]:
        filters[section][field].append(value)


def parse_custom_query(message: str) -> dict:
    text = re.sub(r"\s+", " ", message.lower()).strip()
    filters: dict = {"include": {}, "exclude": {}}

    for field, mapping in ALL_MAPPINGS.items():
        for value, natural_variants in mapping.items():
            phrases = [value.replace("_", " ")]
            if isinstance(natural_variants, dict):
                phrases.extend(str(key).lower() for key in natural_variants.keys())
            elif isinstance(natural_variants, (list, tuple, set)):
                phrases.extend(str(item).lower() for item in natural_variants)

            matched = any(_contains_phrase(text, phrase) for phrase in phrases if phrase)
            if not matched:
                continue

            section = "exclude" if any(_is_negated(text, phrase) for phrase in phrases if phrase) else "include"
            _add_value(filters, section, field, value)

    for value, phrases in TYPE_SYNONYMS.items():
        for phrase in phrases:
            if _contains_phrase(text, phrase):
                section = "exclude" if _is_negated(text, phrase) else "include"
                _add_value(filters, section, "type", value)

    for phrase, values in GENERIC_TYPE_GROUPS.items():
        if _contains_phrase(text, phrase):
            section = "exclude" if _is_negated(text, phrase) else "include"
            for value in values:
                _add_value(filters, section, "type", value)

    return {section: block for section, block in filters.items() if block}


def refine_custom_query(previous_query: str, previous_filters: dict, refinement: str) -> dict:
    updated_filters = deepcopy(previous_filters or {})
    parsed = parse_custom_query(refinement)

    for section in ("include", "exclude"):
        incoming = parsed.get(section, {})
        if not incoming:
            continue
        updated_filters.setdefault(section, {})
        for field, values in incoming.items():
            existing = updated_filters[section].get(field, [])
            updated_filters[section][field] = list(dict.fromkeys(existing + values))

    include = updated_filters.get("include", {})
    exclude = updated_filters.get("exclude", {})
    for field, ex_values in exclude.items():
        if field not in include:
            continue
        include[field] = [value for value in include[field] if value not in ex_values]
        if not include[field]:
            del include[field]

    return {
        "query": f"{previous_query} {refinement}".strip(),
        "filters": {section: block for section, block in updated_filters.items() if block},
    }
