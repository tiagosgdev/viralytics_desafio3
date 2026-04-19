# ════════════════════════════════════════════════════════════
# search_app.py
# Interactive clothing search that combines:
#   1. LLM query parsing  (natural language → structured filters)
#   2. Filtered semantic search  (embedding + Qdrant filters)
# ════════════════════════════════════════════════════════════

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    import ollama
except Exception:
    ollama = None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from DB.vector.VectorDBManager import (
    _load_model,
    _get_client,
    _collection_exists,
    _collection_point_count,
    filtered_search,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOTAL_RESULTS,
    MAX_QUERY_RESULTS,
    SIMILARITY_THRESHOLD,
)
from DB.SQLLite.DBManager import get_items_by_ids
from query_parsing.llm_query_parser import (
    OLLAMA_REFINER_MODEL,
    OLLAMA_ROUTER_MODEL,
    parse_query,
    refine_query,
)
from query_parsing.qp_models.baselines.exporter import load_latest_exported_parser


_CONVERSATION_MODEL = None
_INITIAL_BASELINE_PARSER = None
_INITIAL_BASELINE_META: dict[str, Any] = {}
_INITIAL_BASELINE_LOADED = False


def set_conversation_embedding_model(model: Any) -> None:
    global _CONVERSATION_MODEL
    _CONVERSATION_MODEL = model

# Parser switch (manual toggle).
# Keep one active line and comment the other.
PARSER_MODE = "LLM"
# PARSER_MODE = "Baseline"

DEFAULT_PARSER_MODE = "llm"

DEFAULT_ASSISTANT_MODE = "cruella"
ASSISTANT_MODES = {"cruella", "edna"}

STRICTNESS_STRICT = "strict"
STRICTNESS_FLEXIBLE = "flexible"

YES_WORDS = {"yes", "y", "sure", "ok", "okay", "go ahead"}
NO_WORDS = {"no", "n", "nah", "nope"}

CONFIRM_MARKERS = {
    "confirm",
    "i confirm",
    "correct",
    "that's right",
    "thats right",
    "that is perfect",
    "that's perfect",
    "perfect",
    "looks good",
    "looks perfect",
    "sounds good",
    "works for me",
    "i like that",
    "go ahead",
    "please proceed",
    "proceed",
    "approved",
    "do it",
    "run it",
    "search now",
    "run the search",
    "why not",
}
REVISE_MARKERS = {
    "change",
    "adjust",
    "modify",
    "different",
    "not quite",
    "not exactly",
    "not right",
    "not correct",
    "wrong",
    "instead",
    "update it",
}

SEARCH_MARKERS = {
    "search",
    "show me",
    "show results",
    "find",
    "recommend",
    "go ahead",
    "please proceed",
    "proceed",
    "run it",
    "start search",
    "that is all",
    "thats all",
    "lets go",
    "let's go",
}

FASHION_MARKERS = {
    "fashion",
    "outfit",
    "clothes",
    "clothing",
    "wear",
    "style",
    "look",
    "dress",
    "shirt",
    "t-shirt",
    "tee",
    "top",
    "blouse",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "material",
    "pattern",
    "fit",
    "color",
    "colour",
    "formal",
    "casual",
    "smart casual",
    "sporty",
    "beach",
    "wedding",
    "meeting",
    "occasion",
    "brand",
}

TYPE_HINT_MARKERS = {
    "t-shirt",
    "tshirt",
    "tee",
    "shirt",
    "top",
    "blouse",
    "sweater",
    "hoodie",
    "jacket",
    "coat",
    "blazer",
    "outerwear",
    "vest",
    "dress",
    "dresses",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
}

REFINEMENT_CONTEXT_MARKERS = {
    " not ",
    "don't",
    "dont",
    "without",
    "avoid",
    "exclude",
    "except",
    "instead",
    "also",
    "remove",
    "swap",
    "replace",
    "make it",
    "keep it",
    "want it",
    "it to be",
    "anything but",
    "anything, just",
}

DETAIL_FIELDS = ["type", "color", "style", "pattern", "fit", "occasion", "season", "material", "brand"]

_MAX_RECENT_USER_MESSAGES = 6

TYPE_DISPLAY_MAP = {
    "short_sleeve_top": "T-shirt",
    "long_sleeve_top": "long-sleeve shirt",
    "long_sleeve_outwear": "jacket",
    "vest": "vest",
    "shorts": "shorts",
    "trousers": "trousers",
    "skirt": "skirt",
    "short_sleeve_dress": "short-sleeve dress",
    "long_sleeve_dress": "long-sleeve dress",
    "vest_dress": "sleeveless dress",
    "sling_dress": "slip dress",
}

FIT_DISPLAY_MAP = {
    "slim fit": "slim",
    "slim_fit": "slim",
    "regular_fit": "regular",
    "regular": "regular",
    "relaxed": "relaxed",
    "loose_fit": "loose",
    "loose": "loose",
    "relaxed_fit": "relaxed",
    "oversized": "oversized",
    "tailored": "tailored",
    "fitted": "fitted",
    "athletic": "athletic",
    "baggy": "baggy",
    "cropped": "cropped",
}

GENDER_DISPLAY_MAP = {
    "male": "men",
    "female": "women",
    "unisex": "any gender",
}

AGE_GROUP_DISPLAY_MAP = {
    "baby": "babies",
    "child": "children",
    "teenager": "teenagers",
    "young adult": "young adults",
    "adult": "adults",
    "senior": "seniors",
}

SUMMARY_PRIMARY_FIELDS = [
    "type",
    "color",
    "fit",
    "style",
    "pattern",
    "material",
    "occasion",
    "season",
    "brand",
    "gender",
    "age_group",
]

FIELD_DISPLAY_LABEL = {
    "age_group": "age group",
    "sleeve_style": "sleeve style",
    "hem_style": "hem style",
    "waist_style": "waist style",
    "leg_style": "leg style",
    "dress_style": "dress style",
    "outwear_pockets": "pockets",
    "bottom_pockets": "pockets",
}

FIELD_INCLUDE_TEMPLATE = {
    "style": "with a {values} style",
    "pattern": "with a {values} pattern",
    "fit": "with a {values} fit",
    "material": "in {values}",
    "occasion": "for {values}",
    "season": "for {values}",
    "brand": "from {values}",
    "gender": "for {values}",
    "age_group": "for {values}",
    "neckline": "with a {values}",
    "collar": "with a {values} collar",
    "sleeve_style": "with {values} sleeves",
    "hem_style": "with a {values} hem",
    "closure": "with a {values} closure",
    "hood": "with {values}",
    "insulation": "with {values} insulation",
    "waterproof": "with {values}",
    "outwear_pockets": "with {values}",
    "waist": "in waist size {values}",
    "waist_style": "with a {values} waist",
    "rise": "with a {values} rise",
    "length": "in {values} length",
    "leg_style": "with a {values} leg cut",
    "bottom_pockets": "with {values}",
    "dress_style": "in a {values} silhouette",
}

FIELD_EXCLUDE_TEMPLATE = {
    "fit": "not {values}",
    "type": "excluding {values}",
    "color": "excluding {values} colors",
    "style": "without {values} style",
    "pattern": "without {values} pattern",
    "material": "excluding {values} materials",
    "brand": "not from {values}",
    "gender": "excluding {values}",
    "age_group": "excluding {values}",
    "waterproof": "excluding {values} items",
}

_PROMPTS_DIR = _SCRIPT_DIR / "query_parsing" / "prompts"
_PERSONA_PROMPT_CACHE: dict[str, str] = {}

_CONFIRMATION_LEAD_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "here",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "your",
    "darling",
}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def _to_int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_filter_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {"include": {}, "exclude": {}}

    include = payload.get("include", {})
    exclude = payload.get("exclude", {})

    if not isinstance(include, dict):
        include = {}
    if not isinstance(exclude, dict):
        exclude = {}

    return {
        "include": include,
        "exclude": exclude,
    }


def _normalize_recent_user_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []

    cleaned = []
    for message in messages:
        if not isinstance(message, str):
            continue
        text = message.strip()
        if text:
            cleaned.append(text)

    return cleaned[-_MAX_RECENT_USER_MESSAGES:]


def _append_recent_user_message(messages: list[str], message: str) -> list[str]:
    base = _normalize_recent_user_messages(messages)
    text = str(message or "").strip()
    if text:
        base.append(text)
    return base[-_MAX_RECENT_USER_MESSAGES:]


def _get_parser_mode() -> str:
    # Accept only the two supported modes.
    configured = str(PARSER_MODE).strip().lower()
    if configured in {"baseline", "llm"}:
        return configured

    # Safe fallback if the value is accidentally changed to something invalid.
    return DEFAULT_PARSER_MODE


def _candidate_results_dirs() -> list[Path]:
    return [
        Path("LNIAGIA/query_parsing_models/results"),
        Path("query_parsing_models/results"),
    ]


def _load_exported_initial_parser() -> tuple[Any | None, dict[str, Any]]:
    global _INITIAL_BASELINE_PARSER, _INITIAL_BASELINE_META, _INITIAL_BASELINE_LOADED

    if _INITIAL_BASELINE_LOADED:
        return _INITIAL_BASELINE_PARSER, _INITIAL_BASELINE_META

    _INITIAL_BASELINE_LOADED = True
    last_error = "No exported parser found."

    for results_dir in _candidate_results_dirs():
        try:
            parser, manifest = load_latest_exported_parser(results_dir)
            _INITIAL_BASELINE_PARSER = parser
            _INITIAL_BASELINE_META = dict(manifest)
            _INITIAL_BASELINE_META["results_dir"] = str(results_dir)
            return _INITIAL_BASELINE_PARSER, _INITIAL_BASELINE_META
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    _INITIAL_BASELINE_PARSER = None
    _INITIAL_BASELINE_META = {"load_error": last_error}
    return _INITIAL_BASELINE_PARSER, _INITIAL_BASELINE_META


def _initial_parser_banner() -> str:
    mode = _get_parser_mode()

    if mode == "llm":
        return "LLM only (PARSER_MODE=LLM)"

    parser, manifest = _load_exported_initial_parser()
    if parser is None:
        if mode == "baseline":
            return f"baseline requested but unavailable; fallback to LLM ({manifest.get('load_error', 'unknown')})"
        return "LLM fallback (no exported baseline parser found)"

    model_name = str(manifest.get("model", "unknown"))
    folder_size = manifest.get("folder_size", "?")
    score = None
    selection_row = manifest.get("selection_row")
    if isinstance(selection_row, dict):
        score = selection_row.get("tradeoff_score")

    if isinstance(score, (int, float)):
        return f"exported baseline: {model_name} (size={folder_size}, tradeoff={score:.4f})"
    return f"exported baseline: {model_name} (size={folder_size})"


def _parse_initial_query(query: str) -> tuple[dict, str]:
    mode = _get_parser_mode()

    if mode == "llm":
        return _normalize_filter_payload(parse_query(query, verbose=False)), "llm"

    parser, _ = _load_exported_initial_parser()

    if parser is not None:
        try:
            parsed = parser.parse(query)
            return _normalize_filter_payload(parsed), "baseline"
        except Exception as exc:
            print(f"\n  WARNING: Exported baseline parser failed ({exc}). Falling back to LLM parser.")

    if mode == "baseline":
        print("\n  WARNING: Baseline parser mode requested but no exported parser is available. Falling back to LLM parser.")

    return _normalize_filter_payload(parse_query(query, verbose=False)), "llm"


def _normalize_assistant_mode(value: Any) -> str:
    mode = str(value or DEFAULT_ASSISTANT_MODE).strip().lower()
    if mode == "cruela":
        mode = "cruella"
    if mode in ASSISTANT_MODES:
        return mode
    return DEFAULT_ASSISTANT_MODE


def _strict_preference_for_mode(mode: str) -> str:
    normalized_mode = _normalize_assistant_mode(mode)
    if normalized_mode == "edna":
        return STRICTNESS_FLEXIBLE
    return STRICTNESS_STRICT


def _classify_binary_intent_with_llm(mode: str, stage: str, message: str) -> str | None:
    """Classify short, ambiguous replies for pending binary questions.

    Returns one of:
      - confirmation stage: "confirm" | "revise" | None
    """
    if ollama is None:
        return None

    normalized_mode = _normalize_assistant_mode(mode)
    stage_name = str(stage or "").strip().lower()
    msg = (message or "").strip()
    if not msg:
        return None

    if stage_name != "confirmation":
        return None

    labels = "confirm, revise, unclear"
    task = (
        "The user is responding to: 'Confirm the brief so I can search, or tell me what to change.' "
        "Decide if they confirm or request a change."
    )

    system_prompt = (
        "You are an intent classifier for a fashion search assistant. "
        f"Assistant mode: {normalized_mode}.\n"
        f"Return exactly one lowercase label: {labels}.\n"
        "Do not explain. Output only the label."
    )
    user_prompt = f"{task}\nUser reply: {msg}"

    try:
        response = ollama.chat(
            model=OLLAMA_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0},
        )

        raw = ""
        if isinstance(response, dict):
            message_obj = response.get("message", {})
            if isinstance(message_obj, dict):
                raw = str(message_obj.get("content", "")).strip()
        else:
            message_obj = getattr(response, "message", None)
            raw = str(getattr(message_obj, "content", "")).strip()

        cleaned = raw.lower().strip().strip("`\"' ")
        if not cleaned:
            return None

        tokenized = "".join(ch if (ch.isalpha() or ch.isspace()) else " " for ch in cleaned)
        words = {part for part in tokenized.split() if part}

        if stage_name == "confirmation":
            if "revise" in words or "change" in words or "modify" in words:
                return "revise"
            if "confirm" in words or "approved" in words or "approve" in words:
                return "confirm"
            if cleaned == "revise":
                return "revise"
            if cleaned == "confirm":
                return "confirm"
            return None
        return None
    except Exception:
        return None


def _extract_confirmation_signal(
    message: str,
    awaiting_confirmation: bool = False,
    assistant_mode: str = DEFAULT_ASSISTANT_MODE,
) -> str | None:
    lowered = message.strip().lower()
    if not lowered:
        return None

    for marker in REVISE_MARKERS:
        if marker in lowered:
            return "revise"

    for marker in CONFIRM_MARKERS:
        if marker in lowered:
            return "confirm"

    if awaiting_confirmation:
        if lowered in YES_WORDS or any(lowered.startswith(f"{word} ") for word in YES_WORDS) or "why not" in lowered:
            return "confirm"
        if lowered in NO_WORDS or any(lowered.startswith(f"{word} ") for word in NO_WORDS):
            return "revise"

        llm_signal = _classify_binary_intent_with_llm(
            mode=assistant_mode,
            stage="confirmation",
            message=message,
        )
        if llm_signal in {"confirm", "revise"}:
            return llm_signal

    return None


def _strict_from_preference(strict_preference: str, strict_default: bool = False) -> bool:
    if strict_preference == STRICTNESS_STRICT:
        return True
    if strict_preference == STRICTNESS_FLEXIBLE:
        return False
    return bool(strict_default)


def _is_explicit_search_request(message: str) -> bool:
    lowered = message.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in SEARCH_MARKERS)


def _is_probably_fashion_related(message: str) -> bool:
    lowered = message.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in FASHION_MARKERS)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _count_filter_values(filters: dict) -> int:
    count = 0
    normalized = _normalize_filter_payload(filters)
    for section in ("include", "exclude"):
        block = normalized.get(section, {})
        for values in block.values():
            if not isinstance(values, list):
                continue
            count += len([v for v in values if isinstance(v, str) and v.strip()])
    return count


def _has_filters(filters: dict) -> bool:
    return _count_filter_values(filters) > 0


def _ensure_type_filter(filters: dict, detected_type: str | None) -> dict:
    normalized = _normalize_filter_payload(filters)
    if not detected_type:
        return normalized

    include = normalized.setdefault("include", {})
    current = include.get("type", [])
    if not isinstance(current, list):
        current = []

    include["type"] = _dedupe_preserve_order([detected_type] + [v for v in current if isinstance(v, str)])
    return normalized


def _normalize_type_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned:
            normalized.append(cleaned)

    return _dedupe_preserve_order(normalized)


def _extract_type_values(filters: dict) -> tuple[str, ...]:
    include = _normalize_filter_payload(filters).get("include", {})
    return tuple(sorted(_normalize_type_values(include.get("type"))))


def _type_filters_changed(current_filters: dict, candidate_filters: dict) -> bool:
    candidate_types = _extract_type_values(candidate_filters)
    if not candidate_types:
        return False
    return candidate_types != _extract_type_values(current_filters)


def _is_type_only_request(filters: dict) -> bool:
    normalized = _normalize_filter_payload(filters)
    include = normalized.get("include", {})
    exclude = normalized.get("exclude", {})

    include_fields = {
        field
        for field, values in include.items()
        if isinstance(values, list) and any(isinstance(v, str) and v.strip() for v in values)
    }
    has_excludes = any(
        isinstance(values, list) and any(isinstance(v, str) and v.strip() for v in values)
        for values in exclude.values()
    )

    return include_fields == {"type"} and bool(_extract_type_values(normalized)) and not has_excludes


def _minimal_request_signature(filters: dict) -> str | None:
    if not _is_type_only_request(filters):
        return None

    types = _extract_type_values(filters)
    if not types:
        return None
    return f"type:{'|'.join(types)}"


def _message_mentions_type_hint(message: str) -> bool:
    lowered = f" {message.strip().lower()} "
    if not lowered.strip():
        return False
    return any(f" {hint} " in lowered for hint in TYPE_HINT_MARKERS)


def _is_contextual_refinement_message(message: str) -> bool:
    lowered = f" {message.strip().lower()} "
    if not lowered.strip():
        return False

    if any(marker in lowered for marker in REFINEMENT_CONTEXT_MARKERS):
        return True
    return _is_probably_fashion_related(message)


def _is_detailed_revision_message(message: str, confirmation_signal: str | None) -> bool:
    if confirmation_signal != "revise":
        return False

    text = str(message or "").strip()
    if not text:
        return False

    if len(text.split()) <= 2:
        return False

    return _is_contextual_refinement_message(text) or _message_mentions_type_hint(text)


def _ordered_filter_fields(block: dict) -> list[str]:
    ordered = [field for field in DETAIL_FIELDS if field in block]
    ordered += sorted(field for field in block.keys() if field not in DETAIL_FIELDS)
    return ordered


def _format_filter_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip().replace("_", " ")
        if text:
            cleaned.append(text)
    return cleaned


def _humanize_filter_value(field: str, value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""

    key = raw.lower().replace("-", "_").replace(" ", "_")
    if field == "type":
        return TYPE_DISPLAY_MAP.get(key, raw.replace("_", " "))
    if field == "fit":
        return FIT_DISPLAY_MAP.get(key, raw.replace("_", " "))
    if field == "gender":
        lookup = raw.lower().replace("_", " ")
        return GENDER_DISPLAY_MAP.get(lookup, raw.replace("_", " "))
    if field == "age_group":
        lookup = raw.lower().replace("_", " ")
        return AGE_GROUP_DISPLAY_MAP.get(lookup, raw.replace("_", " "))
    if field == "hood" and key == "none":
        return "no hood"
    if field == "waterproof":
        if key == "none":
            return "non-waterproof"
        if key == "water_resistant":
            return "water-resistant"
    if field in {"outwear_pockets", "bottom_pockets"} and key == "none":
        return "no pockets"
    if field == "brand":
        return raw
    return raw.replace("_", " ")


def _join_readable(values: list[str], conjunction: str = "or") -> str:
    filtered = [v for v in values if v]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} {conjunction} {filtered[1]}"
    return f"{', '.join(filtered[:-1])}, {conjunction} {filtered[-1]}"


def _summary_field_order(block: dict) -> list[str]:
    ordered = [field for field in SUMMARY_PRIMARY_FIELDS if field in block]
    ordered.extend(field for field in _ordered_filter_fields(block) if field not in ordered)
    return ordered


def _display_field_name(field: str) -> str:
    return FIELD_DISPLAY_LABEL.get(field, field.replace("_", " "))


def _build_include_fragment(field: str, values_text: str) -> str:
    template = FIELD_INCLUDE_TEMPLATE.get(field)
    if template:
        return template.format(values=values_text)
    return f"with {_display_field_name(field)} {values_text}"


def _build_exclude_fragment(field: str, values_text: str) -> str:
    template = FIELD_EXCLUDE_TEMPLATE.get(field)
    if template:
        return template.format(values=values_text)
    return f"excluding {_display_field_name(field)} {values_text}"


def _compose_include_phrase(include: dict, query: str) -> str:
    types = [_humanize_filter_value("type", v) for v in _format_filter_values(include.get("type"))][:3]
    colors = [_humanize_filter_value("color", v) for v in _format_filter_values(include.get("color"))][:3]
    fits = [_humanize_filter_value("fit", v) for v in _format_filter_values(include.get("fit"))][:3]

    type_text = _join_readable(types)
    color_text = _join_readable(colors)
    fit_text = _join_readable(fits)

    if type_text and color_text:
        base = f"{color_text} {type_text}"
    elif type_text:
        base = type_text
    elif color_text:
        base = f"clothing in {color_text}"
    else:
        base = "clothing recommendations"

    extras = []
    if fit_text:
        extras.append(_build_include_fragment("fit", fit_text))

    for field in _summary_field_order(include):
        if field in {"type", "color", "fit"}:
            continue
        values = [_humanize_filter_value(field, v) for v in _format_filter_values(include.get(field))][:3]
        values_text = _join_readable(values)
        if not values_text:
            continue
        extras.append(_build_include_fragment(field, values_text))

    if extras:
        return f"{base} {', '.join(extras)}"
    return base


def _compose_exclude_phrase(exclude: dict) -> str:
    fragments = []
    for field in _summary_field_order(exclude):
        values = [_humanize_filter_value(field, v) for v in _format_filter_values(exclude.get(field))][:3]
        values_text = _join_readable(values)
        if not values_text:
            continue
        fragments.append(_build_exclude_fragment(field, values_text))

    return _join_readable(fragments, conjunction="and")


def _build_requirements_summary(query: str, filters: dict) -> str:
    normalized = _normalize_filter_payload(filters)
    include = normalized.get("include", {})
    exclude = normalized.get("exclude", {})

    include_text = _compose_include_phrase(include, query)
    exclude_text = _compose_exclude_phrase(exclude)

    if exclude_text:
        return f"{include_text}, {exclude_text}"
    return include_text


def _extract_situation_label(query: str, filters: dict) -> str | None:
    lowered = query.strip().lower()
    normalized = _normalize_filter_payload(filters)
    include = normalized.get("include", {})

    occasions = {
        value.strip().lower()
        for value in _format_filter_values(include.get("occasion"))
    }
    styles = {
        value.strip().lower()
        for value in _format_filter_values(include.get("style"))
    }

    if "interview" in lowered:
        return "important interview"
    if any(token in lowered for token in ("ceo", "boss", "board", "executive", "promotion")):
        return "high-stakes work moment"
    if "meeting" in lowered and ("work" in occasions or "formal" in styles or "smart casual" in styles):
        return "important work meeting"
    if "wedding" in lowered or "wedding" in occasions:
        return "wedding guest look"
    if "beach" in lowered or "beach" in occasions:
        return "beach outing"
    if "party" in lowered or "party" in occasions or "date night" in occasions:
        return "special social event"
    if "formal event" in occasions:
        return "formal event"
    if "work" in occasions:
        return "work setting"
    return None


def _fallback_confirmation_lead(mode: str, situation_label: str | None) -> str:
    abstract_situation = None
    if situation_label:
        label = situation_label.strip().lower()
        if label in {
            "important interview",
            "high-stakes work moment",
            "important work meeting",
            "work setting",
        }:
            abstract_situation = "important professional moment"
        elif label in {"wedding guest look", "special social event", "formal event"}:
            abstract_situation = "special social moment"
        elif label in {"beach outing"}:
            abstract_situation = "relaxed moment"
        else:
            abstract_situation = "important moment"

    normalized_mode = _normalize_assistant_mode(mode)
    if normalized_mode == "edna":
        if abstract_situation:
            return f"Understood. We are dressing for this {abstract_situation}."
        return "Understood. We will keep this precise."

    if abstract_situation:
        return f"Darling, for this {abstract_situation}, we are building a look with intent."
    return "Darling, we are shaping this look with intention."


def _collect_confirmation_forbidden_phrases(summary: str, filters: dict) -> list[str]:
    _ = summary
    normalized = _normalize_filter_payload(filters)
    phrases = set()

    for section in ("include", "exclude"):
        block = normalized.get(section, {})
        for field in _summary_field_order(block):
            values = _format_filter_values(block.get(field))
            for value in values:
                normalized_value = value.strip().lower()
                if normalized_value:
                    phrases.add(normalized_value)

                humanized = _humanize_filter_value(field, value).strip().lower()
                if humanized:
                    phrases.add(humanized)

    return sorted(p for p in phrases if p and len(p) >= 3)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False

    if " " in phrase or "-" in phrase:
        return phrase in text

    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _overlap_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return {
        token
        for token in cleaned.split()
        if len(token) >= 3 and token not in _CONFIRMATION_LEAD_STOPWORDS
    }


def _lead_violates_confirmation_rules(lead: str, summary: str, filters: dict) -> bool:
    lowered = (lead or "").strip().lower()
    if not lowered:
        return True

    if "?" in lowered:
        return True

    sentence_parts = [part.strip() for part in re.split(r"[.!?]", lowered) if part.strip()]
    if len(sentence_parts) > 1:
        return True

    forbidden_phrases = _collect_confirmation_forbidden_phrases(summary, filters)
    for phrase in forbidden_phrases:
        if _contains_phrase(lowered, phrase):
            return True

    lead_tokens = _overlap_tokens(lowered)
    summary_tokens = _overlap_tokens(summary)
    if lead_tokens and summary_tokens:
        overlap = lead_tokens & summary_tokens
        overlap_ratio = len(overlap) / max(1, len(lead_tokens))
        if len(overlap) >= 2 and overlap_ratio >= 0.45:
            return True

    return False


def _generate_confirmation_lead(
    mode: str,
    *,
    summary: str,
    query: str,
    user_message: str,
    filters: dict,
) -> str:
    normalized_mode = _normalize_assistant_mode(mode)
    situation_label = _extract_situation_label(query, filters)
    fallback = _fallback_confirmation_lead(normalized_mode, situation_label)
    forbidden_phrases = _collect_confirmation_forbidden_phrases(summary, filters)
    forbidden_preview = ", ".join(forbidden_phrases[:30]) if forbidden_phrases else "(none)"

    if ollama is None:
        return fallback

    persona_prompt = _load_persona_prompt_text(normalized_mode)
    temperature = 0.65 if normalized_mode == "cruella" else 0.25

    system_prompt = (
        f"{persona_prompt}\n\n"
        "Write one short lead sentence for a confirmation message.\n"
        "Rules:\n"
        "- Exactly one sentence.\n"
        "- Mention the situation when provided.\n"
        "- Do not ask questions.\n"
        "- Never include any concrete value from the canonical summary or filters.\n"
        "- You may use abstract phrasing like 'interesting color choice' or 'strong style direction'.\n"
        "- Do not rewrite or alter filter constraints.\n"
        "- Do not include phrases like 'Here is your brief' or 'Confirm if this is correct'.\n"
        "- Plain text only."
    )

    user_prompt = (
        f"User message: {user_message or '(none)'}\n"
        f"Semantic query: {query}\n"
        f"Situation: {situation_label or 'none'}\n"
        f"Canonical summary (do not rewrite): {summary}\n"
        f"Forbidden values (never use literally): {forbidden_preview}"
    )

    try:
        response = ollama.chat(
            model=OLLAMA_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )

        raw = ""
        if isinstance(response, dict):
            message_obj = response.get("message", {})
            if isinstance(message_obj, dict):
                raw = str(message_obj.get("content", "")).strip()
        else:
            message_obj = getattr(response, "message", None)
            raw = str(getattr(message_obj, "content", "")).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        lead = raw.splitlines()[0].strip() if raw else ""
        if not lead:
            return fallback

        lowered = lead.lower()
        blocked_fragments = (
            "here is your brief",
            "confirm if this is correct",
            "or tell me what to change",
        )
        if any(fragment in lowered for fragment in blocked_fragments):
            return fallback

        if len(lead) > 220:
            return fallback

        if _lead_violates_confirmation_rules(lead, summary, filters):
            return fallback

        if lead[-1] not in ".!?":
            lead = f"{lead}."
        return lead
    except Exception:
        return fallback


def _build_confirmation_prompt(mode: str, summary: str, lead: str | None = None) -> str:
    if _normalize_assistant_mode(mode) == "edna":
        core = (
            f"Here is your brief: {summary}. Confirm if this is correct and I will search, "
            "or tell me what to change."
        )
    else:
        core = (
            f"Here is your brief: {summary}. Confirm if this is correct and I will search, "
            "or tell me what to change."
        )

    if lead:
        return f"{lead} {core}"
    return core


def _build_revision_prompt(mode: str) -> str:
    if _normalize_assistant_mode(mode) == "edna":
        return "Fine. Tell me exactly what to change, and I will update the brief before searching."
    return "Darling, perfect. Tell me what to change and I will update the brief before we search."


def _missing_detail_fields(filters: dict) -> list[str]:
    include = _normalize_filter_payload(filters).get("include", {})
    missing = []
    for field in DETAIL_FIELDS:
        values = include.get(field)
        if isinstance(values, list) and values:
            continue
        missing.append(field)
    return missing[:3]


def _load_persona_prompt_text(mode: str) -> str:
    normalized_mode = _normalize_assistant_mode(mode)
    cached = _PERSONA_PROMPT_CACHE.get(normalized_mode)
    if cached is not None:
        return cached

    prompt_file = _PROMPTS_DIR / f"{normalized_mode}_persona.md"
    default_prompt = (
        "You are a fashion assistant with a strong and distinctive voice. "
        "You only handle clothing and style recommendations."
    )

    if prompt_file.exists():
        try:
            content = prompt_file.read_text(encoding="utf-8").strip()
        except OSError:
            content = default_prompt
    else:
        content = default_prompt

    _PERSONA_PROMPT_CACHE[normalized_mode] = content
    return content


def _fallback_persona_reply(mode: str, scenario: str, context: dict[str, Any]) -> str:
    detail_fields = context.get("detail_fields")
    if isinstance(detail_fields, list) and detail_fields:
        detail_text = ", ".join(detail_fields)
    else:
        detail_text = "color, style, or occasion"

    result_count = _to_int_or_none(context.get("result_count")) or 0

    if mode == "edna":
        fallback = {
            "intro": "I am Edna. I handle fashion, and I handle it correctly. Tell me what you need.",
            "nudge": "Give me a clothing request. Clear, direct, and useful.",
            "reset": "Reset complete. Start again. Better this time.",
            "off_topic": "No. I am not here for that. Ask me about clothes, fit, color, or occasion.",
            "ask_details": f"Too vague. Give me more: {detail_text}.",
            "search_unavailable": "Search is unavailable right now. Try again in a moment.",
            "results_ready": f"Analysis complete. I found {result_count} strong option(s).",
            "no_results": "I found 0 results for that exact brief. Add more detail or relax one constraint and we try again.",
        }
    else:
        fallback = {
            "intro": "Darling, I am Cruella, your fashion accomplice. Tell me what power look you want.",
            "nudge": "Darling, give me a clothing direction and I will do the rest.",
            "reset": "Fresh canvas, darling. We begin again.",
            "off_topic": "Absolutely ghastly use of my time. Ask me about fashion, not that.",
            "ask_details": f"Promising start, darling, but I need more edge: {detail_text}.",
            "search_unavailable": "I would search right now, but the fashion vault is unavailable for a moment.",
            "results_ready": f"Now that is power. I analyzed everything and found {result_count} option(s).",
            "no_results": "Darling, we have hit a roadblock: I found 0 results for that exact brief. Let us loosen one constraint and find something fabulous.",
        }

    return fallback.get(scenario, fallback["nudge"])


def _fixed_persona_intro(mode: str) -> str:
    normalized_mode = _normalize_assistant_mode(mode)
    if normalized_mode == "edna":
        return "I am Edna. I handle fashion, and I handle it correctly. Tell me what you need."
    return "Darling, I am Cruella, your fashion accomplice. Tell me what power look you want."


def _ensure_no_results_clarity(mode: str, text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return text

    clarity_markers = (
        "no results",
        "0 result",
        "zero result",
        "did not find any results",
        "didn't find any results",
        "found nothing",
    )
    if any(marker in lowered for marker in clarity_markers):
        return text

    normalized_mode = _normalize_assistant_mode(mode)
    if normalized_mode == "edna":
        return f"I found 0 results for that exact brief. {text}".strip()
    return f"Darling, I found 0 results for that exact brief. {text}".strip()


def _generate_persona_reply(mode: str, scenario: str, user_message: str, context: dict[str, Any]) -> str:
    normalized_mode = _normalize_assistant_mode(mode)
    fallback = _fallback_persona_reply(normalized_mode, scenario, context)

    if ollama is None:
        return fallback

    persona_prompt = _load_persona_prompt_text(normalized_mode)
    temperature = 0.75 if normalized_mode == "cruella" else 0.35

    system_prompt = (
        f"{persona_prompt}\n\n"
        "Output rules:\n"
        "- Reply in plain text only.\n"
        "- No markdown and no JSON.\n"
        "- Keep it concise (1 to 3 sentences).\n"
        "- If scenario is no_results, explicitly state that 0 results were found.\n"
        "- Stay fashion-focused."
    )
    user_prompt = (
        f"Scenario: {scenario}\n"
        f"User message: {user_message or '(none)'}\n"
        f"Context: {json.dumps(context, ensure_ascii=False)}\n"
        "Write the assistant reply now."
    )

    try:
        response = ollama.chat(
            model=OLLAMA_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )

        raw = ""
        if isinstance(response, dict):
            message_obj = response.get("message", {})
            if isinstance(message_obj, dict):
                raw = str(message_obj.get("content", "")).strip()
        else:
            message_obj = getattr(response, "message", None)
            raw = str(getattr(message_obj, "content", "")).strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()

        if raw:
            if scenario == "no_results":
                return _ensure_no_results_clarity(normalized_mode, raw)
            return raw
    except Exception:
        pass

    return fallback


def _ensure_vector_db_ready() -> tuple[bool, str | None]:
    try:
        client = _get_client()
    except Exception as exc:
        return False, f"Vector DB client could not be created ({exc})."

    try:
        if not _collection_exists(client):
            return False, "Vector DB not found. Run VectorDBManager.py first to create it."
    except Exception as exc:
        return False, f"Vector DB check failed ({exc})."
    finally:
        client.close()

    return True, None


def _build_ranked_results(hits: list) -> list[dict[str, Any]]:
    ranked_item_ids = []
    for hit in hits:
        payload = hit.payload or {}
        item_id = _to_int_or_none(payload.get("item_id"))
        if item_id is not None:
            ranked_item_ids.append(item_id)

    sqlite_items = get_items_by_ids(ranked_item_ids)
    sqlite_items_by_id = {}
    for item in sqlite_items:
        row_id = _to_int_or_none(item.get("id"))
        if row_id is not None:
            sqlite_items_by_id[row_id] = item

    ranked_results = []
    for rank, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        raw_item_id = payload.get("item_id")
        item_id = _to_int_or_none(raw_item_id)

        item_data = sqlite_items_by_id.get(item_id)
        if item_data is None:
            item_data = {"id": raw_item_id, "error": "Item details not found in SQL database."}

        ranked_results.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "item_id": raw_item_id,
                "item": item_data,
            }
        )

    return ranked_results

def search_detected_items(
    detected_categories: list[str] | None,
    strict: bool = False,
    # ── optional profile filters ─────────────────────────────────────────────
    colors: list[str] | None = None,
    styles: list[str] | None = None,
    materials: list[str] | None = None,
    seasons: list[str] | None = None,
    occasions: list[str] | None = None,
    gender: str | None = None,
    age_group: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:

    detected = _dedupe_preserve_order(
        [
            str(cat).strip().lower()
            for cat in (detected_categories or [])
            if isinstance(cat, str) and str(cat).strip()
        ]
    )

    # ─────────────────────────────────────────────────────────────
    # HARD FILTERS (used to restrict database results)
    # ─────────────────────────────────────────────────────────────
    include: dict[str, list[str]] = {}

    if detected:
        include["type"] = detected

    if gender and gender.strip():
        include["gender"] = [gender.strip().lower()]

    if age_group and age_group.strip():
        include["age_group"] = [age_group.strip().lower()]

    filters = {
        "include": include,
        "exclude": {},
    }

    # ─────────────────────────────────────────────────────────────
    # SOFT PROFILE SIGNALS (ONLY for semantic ranking)
    # ─────────────────────────────────────────────────────────────
    profile_parts = []

    if colors:
        cleaned = [c.strip() for c in colors if c and c.strip()]
        if cleaned:
            profile_parts.append(f"preferred colors: {', '.join(cleaned)}")

    if styles:
        cleaned = [s.strip() for s in styles if s and s.strip()]
        if cleaned:
            profile_parts.append(f"styles: {', '.join(cleaned)}")

    if materials:
        cleaned = [m.strip() for m in materials if m and m.strip()]
        if cleaned:
            profile_parts.append(f"materials: {', '.join(cleaned)}")

    if seasons:
        cleaned = [s.strip() for s in seasons if s and s.strip()]
        if cleaned:
            profile_parts.append(f"seasons: {', '.join(cleaned)}")

    if occasions:
        cleaned = [o.strip() for o in occasions if o and o.strip()]
        if cleaned:
            profile_parts.append(f"occasions: {', '.join(cleaned)}")

    profile_ctx = ""
    if profile_parts:
        profile_ctx = "User preferences (soft ranking signals): " + "; ".join(profile_parts) + ". "

    # ─────────────────────────────────────────────────────────────
    # SEMANTIC QUERY (clean + model-friendly)
    # ─────────────────────────────────────────────────────────────
    if detected:
        worn = "The user is wearing " + ", ".join(
            cat.replace("_", " ") for cat in detected
        ) + ". "
    else:
        worn = "Find versatile clothing recommendations. "

    query = (
        f"{worn}"
        f"{profile_ctx}"
        "Recommend visually compatible clothing items that match style and aesthetic."
    )

    # ─────────────────────────────────────────────────────────────
    # DEBUG
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 search_detected_items QUERY")
    print(f"query   : {query}")
    print(f"filters : {filters}")
    print(f"strict  : {strict}")
    print("=" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────
    # VECTOR DB CHECK
    # ─────────────────────────────────────────────────────────────
    db_ready, db_error = _ensure_vector_db_ready()
    if not db_ready:
        return [], db_error

    global _CONVERSATION_MODEL
    if _CONVERSATION_MODEL is None:
        try:
            _CONVERSATION_MODEL = _load_model()
        except Exception as exc:
            return [], f"Embedding model load failed ({exc})."

    # ─────────────────────────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────────────────────────
    try:
        hits = filtered_search(
            query,
            filters,
            _CONVERSATION_MODEL,
            strict=bool(strict),
        )
    except Exception as exc:
        return [], f"Search failed ({exc})."

    return _build_ranked_results(hits), None

def _parse_with_mode(mode: str, message: str) -> tuple[dict, str, str | None]:
    normalized_mode = _normalize_assistant_mode(mode)

    if normalized_mode == "edna":
        parser, manifest = _load_exported_initial_parser()
        if parser is not None:
            try:
                parsed = parser.parse(message)
                return _normalize_filter_payload(parsed), "baseline", None
            except Exception as exc:
                warning = f"Baseline parser failed ({exc})."
            
        else:
            warning = f"Baseline parser unavailable ({manifest.get('load_error', 'unknown')})."

        try:
            llm_parsed = parse_query(message, verbose=False)
            return _normalize_filter_payload(llm_parsed), "llm_fallback", warning
        except Exception as exc:
            return {"include": {}, "exclude": {}}, "none", f"{warning} LLM fallback also failed ({exc})."

    try:
        parsed = parse_query(message, verbose=False)
        return _normalize_filter_payload(parsed), "llm", None
    except Exception as exc:
        return {"include": {}, "exclude": {}}, "none", f"LLM parser failed ({exc})."


def _update_state_with_message(
    mode: str,
    current_query: str,
    current_filters: dict,
    message: str,
    has_previous_context: bool,
    recent_user_messages: list[str] | None = None,
) -> tuple[str, dict, str, str | None]:
    normalized_mode = _normalize_assistant_mode(mode)

    # With prior context, always use LLM refinement regardless of persona mode.
    if has_previous_context:
        try:
            updated = refine_query(
                previous_query=current_query,
                previous_filters=current_filters,
                refinement=message,
                recent_messages=recent_user_messages,
                verbose=False,
            )
            next_query = updated.get("query") if isinstance(updated.get("query"), str) else f"{current_query} {message}".strip()
            next_filters = _normalize_filter_payload(updated.get("filters", current_filters))
            return next_query, next_filters, "llm_refine", None
        except Exception as exc:
            fallback_query = f"{current_query} {message}".strip()
            warning = f"LLM refine failed ({exc}). Keeping previous filters."
            return fallback_query, _normalize_filter_payload(current_filters), "llm_refine_failed", warning

    # First conversational parse with no prior context:
    # - Cruella -> parse_query
    # - Edna    -> exported baseline parser (LLM fallback)
    parsed, source, warning = _parse_with_mode(normalized_mode, message)
    return message.strip(), parsed, source, warning


def _build_state_payload(
    *,
    query: str,
    filters: dict,
    strict: bool,
    assistant_mode: str,
    parser_source: str,
    awaiting_confirmation: bool = False,
    has_completed_search: bool = False,
    pending_detail_signature: str = "",
    recent_user_messages: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "filters": _normalize_filter_payload(filters),
        "strict": bool(strict),
        "assistant_mode": _normalize_assistant_mode(assistant_mode),
        "parser_source": parser_source,
        "awaiting_confirmation": bool(awaiting_confirmation),
        "has_completed_search": bool(has_completed_search),
        "pending_detail_signature": str(pending_detail_signature or ""),
        "started": True,
    }

    if isinstance(recent_user_messages, list):
        payload["recent_user_messages"] = _normalize_recent_user_messages(recent_user_messages)

    return payload


def _display_results(hits: list, parsed_filters: dict):
    """Pretty-print the search results."""

    if not hits:
        print("\n  No results found matching your criteria.\n")
        return

    print(f"\n  Found {len(hits)} result(s):\n")

    for rank, hit in enumerate(hits, 1):
        p = hit.payload
        score = hit.score

        print(f"  --- #{rank}  (score: {score:.4f}) ---")
        print(f"  Item ID : {p.get('item_id', '?')}")
        print(f"  Type    : {p.get('type', '?')}")
        print(f"  Color   : {p.get('color', '?')}")
        print(f"  Style   : {p.get('style', '?')}")
        print(f"  Pattern : {p.get('pattern', '?')}")
        print(f"  Material: {p.get('material', '?')}")
        print(f"  Fit     : {p.get('fit', '?')}")
        print(f"  Gender  : {p.get('gender', '?')}")
        print(f"  Age Grp : {p.get('age_group', '?')}")
        print(f"  Season  : {p.get('season', '?')}")
        print(f"  Occasion : {p.get('occasion', '?')}")
        print(f"  Brand   : {p.get('brand', '?')}")
        print(f"  Price   : ${p.get('price', '?')}")


        desc = p.get("description", "")
        if len(desc) > 200:
            desc = desc[:200] + "..."
        print(f"  Desc    : {desc}")
        print()


def main():
    clear()
    print("=" * 60)
    print("  Clothing Search App  -  Viralytics")
    print("=" * 60)

    # ── Pre-flight: vector DB must exist ──
    client = _get_client()
    if not _collection_exists(client):
        print("\n  Vector DB not found.")
        print("  Run VectorDBManager.py first to create it.")
        client.close()
        return
    points = _collection_point_count(client)
    client.close()

    print(f"  Vector DB    : {COLLECTION_NAME} ({points} points)")
    print(f"  Embed model  : {EMBEDDING_MODEL_NAME}")
    print(f"  Initial parser: {_initial_parser_banner()}")
    print(f"  LLM parser/refiner: {OLLAMA_REFINER_MODEL}")
    print(f"  LLM interaction : {OLLAMA_ROUTER_MODEL}")
    print(f"  Total results: {TOTAL_RESULTS} (evaluating {MAX_QUERY_RESULTS} cands)")
    print(f"  Threshold    : {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    # ── Load embedding model once ──
    model = _load_model()

    # ── Search loop ──
    print("\n  Type a natural-language query, or 'exit' to quit.")
    print("  Commands: 'new: <query>' to start a new query, 'reset' to clear state, 'show' to inspect state.\n")

    assistant_mode = _normalize_assistant_mode(DEFAULT_ASSISTANT_MODE)
    strict = _strict_from_preference(_strict_preference_for_mode(assistant_mode), strict_default=False)
    strict_label = "strict" if strict else "flexible"
    print(f"  Mode policy : {assistant_mode} ({strict_label} matching)\n")

    current_query: str | None = None
    current_filters: dict = {}

    while True:
        user_input = input("  Query > ").strip()
        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered == "exit":
            print("\n  Goodbye!\n")
            break

        if lowered == "reset":
            current_query = None
            current_filters = {}
            print("\n  Context reset. Start with a new query.\n")
            continue

        if lowered == "show":
            print("\n  Current semantic query:")
            print(f"  {current_query if current_query else '(none)'}")
            print("\n  Current filters:")
            print(f"  {json.dumps(current_filters, indent=4)}\n")
            continue

        is_new_query = lowered.startswith("new:")
        if is_new_query:
            user_query = user_input[4:].strip()
            if not user_query:
                print("\n  Please provide text after 'new:'.\n")
                continue
        else:
            user_query = user_input

        if current_query is None or is_new_query:
            current_query = user_query
            current_filters, parser_source = _parse_initial_query(current_query)
            if parser_source == "baseline":
                print("\n  Parsed initial query with exported baseline parser.")
            else:
                print("\n  Parsed initial query with LLM parser.")
        else:
            print("\n  Refining current query + filters with LLM ...")
            updated = refine_query(
                previous_query=current_query,
                previous_filters=current_filters,
                refinement=user_query,
                verbose=False,
            )
            current_query = updated["query"]
            current_filters = updated["filters"]

        print("\n  Active semantic query:")
        print(f"  {current_query}")

        print("\n  Extracted filters:")
        print(f"  {json.dumps(current_filters, indent=4)}")

        # Step 2: Filtered semantic search
        print("\n  Searching ...")
        hits = filtered_search(current_query, current_filters, model, strict=strict)

        # Step 3: Display results
        _display_results(hits, current_filters)


def run_conversation_model(
    detected_type: str | None,
    user_input: str | None = None,
    conversation_state: dict | None = None,
    strict: bool = False,
    assistant_mode: str | None = None,
) -> dict[str, Any]:
    """
    Single HTTP-friendly conversation step with persona support.

    Supported behavior:
    - Cruella and Edna persona modes.
    - Off-topic guardrails (fashion-only scope).
    - Clarifying questions when user request is underspecified.
    - Persona-locked strictness policy (Cruella strict, Edna flexible).
    - First parse routing by mode:
        Cruella -> LLM parse_query
        Edna    -> baseline parser (LLM fallback)
    - Follow-up updates with prior context always use LLM refine_query.
    """

    if detected_type is not None and not isinstance(detected_type, str):
        return {"ok": False, "error": "detected_type must be a string when provided."}

    state = conversation_state if isinstance(conversation_state, dict) else {}
    started_before = bool(state.get("started"))

    requested_mode = _normalize_assistant_mode(assistant_mode) if assistant_mode else DEFAULT_ASSISTANT_MODE
    existing_mode = _normalize_assistant_mode(state.get("assistant_mode"))
    mode = existing_mode if started_before else requested_mode

    detected = detected_type.strip() if isinstance(detected_type, str) and detected_type.strip() else None
    base_query = f"I want {detected}" if detected else "I want clothing recommendations"
    base_filters = {
        "include": {"type": [detected]} if detected else {},
        "exclude": {},
    }

    state_query = state.get("query") if isinstance(state.get("query"), str) else ""
    state_query = state_query.strip()
    state_filters = _normalize_filter_payload(state.get("filters") if isinstance(state.get("filters"), dict) else {})
    normalized_base_filters = _normalize_filter_payload(base_filters)

    has_non_base_query = bool(state_query) and state_query != base_query
    has_non_base_filters = _has_filters(state_filters) and state_filters != normalized_base_filters
    has_previous_context = has_non_base_query or has_non_base_filters

    current_query = state_query if state_query else base_query
    current_filters = state_filters if has_previous_context else normalized_base_filters
    current_filters = _ensure_type_filter(current_filters, detected)

    # Strictness is derived from the selected persona mode; free-form strict/flexible
    # negotiation is intentionally disabled.
    strict_preference = _strict_preference_for_mode(mode)
    awaiting_confirmation = bool(state.get("awaiting_confirmation"))
    parser_source = str(state.get("parser_source") or "none")
    has_completed_search = bool(state.get("has_completed_search"))
    pending_detail_signature = str(state.get("pending_detail_signature") or "").strip()
    recent_user_messages = _normalize_recent_user_messages(state.get("recent_user_messages"))

    message = (user_input or "").strip()
    recent_user_messages = _append_recent_user_message(recent_user_messages, message)

    if not message:
        scenario = "intro" if not started_before else "nudge"
        if not started_before:
            reply = _fixed_persona_intro(mode)
        else:
            reply = _generate_persona_reply(
                mode,
                scenario,
                user_message="",
                context={"detected_type": detected or "unspecified"},
            )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": scenario,
            "results": [],
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=awaiting_confirmation,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    if message.upper() in {"NEW", "RESET"}:
        mode = _normalize_assistant_mode(assistant_mode) if assistant_mode else (existing_mode if started_before else requested_mode)
        current_query = base_query
        current_filters = _ensure_type_filter(base_filters, detected)
        strict_preference = _strict_preference_for_mode(mode)
        awaiting_confirmation = False
        parser_source = "reset"
        has_completed_search = False
        pending_detail_signature = ""
        reply = _generate_persona_reply(
            mode,
            "reset",
            user_message=message,
            context={"detected_type": detected or "unspecified"},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "reset",
            "results": [],
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=awaiting_confirmation,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    explicit_search = _is_explicit_search_request(message)
    confirmation_signal = _extract_confirmation_signal(
        message,
        awaiting_confirmation=awaiting_confirmation,
        assistant_mode=mode,
    )

    contextual_refinement = (
        has_previous_context
        and _is_contextual_refinement_message(message)
        and (
            (has_completed_search and not awaiting_confirmation)
            or (awaiting_confirmation and confirmation_signal == "revise")
        )
    )

    if (
        confirmation_signal is None
        and not awaiting_confirmation
        and not explicit_search
        and not contextual_refinement
        and not _is_probably_fashion_related(message)
    ):
        reply = _generate_persona_reply(
            mode,
            "off_topic",
            user_message=message,
            context={},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "off_topic",
            "results": [],
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=awaiting_confirmation,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    parser_warning = None
    new_query_from_type_change = False
    has_previous_context_for_update = has_previous_context

    if contextual_refinement and _message_mentions_type_hint(message):
        candidate_filters, candidate_source, candidate_warning = _parse_with_mode(mode, message)
        if _type_filters_changed(current_filters, candidate_filters):
            new_query_from_type_change = True
            has_previous_context_for_update = False
            has_completed_search = False
            pending_detail_signature = ""
            strict_preference = _strict_preference_for_mode(mode)
            awaiting_confirmation = False
            current_query = message.strip()
            current_filters = _ensure_type_filter(candidate_filters, detected)
            parser_source = f"{candidate_source}_new_query" if candidate_source else "new_query"
            parser_warning = candidate_warning

    detailed_revision_message = _is_detailed_revision_message(message, confirmation_signal)
    short_confirmation_answer = (
        confirmation_signal is not None
        and len(message.split()) <= 6
        and not detailed_revision_message
    )
    should_update_with_parser = not new_query_from_type_change and not (
        awaiting_confirmation and short_confirmation_answer and _has_filters(current_filters)
    )

    if should_update_with_parser:
        current_query, current_filters, parser_source, parser_warning = _update_state_with_message(
            mode=mode,
            current_query=current_query,
            current_filters=current_filters,
            message=message,
            has_previous_context=has_previous_context_for_update,
            recent_user_messages=recent_user_messages,
        )
        current_filters = _ensure_type_filter(current_filters, detected)

    filter_count = _count_filter_values(current_filters)
    minimal_signature = _minimal_request_signature(current_filters)
    repeated_minimal_request = (
        not has_completed_search
        and bool(pending_detail_signature)
        and minimal_signature is not None
        and minimal_signature == pending_detail_signature
        and confirmation_signal is None
        and not awaiting_confirmation
    )

    auto_search_refinement_turn = contextual_refinement and not new_query_from_type_change
    needs_more_detail = (
        filter_count == 0 or (filter_count <= 1 and not explicit_search)
    ) and not repeated_minimal_request and not auto_search_refinement_turn

    if needs_more_detail and confirmation_signal is None:
        detail_fields = _missing_detail_fields(current_filters)
        pending_detail_signature = minimal_signature or ""
        reply = _generate_persona_reply(
            mode,
            "ask_details",
            user_message=message,
            context={"detail_fields": detail_fields},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "ask_details",
            "results": [],
            "warning": parser_warning,
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=False,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    if not needs_more_detail:
        pending_detail_signature = ""

    should_skip_confirmation = (
        repeated_minimal_request
        or auto_search_refinement_turn
        or (
            explicit_search
            and confirmation_signal != "revise"
            and has_previous_context_for_update
            and _has_filters(current_filters)
        )
    )

    if awaiting_confirmation:
        if confirmation_signal == "confirm" or explicit_search:
            awaiting_confirmation = False
        elif confirmation_signal == "revise" and not should_update_with_parser:
            reply = _build_revision_prompt(mode)
            return {
                "ok": True,
                "reply": reply,
                "mode": mode,
                "action": "confirm_edit",
                "results": [],
                "warning": parser_warning,
                "state": _build_state_payload(
                    query=current_query,
                    filters=current_filters,
                    strict=_strict_from_preference(strict_preference, strict_default=False),
                    assistant_mode=mode,
                    parser_source=parser_source,
                    awaiting_confirmation=False,
                    has_completed_search=has_completed_search,
                    pending_detail_signature=pending_detail_signature,
                    recent_user_messages=recent_user_messages,
                ),
            }
        else:
            summary = _build_requirements_summary(current_query, current_filters)
            confirmation_lead = _generate_confirmation_lead(
                mode,
                summary=summary,
                query=current_query,
                user_message=message,
                filters=current_filters,
            )
            reply = _build_confirmation_prompt(mode, summary, lead=confirmation_lead)
            return {
                "ok": True,
                "reply": reply,
                "mode": mode,
                "action": "confirm_requirements",
                "results": [],
                "warning": parser_warning,
                "state": _build_state_payload(
                    query=current_query,
                    filters=current_filters,
                    strict=_strict_from_preference(strict_preference, strict_default=False),
                    assistant_mode=mode,
                    parser_source=parser_source,
                    awaiting_confirmation=True,
                    has_completed_search=has_completed_search,
                    pending_detail_signature=pending_detail_signature,
                    recent_user_messages=recent_user_messages,
                ),
            }
    elif not should_skip_confirmation:
        summary = _build_requirements_summary(current_query, current_filters)
        confirmation_lead = _generate_confirmation_lead(
            mode,
            summary=summary,
            query=current_query,
            user_message=message,
            filters=current_filters,
        )
        reply = _build_confirmation_prompt(mode, summary, lead=confirmation_lead)
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "confirm_requirements",
            "results": [],
            "warning": parser_warning,
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=True,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    search_is_strict = _strict_from_preference(strict_preference, strict_default=False)

    db_ready, db_error = _ensure_vector_db_ready()
    if not db_ready:
        reply = _generate_persona_reply(
            mode,
            "search_unavailable",
            user_message=message,
            context={"error": db_error or "unknown"},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "search_unavailable",
            "results": [],
            "warning": db_error,
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=False,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    global _CONVERSATION_MODEL
    if _CONVERSATION_MODEL is None:
        try:
            _CONVERSATION_MODEL = _load_model()
        except Exception as exc:
            reply = _generate_persona_reply(
                mode,
                "search_unavailable",
                user_message=message,
                context={"error": str(exc)},
            )
            return {
                "ok": True,
                "reply": reply,
                "mode": mode,
                "action": "search_unavailable",
                "results": [],
                "warning": f"Embedding model load failed ({exc}).",
                "state": _build_state_payload(
                    query=current_query,
                    filters=current_filters,
                    strict=_strict_from_preference(strict_preference, strict_default=False),
                    assistant_mode=mode,
                    parser_source=parser_source,
                    awaiting_confirmation=False,
                    has_completed_search=has_completed_search,
                    pending_detail_signature=pending_detail_signature,
                    recent_user_messages=recent_user_messages,
                ),
            }

    try:
        hits = filtered_search(
            current_query,
            current_filters,
            _CONVERSATION_MODEL,
            strict=search_is_strict,
        )
    except Exception as exc:
        reply = _generate_persona_reply(
            mode,
            "search_unavailable",
            user_message=message,
            context={"error": str(exc)},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "search_failed",
            "results": [],
            "warning": f"Search failed ({exc}).",
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict=_strict_from_preference(strict_preference, strict_default=False),
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_confirmation=False,
                has_completed_search=has_completed_search,
                pending_detail_signature=pending_detail_signature,
                recent_user_messages=recent_user_messages,
            ),
        }

    ranked_results = _build_ranked_results(hits)
    scenario = "results_ready" if ranked_results else "no_results"
    reply = _generate_persona_reply(
        mode,
        scenario,
        user_message=message,
        context={
            "result_count": len(ranked_results),
            "strict": search_is_strict,
            "parser_source": parser_source,
        },
    )

    has_completed_search = True
    pending_detail_signature = ""
    return {
        "ok": True,
        "reply": reply,
        "mode": mode,
        "action": "searched",
        "results": ranked_results,
        "warning": parser_warning,
        "state": _build_state_payload(
            query=current_query,
            filters=current_filters,
            strict=_strict_from_preference(strict_preference, strict_default=False),
            assistant_mode=mode,
            parser_source=parser_source,
            awaiting_confirmation=False,
            has_completed_search=has_completed_search,
            pending_detail_signature=pending_detail_signature,
            recent_user_messages=recent_user_messages,
        ),
    }

if __name__ == "__main__":
    main()
