# ════════════════════════════════════════════════════════════
# search_app.py
# Interactive clothing search that combines:
#   1. LLM query parsing  (natural language → structured filters)
#   2. Filtered semantic search  (embedding + Qdrant filters)
# ════════════════════════════════════════════════════════════

import json
import os
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
from query_parsing.llm_query_parser import parse_query, refine_query, OLLAMA_MODEL
from query_parsing.qp_models.baselines.exporter import load_latest_exported_parser


_CONVERSATION_MODEL = None
_INITIAL_BASELINE_PARSER = None
_INITIAL_BASELINE_META: dict[str, Any] = {}
_INITIAL_BASELINE_LOADED = False

# Parser switch (manual toggle).
# Keep one active line and comment the other.
PARSER_MODE = "LLM"
# PARSER_MODE = "Baseline"

DEFAULT_PARSER_MODE = "llm"

DEFAULT_ASSISTANT_MODE = "cruella"
ASSISTANT_MODES = {"cruella", "edna"}

STRICTNESS_UNKNOWN = "unknown"
STRICTNESS_STRICT = "strict"
STRICTNESS_FLEXIBLE = "flexible"

STRICT_MARKERS = {
    "strict",
    "exact",
    "exactly",
    "precise",
    "picky",
    "must match",
    "only exact",
    "only strict",
}
FLEXIBLE_MARKERS = {
    "flexible",
    "open",
    "similar",
    "close matches",
    "not strict",
    "anything close",
    "creative",
}

YES_WORDS = {"yes", "y", "sure", "ok", "okay", "go ahead"}
NO_WORDS = {"no", "n", "nah", "nope"}

SEARCH_MARKERS = {
    "search",
    "show me",
    "show results",
    "find",
    "recommend",
    "go ahead",
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

DETAIL_FIELDS = ["type", "color", "style", "pattern", "fit", "occasion", "season", "material", "brand"]

_PROMPTS_DIR = _SCRIPT_DIR / "query_parsing" / "prompts"
_PERSONA_PROMPT_CACHE: dict[str, str] = {}


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


def _normalize_strict_preference(value: Any) -> str:
    lowered = str(value or STRICTNESS_UNKNOWN).strip().lower()
    if lowered in {STRICTNESS_STRICT, STRICTNESS_FLEXIBLE}:
        return lowered
    return STRICTNESS_UNKNOWN


def _extract_strict_preference(message: str, awaiting_strictness: bool = False) -> str | None:
    lowered = message.strip().lower()
    if not lowered:
        return None

    if awaiting_strictness:
        if lowered in YES_WORDS:
            return STRICTNESS_STRICT
        if lowered in NO_WORDS:
            return STRICTNESS_FLEXIBLE

    for marker in FLEXIBLE_MARKERS:
        if marker in lowered:
            return STRICTNESS_FLEXIBLE

    for marker in STRICT_MARKERS:
        if marker in lowered:
            return STRICTNESS_STRICT

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
            "ask_strictness": "Do you want only exact matches, or should I include near matches with potential?",
            "strict_reask": "Answer clearly: exact matches only, or flexible suggestions?",
            "search_unavailable": "Search is unavailable right now. Try again in a moment.",
            "results_ready": f"Analysis complete. I found {result_count} strong option(s).",
            "no_results": "No strong matches. Add more detail and we try again.",
        }
    else:
        fallback = {
            "intro": "Darling, I am Cruella, your fashion accomplice. Tell me what power look you want.",
            "nudge": "Darling, give me a clothing direction and I will do the rest.",
            "reset": "Fresh canvas, darling. We begin again.",
            "off_topic": "Absolutely ghastly use of my time. Ask me about fashion, not that.",
            "ask_details": f"Promising start, darling, but I need more edge: {detail_text}.",
            "ask_strictness": "Do you want razor-sharp exact matches only, or a wider sweep with bold near matches?",
            "strict_reask": "Pick one, darling: exact and uncompromising, or flexible and exploratory?",
            "search_unavailable": "I would search right now, but the fashion vault is unavailable for a moment.",
            "results_ready": f"Now that is power. I analyzed everything and found {result_count} option(s).",
            "no_results": "I searched hard and found nothing worthy yet. Give me new constraints and we strike again.",
        }

    return fallback.get(scenario, fallback["nudge"])


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
            model=OLLAMA_MODEL,
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
            item_data = {"id": raw_item_id, **payload}

        ranked_results.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "item_id": raw_item_id,
                "item": item_data,
            }
        )

    return ranked_results


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
) -> tuple[str, dict, str, str | None]:
    normalized_mode = _normalize_assistant_mode(mode)

    # With prior context, always use LLM refinement regardless of persona mode.
    if has_previous_context:
        try:
            updated = refine_query(
                previous_query=current_query,
                previous_filters=current_filters,
                refinement=message,
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
    strict_preference: str,
    assistant_mode: str,
    parser_source: str,
    awaiting_strictness: bool,
    strict_default: bool,
) -> dict[str, Any]:
    return {
        "query": query,
        "filters": _normalize_filter_payload(filters),
        "strict": _strict_from_preference(strict_preference, strict_default=strict_default),
        "strict_preference": _normalize_strict_preference(strict_preference),
        "assistant_mode": _normalize_assistant_mode(assistant_mode),
        "parser_source": parser_source,
        "awaiting_strictness": bool(awaiting_strictness),
        "started": True,
    }


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
    print(f"  LLM refiner  : {OLLAMA_MODEL}")
    print(f"  Total results: {TOTAL_RESULTS} (evaluating {MAX_QUERY_RESULTS} cands)")
    print(f"  Threshold    : {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    # ── Load embedding model once ──
    model = _load_model()

    # ── Search loop ──
    print("\n  Type a natural-language query, or 'exit' to quit.")
    print("  Commands: 'new: <query>' to start a new query, 'reset' to clear state, 'show' to inspect state.\n")

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

        # Step 2: Ask user for strictness
        strict_ans = input("\n  Do you want strict matching (only exact filter matches)? (y/n) > ").strip().lower()
        strict = strict_ans == 'y'

        # Step 3: Filtered semantic search
        print("\n  Searching ...")
        hits = filtered_search(current_query, current_filters, model, strict=strict)

        # Step 4: Display results
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
    - Non-technical strictness question before search.
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

    strict_preference = _normalize_strict_preference(state.get("strict_preference"))
    if strict and strict_preference == STRICTNESS_UNKNOWN:
        strict_preference = STRICTNESS_STRICT

    awaiting_strictness = bool(state.get("awaiting_strictness"))
    parser_source = str(state.get("parser_source") or "none")

    message = (user_input or "").strip()

    if not message:
        scenario = "intro" if not started_before else "nudge"
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
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=awaiting_strictness,
                strict_default=strict,
            ),
        }

    if message.upper() in {"NEW", "RESET"}:
        mode = _normalize_assistant_mode(assistant_mode) if assistant_mode else (existing_mode if started_before else requested_mode)
        current_query = base_query
        current_filters = _ensure_type_filter(base_filters, detected)
        strict_preference = STRICTNESS_UNKNOWN
        awaiting_strictness = False
        parser_source = "reset"
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
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=awaiting_strictness,
                strict_default=strict,
            ),
        }

    explicit_search = _is_explicit_search_request(message)
    strict_signal = _extract_strict_preference(message, awaiting_strictness=awaiting_strictness)
    if strict_signal is not None:
        strict_preference = strict_signal
        awaiting_strictness = False

    if (
        strict_signal is None
        and not awaiting_strictness
        and not explicit_search
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
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=awaiting_strictness,
                strict_default=strict,
            ),
        }

    parser_warning = None
    short_strict_answer = strict_signal is not None and len(message.split()) <= 5
    should_update_with_parser = not (awaiting_strictness and short_strict_answer and _has_filters(current_filters))

    if should_update_with_parser:
        current_query, current_filters, parser_source, parser_warning = _update_state_with_message(
            mode=mode,
            current_query=current_query,
            current_filters=current_filters,
            message=message,
            has_previous_context=has_previous_context,
        )
        current_filters = _ensure_type_filter(current_filters, detected)

    filter_count = _count_filter_values(current_filters)

    needs_more_detail = filter_count == 0 or (filter_count <= 1 and not explicit_search)
    if needs_more_detail and strict_signal is None and not awaiting_strictness:
        detail_fields = _missing_detail_fields(current_filters)
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
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=False,
                strict_default=strict,
            ),
        }

    if awaiting_strictness and strict_signal is None:
        reply = _generate_persona_reply(
            mode,
            "strict_reask",
            user_message=message,
            context={},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "ask_strictness",
            "results": [],
            "warning": parser_warning,
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=True,
                strict_default=strict,
            ),
        }

    if strict_preference == STRICTNESS_UNKNOWN:
        reply = _generate_persona_reply(
            mode,
            "ask_strictness",
            user_message=message,
            context={},
        )
        return {
            "ok": True,
            "reply": reply,
            "mode": mode,
            "action": "ask_strictness",
            "results": [],
            "warning": parser_warning,
            "state": _build_state_payload(
                query=current_query,
                filters=current_filters,
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=True,
                strict_default=strict,
            ),
        }

    search_is_strict = _strict_from_preference(strict_preference, strict_default=strict)

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
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=False,
                strict_default=search_is_strict,
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
                    strict_preference=strict_preference,
                    assistant_mode=mode,
                    parser_source=parser_source,
                    awaiting_strictness=False,
                    strict_default=search_is_strict,
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
                strict_preference=strict_preference,
                assistant_mode=mode,
                parser_source=parser_source,
                awaiting_strictness=False,
                strict_default=search_is_strict,
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
            strict_preference=strict_preference,
            assistant_mode=mode,
            parser_source=parser_source,
            awaiting_strictness=False,
            strict_default=search_is_strict,
        ),
    }

if __name__ == "__main__":
    main()
