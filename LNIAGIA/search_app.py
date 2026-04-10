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
from llm_query_parser import parse_query, refine_query, OLLAMA_MODEL
from query_parsing_models.baselines.exporter import load_latest_exported_parser


_CONVERSATION_MODEL = None
_INITIAL_BASELINE_PARSER = None
_INITIAL_BASELINE_META: dict[str, Any] = {}
_INITIAL_BASELINE_LOADED = False

PARSER_MODE_ENV_VAR = "VIRALYTICS_PARSER_MODE"
DEFAULT_PARSER_MODE = "auto"


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
    configured = os.getenv(PARSER_MODE_ENV_VAR, DEFAULT_PARSER_MODE).strip().lower()
    if configured in {"auto", "baseline", "llm"}:
        return configured
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
        return f"LLM only ({PARSER_MODE_ENV_VAR}=llm)"

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
    detected_type: str,
    user_input: str | None = None,
    conversation_state: dict | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Single HTTP-friendly conversation step.

    Expected request flow:
    - First request: send only detected_type.
    - Next requests: send user_input and previous state.

    Returns JSON-serializable state + ranked results.
    """

    if not isinstance(detected_type, str) or not detected_type.strip():
        return {"ok": False, "error": "detected_type is required."}

    base_query = f"I want {detected_type.strip()}"
    base_filters = {
        "include": {
            "type": [detected_type.strip()],
        },
        "exclude": {},
    }

    # Pre-flight: vector DB must exist
    client = _get_client()
    if not _collection_exists(client):
        client.close()
        return {
            "ok": False,
            "error": "Vector DB not found. Run VectorDBManager.py first to create it.",
        }
    client.close()

    global _CONVERSATION_MODEL
    if _CONVERSATION_MODEL is None:
        _CONVERSATION_MODEL = _load_model()

    state = conversation_state if isinstance(conversation_state, dict) else {}
    current_query = state.get("query") if isinstance(state.get("query"), str) else base_query
    current_filters = state.get("filters") if isinstance(state.get("filters"), dict) else base_filters

    message = (user_input or "").strip()

    if message:
        if message.upper() == "NEW":
            current_query = base_query
            current_filters = base_filters
        else:
            updated = refine_query(
                previous_query=current_query,
                previous_filters=current_filters,
                refinement=message,
                verbose=False,
            )
            current_query = updated["query"]
            current_filters = updated["filters"]

    hits = filtered_search(current_query, current_filters, _CONVERSATION_MODEL, strict=strict)

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

    return {
        "ok": True,
        "results": ranked_results,
        "state": {
            "query": current_query,
            "filters": current_filters,
            "strict": bool(strict),
        },
    }

if __name__ == "__main__":
    main()
