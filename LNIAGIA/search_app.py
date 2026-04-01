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
from llm_query_parser import parse_query, refine_query, OLLAMA_MODEL


def clear():
    os.system("cls" if os.name == "nt" else "clear")


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
    print(f"  LLM parser   : {OLLAMA_MODEL}")
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
            print("\n  Parsing query with LLM ...")
            current_query = user_query
            current_filters = parse_query(current_query, verbose=False)
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


if __name__ == "__main__":
    main()
