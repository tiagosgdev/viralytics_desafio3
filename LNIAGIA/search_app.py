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
from llm_query_parser import parse_query, OLLAMA_MODEL


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
        print(f"  Brand   : {p.get('brand', '?')}")
        print(f"  Price   : {p.get('price', '?')}")
        print(f"  Season  : {p.get('season', '?')}")
        print(f"  Occasion: {p.get('occasion', '?')}")
        print(f"  Gender  : {p.get('gender', '?')}")

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
    print("\n  Type a natural-language query, or 'exit' to quit.\n")

    while True:
        query = input("  Query > ").strip()
        if not query or query.lower() == "exit":
            print("\n  Goodbye!\n")
            break

        # Step 1: Parse query with LLM
        print("\n  Parsing query with LLM ...")
        parsed_filters = parse_query(query, verbose=False)

        print("\n  Extracted filters:")
        print(f"  {json.dumps(parsed_filters, indent=4)}")

        # Step 2: Filtered semantic search
        print("\n  Searching ...")
        hits = filtered_search(query, parsed_filters, model)

        # Step 3: Display results
        _display_results(hits, parsed_filters)


if __name__ == "__main__":
    main()
