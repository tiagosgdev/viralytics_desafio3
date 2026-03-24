# ════════════════════════════════════════════════════════════
# VectorDBManager.py
# Interactive CLI for embedding generation, vector DB
# management and semantic search over clothing descriptions.
# ════════════════════════════════════════════════════════════

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    Range,
)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "clothing_search"
TOTAL_RESULTS = 10
MAX_QUERY_RESULTS = 50
SIMILARITY_THRESHOLD = 0.25
BATCH_SIZE = 64

# Weights for soft filtering
PENALTY_WEIGHT = 0.2
BOOST_WEIGHT = 0.2

# BGE models perform better when queries are prefixed with this instruction.
# Documents are encoded WITHOUT the prefix (done during build_vector_db).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════

_SCRIPT_DIR = Path(__file__).resolve().parent
_NL_DIR = _SCRIPT_DIR / "NL_Items_Descriptions"
_QDRANT_DIR = _SCRIPT_DIR / "qdrant_storage"


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def _load_model():
    """Load (and warm up) the sentence-transformer model."""
    print(f"\n  Loading model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Warm-up encode so first real query is fast
    model.encode(["warm-up"], show_progress_bar=False)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model ready  —  dimension: {dim}")
    return model


def _get_client() -> QdrantClient:
    _QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(_QDRANT_DIR))


def _collection_exists(client: QdrantClient) -> bool:
    names = [c.name for c in client.get_collections().collections]
    return COLLECTION_NAME in names


def _collection_point_count(client: QdrantClient) -> int:
    if not _collection_exists(client):
        return 0
    return client.get_collection(COLLECTION_NAME).points_count


# ══════════════════════════════════════════════════════════════
# CORE OPERATIONS
# ══════════════════════════════════════════════════════════════

def list_nl_files() -> list[Path]:
    if not _NL_DIR.exists():
        return []
    return sorted(_NL_DIR.glob("*.json"))


def pick_nl_file() -> Path | None:
    files = list_nl_files()
    if not files:
        print("\n  No description files found in NL_Items_Descriptions/.")
        return None

    print("\n  Available NL description files:")
    for i, f in enumerate(files, 1):
        size_kb = f.stat().st_size / 1024
        print(f"    [{i}] {f.name}  ({size_kb:.1f} KB)")

    while True:
        choice = input("\n  Select a file (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("  Invalid choice. Please try again.")


def build_vector_db(json_path: Path, model: SentenceTransformer):
    """Load NL descriptions, embed them and store in Qdrant."""

    # ── Load descriptions ──
    with open(json_path, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    print(f"\n  Loaded {len(items)} items from {json_path.name}")

    # ── Encode descriptions in batches ──
    descriptions = [item["description"] for item in items]

    print(f"  Generating embeddings (batch_size={BATCH_SIZE}) ...")
    embeddings = model.encode(
        descriptions,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # ── Create / recreate collection ──
    client = _get_client()
    dim = embeddings.shape[1]

    if _collection_exists(client):
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted old collection.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # ── Upload points ──
    print("  Uploading to Qdrant ...")
    points = []
    for idx, (item, emb) in enumerate(zip(items, embeddings)):
        payload = {
            "item_id": item["item_id"],
            "description": item["description"],
        }
        if "metadata" in item:
            payload.update(item["metadata"])

        points.append(PointStruct(id=idx, vector=emb.tolist(), payload=payload))

    # Upsert in batches
    for i in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + BATCH_SIZE],
        )

    count = _collection_point_count(client)
    client.close()
    print(f"\n  Vector DB ready  —  {count} points indexed.")


def search(query: str, model: SentenceTransformer):
    """Embed a free-text query and return the closest items."""

    query_emb = model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    client = _get_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=TOTAL_RESULTS,
        score_threshold=SIMILARITY_THRESHOLD,
    )

    hits = results.points
    client.close()
    return hits


def _build_qdrant_filter(parsed_filters: dict, strict_exclude: bool = False, strict_include: bool = False) -> Filter | None:
    """
    Convert the LLM parser output into a Qdrant Filter.

    parsed_filters example:
        {
            "include": {"type": ["shorts"], "style": ["sporty"]},
            "exclude": {"pattern": ["floral"]}
        }
    """
    must = []
    must_not = []

    # Include filters → must (only if strict_include is True)
    if strict_include:
        for field, values in parsed_filters.get("include", {}).items():
            if values:
                must.append(FieldCondition(key=field, match=MatchAny(any=values)))

    # Exclude filters → must_not (only if strict_exclude is True)
    if strict_exclude:
        for field, values in parsed_filters.get("exclude", {}).items():
            if values:
                must_not.append(FieldCondition(key=field, match=MatchAny(any=values)))

    if not must and not must_not:
        return None

    return Filter(must=must or None, must_not=must_not or None)


def filtered_search(
    query: str,
    parsed_filters: dict,
    model: SentenceTransformer,
    strict: bool = False,
    max_results: int | None = None,
    score_threshold: float | None = None,
    penalty_weight: float | None = None,
    boost_weight: float | None = None,
):
    """
    Semantic search with Qdrant metadata filters, applying strict or soft filtering.
    """
    query_emb = model.encode(
        [BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    client = _get_client()
    fetch_limit = MAX_QUERY_RESULTS
    
    excludes = parsed_filters.get("exclude", {})
    includes = parsed_filters.get("include", {})

    if strict:
        qfilter = _build_qdrant_filter(parsed_filters, strict_exclude=True, strict_include=True)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_emb,
            query_filter=qfilter,
            limit=fetch_limit,
        )
        hits = results.points
    else:
        if not excludes:
            # no values on "exclude" -> search normally without any filtering
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_emb,
                limit=fetch_limit,
            )
            hits = results.points
        else:
            # two searches
            qfilter_exclude = _build_qdrant_filter(parsed_filters, strict_exclude=True, strict_include=False)
            results_exclude = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_emb,
                query_filter=qfilter_exclude,
                limit=fetch_limit,
            )
            results_normal = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_emb,
                limit=fetch_limit,
            )
            
            # merge hits, avoiding duplicates
            hits_dict = {}
            for hit in results_normal.points + results_exclude.points:
                if hit.id not in hits_dict:
                    hits_dict[hit.id] = hit
            hits = list(hits_dict.values())
            
        p_weight = penalty_weight if penalty_weight is not None else PENALTY_WEIGHT
        b_weight = boost_weight if boost_weight is not None else BOOST_WEIGHT

        if excludes or includes:
            for hit in hits:
                # Penalize
                for field, values in excludes.items():
                    payload_val = hit.payload.get(field)
                    if payload_val is None:
                        continue
                    if isinstance(payload_val, list):
                        if any(v in payload_val for v in values):
                            hit.score -= p_weight
                    else:
                        if payload_val in values:
                            hit.score -= p_weight

                # # Check if it lacks anything in include
                # for field, values in includes.items():
                #     payload_val = hit.payload.get(field)
                #     has_include = False
                #     if payload_val is not None:
                #         if isinstance(payload_val, list):
                #             if any(v in payload_val for v in values):
                #                 has_include = True
                #         else:
                #             if payload_val in values:
                #                 has_include = True
                #     # The user said: "penalising... the ones that do not have something in the "include""
                #     if not has_include:
                #         hit.score -= p_weight

    client.close()

    # Filter out anything that drops below the score threshold after penalization
    final_threshold = score_threshold if score_threshold is not None else SIMILARITY_THRESHOLD
    valid_hits = [h for h in hits if h.score >= final_threshold]

    # Re-sort descending by score and trim to max_results (TOTAL_RESULTS)
    valid_hits.sort(key=lambda h: h.score, reverse=True)
    return valid_hits[:(max_results or TOTAL_RESULTS)]


def show_db_stats():
    client = _get_client()
    if not _collection_exists(client):
        print("\n  No vector DB found. Create one first.")
        client.close()
        return

    info = client.get_collection(COLLECTION_NAME)
    client.close()

    print("\n  " + "=" * 50)
    print("  Vector DB Statistics")
    print("  " + "=" * 50)
    print(f"  Collection         : {COLLECTION_NAME}")
    print(f"  Total points       : {info.points_count}")
    print(f"  Vector dimension   : {info.config.params.vectors.size}")
    print(f"  Distance metric    : {info.config.params.vectors.distance}")
    print(f"  Storage path       : {_QDRANT_DIR}")
    print(f"  Model              : {EMBEDDING_MODEL_NAME}")
    print(f"  Total results      : {TOTAL_RESULTS}")
    print(f"  Max query results  : {MAX_QUERY_RESULTS}")
    print(f"  Similarity threshold: {SIMILARITY_THRESHOLD}")
    print("  " + "=" * 50)


def interactive_search(model: SentenceTransformer):
    """Prompt loop: user types queries, sees ranked results."""

    client = _get_client()
    if not _collection_exists(client):
        print("\n  No vector DB found. Create one first.")
        client.close()
        return
    client.close()

    print("\n  " + "=" * 50)
    print("  Semantic Search")
    print("  " + "=" * 50)
    print(f"  Total results        : {TOTAL_RESULTS}")
    print(f"  Similarity threshold : {SIMILARITY_THRESHOLD}")
    print("  Type 'back' to return to menu.\n")

    while True:
        query = input("  Query > ").strip()
        if not query or query.lower() == "back":
            break

        hits = search(query, model)

        if not hits:
            print("  No results above the similarity threshold.\n")
            continue

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
            print(f"  Occasion: {p.get('occasion', '?')}")
            print(f"  Brand   : {p.get('brand', '?')}")
            print(f"  Price   : {p.get('price', '?')}")

            desc = p.get("description", "")
            if len(desc) > 200:
                desc = desc[:200] + "..."
            print(f"  Desc    : {desc}")
            print()

        print()


# ══════════════════════════════════════════════════════════════
# MENU
# ══════════════════════════════════════════════════════════════

def show_menu(db_ready: bool):
    clear()
    print("=" * 50)
    print("     Vector DB Manager  -  Viralytics")
    print("=" * 50)

    if not db_ready:
        print("  Vector DB: NOT created yet\n")
        print("  [1] Create Vector DB")
        print("  [2] DB Statistics")
        print("  [0] Exit")
    else:
        print("  Vector DB: Ready\n")
        print("  [1] Recreate Vector DB")
        print("  [2] Search")
        print("  [3] DB Statistics")
        print("  [0] Exit")

    print("=" * 50)


def main():
    model = _load_model()

    while True:
        client = _get_client()
        db_ready = _collection_exists(client) and _collection_point_count(client) > 0
        client.close()

        show_menu(db_ready)
        choice = input("  Choose an option: ").strip()

        if not db_ready:
            if choice == "1":
                nl_file = pick_nl_file()
                if nl_file:
                    build_vector_db(nl_file, model)
                input("\n  Press Enter to return to the menu...")
            elif choice == "2":
                show_db_stats()
                input("\n  Press Enter to return to the menu...")
            elif choice == "0":
                print("\n  Goodbye!\n")
                break
            else:
                print("\n  Invalid option.")
                input("  Press Enter to continue...")
        else:
            if choice == "1":
                nl_file = pick_nl_file()
                if nl_file:
                    build_vector_db(nl_file, model)
                input("\n  Press Enter to return to the menu...")
            elif choice == "2":
                interactive_search(model)
                input("\n  Press Enter to return to the menu...")
            elif choice == "3":
                show_db_stats()
                input("\n  Press Enter to return to the menu...")
            elif choice == "0":
                print("\n  Goodbye!\n")
                break
            else:
                print("\n  Invalid option.")
                input("  Press Enter to continue...")


if __name__ == "__main__":
    main()
