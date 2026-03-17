# ════════════════════════════════════════════════════════════
# run_evaluation.py
# Runs every query in input/test_queries.json against the
# vector DB and writes per-query result files + a summary
# into output/.
# ════════════════════════════════════════════════════════════

import json
import sys
from pathlib import Path
from datetime import datetime

# Allow importing from the parent vector/ folder
_SCRIPT_DIR = Path(__file__).resolve().parent
_VECTOR_DIR = _SCRIPT_DIR.parent
if str(_VECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VECTOR_DIR))

from VectorDBManager import (
    _load_model,
    _get_client,
    _collection_exists,
    _collection_point_count,
    search,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    MAX_RESULTS,
    SIMILARITY_THRESHOLD,
)

# ── paths ──
_INPUT_DIR = _SCRIPT_DIR / "input"
_OUTPUT_DIR = _SCRIPT_DIR / "output"
_QUERIES_FILE = _INPUT_DIR / "test_queries.json"


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def _compute_metrics(expected_ids: list[int], returned_hits: list) -> dict:
    """
    Compute retrieval-quality metrics for a single query.

    Returns dict with:
        precision    – fraction of returned items that are relevant
        recall       – fraction of expected items that were returned
        f1           – harmonic mean of precision and recall
        avg_score    – mean similarity score of all returned items
        hit_ids      – returned item_ids that are in expected
        miss_ids     – expected item_ids NOT returned
        extra_ids    – returned item_ids NOT in expected
    """
    expected_set = set(expected_ids)
    returned_ids = [h.payload.get("item_id") for h in returned_hits]
    returned_set = set(returned_ids)

    hits = expected_set & returned_set
    misses = expected_set - returned_set
    extras = returned_set - expected_set

    precision = len(hits) / len(returned_set) if returned_set else 0.0
    recall = len(hits) / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    scores = [h.score for h in returned_hits]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_score": round(avg_score, 4),
        "returned_count": len(returned_ids),
        "expected_count": len(expected_ids),
        "hit_count": len(hits),
        "hit_ids": sorted(hits),
        "miss_ids": sorted(misses),
        "extra_ids": sorted(extras),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ── Pre-flight checks ──
    client = _get_client()
    if not _collection_exists(client):
        print("  Vector DB not found. Run VectorDBManager first to create it.")
        client.close()
        return
    points = _collection_point_count(client)
    client.close()

    print("=" * 60)
    print("  Evaluation Runner")
    print("=" * 60)
    print(f"  Collection : {COLLECTION_NAME}  ({points} points)")
    print(f"  Model      : {EMBEDDING_MODEL_NAME}")
    print(f"  Max results: {MAX_RESULTS}")
    print(f"  Threshold  : {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    # ── Load queries ──
    with open(_QUERIES_FILE, "r", encoding="utf-8") as f:
        queries: list[dict] = json.load(f)

    print(f"\n  Loaded {len(queries)} test queries from {_QUERIES_FILE.name}\n")

    # ── Load model once ──
    model = _load_model()

    # ── Prepare output folder ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Run each query ──
    all_metrics: list[dict] = []

    for idx, q in enumerate(queries, 1):
        query_text = q["query"]
        expected_ids = q["expected_item_ids"]

        print(f"  [{idx}/{len(queries)}] {query_text[:70]}...")

        hits = search(query_text, model)
        metrics = _compute_metrics(expected_ids, hits)

        # Build per-query output
        returned_items = []
        for rank, h in enumerate(hits, 1):
            returned_items.append({
                "rank": rank,
                "item_id": h.payload.get("item_id"),
                "score": round(h.score, 4),
                "is_expected": h.payload.get("item_id") in set(expected_ids),
                "payload": h.payload,
            })

        query_result = {
            "query": query_text,
            "expected_item_ids": expected_ids,
            "evaluation": metrics,
            "results": returned_items,
        }

        # Save individual query file
        filename = f"query_{idx:02d}.json"
        with open(run_dir / filename, "w", encoding="utf-8") as f:
            json.dump(query_result, f, indent=2, ensure_ascii=False)

        all_metrics.append({
            "query_index": idx,
            "query": query_text,
            **metrics,
        })

        tag = "OK" if metrics["recall"] > 0 else "--"
        print(f"           P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  "
              f"F1={metrics['f1']:.2f}  avg_s={metrics['avg_score']:.4f}  [{tag}]")

    # ── Summary ──
    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
    avg_s = sum(m["avg_score"] for m in all_metrics) / len(all_metrics)

    summary = {
        "run_timestamp": timestamp,
        "config": {
            "model": EMBEDDING_MODEL_NAME,
            "max_results": MAX_RESULTS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "collection": COLLECTION_NAME,
            "total_points": points,
        },
        "aggregate": {
            "mean_precision": round(avg_p, 4),
            "mean_recall": round(avg_r, 4),
            "mean_f1": round(avg_f1, 4),
            "mean_avg_score": round(avg_s, 4),
            "total_queries": len(all_metrics),
        },
        "per_query": all_metrics,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries        : {len(all_metrics)}")
    print(f"  Mean Precision : {avg_p:.4f}")
    print(f"  Mean Recall    : {avg_r:.4f}")
    print(f"  Mean F1        : {avg_f1:.4f}")
    print(f"  Mean Avg Score : {avg_s:.4f}")
    print(f"\n  Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
