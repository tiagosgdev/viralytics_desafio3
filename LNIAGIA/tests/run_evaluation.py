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

from DB.vector.VectorDBManager import (
    _load_model,
    _get_client,
    _collection_exists,
    _collection_point_count,
    filtered_search,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOTAL_RESULTS,
    SIMILARITY_THRESHOLD,
)
from llm_query_parser import parse_query

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
    print(f"  Max results: {TOTAL_RESULTS}")
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
    all_metrics_strict: list[dict] = []
    all_metrics_non_strict: list[dict] = []

    for idx, q in enumerate(queries, 1):
        query_text = q["query"]
        expected_ids = q["expected_item_ids"]

        print(f"  [{idx}/{len(queries)}] {query_text[:70]}...")

        # Parse query with LLM
        parsed_filters = parse_query(query_text, verbose=False)

        # Strict Search
        hits_strict = filtered_search(query_text, parsed_filters, model, strict=True)
        metrics_strict = _compute_metrics(expected_ids, hits_strict)

        # Non-Strict Search
        hits_non_strict = filtered_search(query_text, parsed_filters, model, strict=False)
        metrics_non_strict = _compute_metrics(expected_ids, hits_non_strict)

        # Build per-query output - Strict
        returned_items_strict = []
        for rank, h in enumerate(hits_strict, 1):
            returned_items_strict.append({
                "rank": rank,
                "item_id": h.payload.get("item_id"),
                "score": round(h.score, 4),
                "is_expected": h.payload.get("item_id") in set(expected_ids),
                "payload": h.payload,
            })

        query_result_strict = {
            "query": query_text,
            "parsed_filters": parsed_filters,
            "strict_mode": True,
            "expected_item_ids": expected_ids,
            "evaluation": metrics_strict,
            "results": returned_items_strict,
        }

        # Build per-query output - Non Strict
        returned_items_non_strict = []
        for rank, h in enumerate(hits_non_strict, 1):
            returned_items_non_strict.append({
                "rank": rank,
                "item_id": h.payload.get("item_id"),
                "score": round(h.score, 4),
                "is_expected": h.payload.get("item_id") in set(expected_ids),
                "payload": h.payload,
            })

        query_result_non_strict = {
            "query": query_text,
            "parsed_filters": parsed_filters,
            "strict_mode": False,
            "expected_item_ids": expected_ids,
            "evaluation": metrics_non_strict,
            "results": returned_items_non_strict,
        }

        # Save individual query file - Strict
        filename_strict = f"query_{idx:02d}_strict.json"
        with open(run_dir / filename_strict, "w", encoding="utf-8") as f:
            json.dump(query_result_strict, f, indent=2, ensure_ascii=False)

        # Save individual query file - Non Strict
        filename_non_strict = f"query_{idx:02d}_non_strict.json"
        with open(run_dir / filename_non_strict, "w", encoding="utf-8") as f:
            json.dump(query_result_non_strict, f, indent=2, ensure_ascii=False)

        all_metrics_strict.append({
            "query_index": idx,
            "query": query_text,
            **metrics_strict,
        })
        
        all_metrics_non_strict.append({
            "query_index": idx,
            "query": query_text,
            **metrics_non_strict,
        })

        tag_strict = "OK" if metrics_strict["recall"] > 0 else "--"
        tag_non_strict = "OK" if metrics_non_strict["recall"] > 0 else "--"
        print(f"       Strict: P={metrics_strict['precision']:.2f}  R={metrics_strict['recall']:.2f}  "
              f"F1={metrics_strict['f1']:.2f}  avg_s={metrics_strict['avg_score']:.4f}  [{tag_strict}]")
        print(f"   Non-Strict: P={metrics_non_strict['precision']:.2f}  R={metrics_non_strict['recall']:.2f}  "
              f"F1={metrics_non_strict['f1']:.2f}  avg_s={metrics_non_strict['avg_score']:.4f}  [{tag_non_strict}]")

    # ── Summary ──
    avg_p_strict = sum(m["precision"] for m in all_metrics_strict) / len(all_metrics_strict)
    avg_r_strict = sum(m["recall"] for m in all_metrics_strict) / len(all_metrics_strict)
    avg_f1_strict = sum(m["f1"] for m in all_metrics_strict) / len(all_metrics_strict)
    avg_s_strict = sum(m["avg_score"] for m in all_metrics_strict) / len(all_metrics_strict)

    avg_p_non_strict = sum(m["precision"] for m in all_metrics_non_strict) / len(all_metrics_non_strict)
    avg_r_non_strict = sum(m["recall"] for m in all_metrics_non_strict) / len(all_metrics_non_strict)
    avg_f1_non_strict = sum(m["f1"] for m in all_metrics_non_strict) / len(all_metrics_non_strict)
    avg_s_non_strict = sum(m["avg_score"] for m in all_metrics_non_strict) / len(all_metrics_non_strict)

    summary = {
        "run_timestamp": timestamp,
        "config": {
            "model": EMBEDDING_MODEL_NAME,
            "max_results": TOTAL_RESULTS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "collection": COLLECTION_NAME,
            "total_points": points,
        },
        "aggregate_strict": {
            "mean_precision": round(avg_p_strict, 4),
            "mean_recall": round(avg_r_strict, 4),
            "mean_f1": round(avg_f1_strict, 4),
            "mean_avg_score": round(avg_s_strict, 4),
        },
        "aggregate_non_strict": {
            "mean_precision": round(avg_p_non_strict, 4),
            "mean_recall": round(avg_r_non_strict, 4),
            "mean_f1": round(avg_f1_non_strict, 4),
            "mean_avg_score": round(avg_s_non_strict, 4),
        },
        "total_queries": len(all_metrics_strict),
        "per_query_strict": all_metrics_strict,
        "per_query_non_strict": all_metrics_non_strict,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries        : {len(all_metrics_strict)}")
    print("\n  [STRICT LIST]")
    print(f"  Mean Precision : {avg_p_strict:.4f}")
    print(f"  Mean Recall    : {avg_r_strict:.4f}")
    print(f"  Mean F1        : {avg_f1_strict:.4f}")
    print(f"  Mean Avg Score : {avg_s_strict:.4f}")
    print("\n  [NON-STRICT LIST]")
    print(f"  Mean Precision : {avg_p_non_strict:.4f}")
    print(f"  Mean Recall    : {avg_r_non_strict:.4f}")
    print(f"  Mean F1        : {avg_f1_non_strict:.4f}")
    print(f"  Mean Avg Score : {avg_s_non_strict:.4f}")
    print(f"\n  Results saved to: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
