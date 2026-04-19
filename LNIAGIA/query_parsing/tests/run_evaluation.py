from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None


# ============================================================
# CONFIG
# ============================================================
INPUT_FOLDER = Path(__file__).resolve().parent / "input"
OUTPUT_FOLDER = Path(__file__).resolve().parent / "output"

INPUT_TEST_QUERIES_FILE = INPUT_FOLDER / "test_queries.json"

LLM_RESULTS_FILE = OUTPUT_FOLDER / "llm_results.json"
BASELINE_RESULTS_FILE = OUTPUT_FOLDER / "winner_baseline_results.json"
SUMMARY_RESULTS_FILE = OUTPUT_FOLDER / "evaluation_summary.json"

WINNER_RESULTS_DIR = Path(__file__).resolve().parent.parent / "qp_models" / "results"


# Import paths for local modules.
_this_dir = Path(__file__).resolve().parent
for import_dir in (_this_dir.parent, _this_dir.parent.parent):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from llm_query_parser import OLLAMA_MODEL, parse_query
from qp_models.baselines.exporter import load_latest_exported_parser


EPS = 1e-12


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: object) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(latencies_ms: Sequence[float]) -> dict:
    total_ms = float(sum(latencies_ms))

    return {
        "count": len(latencies_ms),
        "mean_ms": float(mean(latencies_ms)) if latencies_ms else 0.0,
        "median_ms": float(median(latencies_ms)) if latencies_ms else 0.0,
        "p95_ms": float(_percentile(latencies_ms, 0.95)),
        "total_ms": total_ms,
        "throughput_qps": _safe_div(len(latencies_ms) * 1000.0, total_ms),
    }


def _normalize_filter_block(block: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}

    if not isinstance(block, Mapping):
        return normalized

    for raw_field, raw_values in block.items():
        field = str(raw_field).strip().lower()
        if not field:
            continue

        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            continue

        values: list[str] = []
        seen: set[str] = set()

        for raw_value in raw_values:
            value = str(raw_value).strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)

        if values:
            normalized[field] = sorted(values)

    return normalized


def normalize_filters(payload: Any) -> dict:
    if not isinstance(payload, Mapping):
        return {"include": {}, "exclude": {}}

    include = _normalize_filter_block(payload.get("include", {}))
    exclude = _normalize_filter_block(payload.get("exclude", {}))

    # If the same value appears in include/exclude for the same field, keep it in exclude.
    for field, ex_values in list(exclude.items()):
        if field not in include:
            continue

        ex_set = set(ex_values)
        include[field] = [value for value in include[field] if value not in ex_set]
        if not include[field]:
            del include[field]

    return {
        "include": include,
        "exclude": exclude,
    }


def flatten_pairs(filters_payload: Mapping[str, Mapping[str, Sequence[str]]]) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()

    for polarity in ("include", "exclude"):
        block = filters_payload.get(polarity, {})
        if not isinstance(block, Mapping):
            continue

        for field, values in block.items():
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue

            for value in values:
                pairs.add((polarity, str(field).lower(), str(value).lower()))

    return pairs


def _pairs_to_rows(pairs: set[tuple[str, str, str]]) -> list[dict]:
    return [
        {
            "polarity": polarity,
            "field": field,
            "value": value,
        }
        for polarity, field, value in sorted(pairs)
    ]


def evaluate_prediction(expected_output: Any, predicted_output: Any) -> dict:
    expected_norm = normalize_filters(expected_output)
    predicted_norm = normalize_filters(predicted_output)

    gold_pairs = flatten_pairs(expected_norm)
    pred_pairs = flatten_pairs(predicted_norm)

    tp_pairs = gold_pairs & pred_pairs
    fp_pairs = pred_pairs - gold_pairs
    fn_pairs = gold_pairs - pred_pairs

    tp = len(tp_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1_score = _f1(precision, recall)

    return {
        "expected_normalized": expected_norm,
        "predicted_normalized": predicted_norm,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "exact_match": expected_norm == predicted_norm,
        "tp_pairs": _pairs_to_rows(tp_pairs),
        "fp_pairs": _pairs_to_rows(fp_pairs),
        "fn_pairs": _pairs_to_rows(fn_pairs),
    }


def _normalize_case(raw_case: Mapping[str, Any], index: int, category: str) -> dict | None:
    query = raw_case.get("query", "")
    expected_output = raw_case.get("expected_output", {})

    if not isinstance(query, str) or not query.strip():
        return None

    if not isinstance(expected_output, Mapping):
        expected_output = {}

    raw_id = raw_case.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        case_id = raw_id.strip()
    else:
        case_id = f"{category}_{index:03d}"

    return {
        "index": index,
        "id": case_id,
        "category": category,
        "query": query.strip(),
        "expected_output": dict(expected_output),
    }


def load_test_cases_by_category(path: Path) -> dict[str, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categorized: dict[str, list[dict]] = {}

    # Backward compatibility: old format was a single list.
    if isinstance(payload, list):
        category_name = "all_queries"
        bucket: list[dict] = []
        for idx, row in enumerate(payload, start=1):
            if not isinstance(row, Mapping):
                continue
            normalized = _normalize_case(row, idx, category_name)
            if normalized is not None:
                bucket.append(normalized)
        if bucket:
            categorized[category_name] = bucket
        return categorized

    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object or list in {path}, got {type(payload).__name__}.")

    for category_name, rows in payload.items():
        if not isinstance(rows, list):
            continue

        bucket: list[dict] = []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                continue
            normalized = _normalize_case(row, idx, str(category_name))
            if normalized is not None:
                bucket.append(normalized)

        if bucket:
            categorized[str(category_name)] = bucket

    if not categorized:
        raise ValueError(f"No valid categorized test cases found in {path}.")

    return categorized


def _aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict:
    total = len(records)

    total_tp = sum(int(record.get("tp", 0)) for record in records)
    total_fp = sum(int(record.get("fp", 0)) for record in records)
    total_fn = sum(int(record.get("fn", 0)) for record in records)

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _f1(micro_precision, micro_recall)

    exact_matches = sum(1 for record in records if bool(record.get("exact_match", False)))
    mean_query_f1 = float(mean(float(record.get("f1", 0.0)) for record in records)) if records else 0.0

    latencies = [float(record.get("latency_ms", 0.0)) for record in records]
    errors = [record for record in records if record.get("error")]

    fp_by_field: Counter[str] = Counter()
    fn_by_field: Counter[str] = Counter()

    for record in records:
        for pair in record.get("fp_pairs", []):
            field = str(pair.get("field", "")).strip().lower()
            if field:
                fp_by_field[field] += 1

        for pair in record.get("fn_pairs", []):
            field = str(pair.get("field", "")).strip().lower()
            if field:
                fn_by_field[field] += 1

    return {
        "query_count": total,
        "error_count": len(errors),
        "pair_tp": total_tp,
        "pair_fp": total_fp,
        "pair_fn": total_fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_match_rate": _safe_div(exact_matches, total),
        "mean_query_f1": mean_query_f1,
        "latency": _latency_summary(latencies),
        "top_false_positive_fields": fp_by_field.most_common(10),
        "top_false_negative_fields": fn_by_field.most_common(10),
    }


def evaluate_category(
    approach_name: str,
    category_name: str,
    predict_fn: Callable[[str], Any],
    cases: Sequence[Mapping[str, Any]],
) -> dict:
    start_total = perf_counter()
    total = len(cases)
    records: list[dict] = []

    bar_label = f"{approach_name}:{category_name}"
    iterable = tqdm(cases, desc=bar_label, unit="query") if tqdm is not None else cases

    for idx, case in enumerate(iterable, start=1):
        query = str(case.get("query", ""))
        expected_output = case.get("expected_output", {})

        start_query = perf_counter()
        error_text = ""

        try:
            predicted_raw = predict_fn(query)
        except Exception as exc:
            predicted_raw = {}
            error_text = f"{type(exc).__name__}: {exc}"

        latency_ms = (perf_counter() - start_query) * 1000.0
        comparison = evaluate_prediction(expected_output, predicted_raw)

        records.append(
            {
                "index": int(case.get("index", 0)),
                "id": case.get("id", ""),
                "category": case.get("category", category_name),
                "query": query,
                "latency_ms": latency_ms,
                "error": error_text,
                "expected_output": expected_output,
                "predicted_output_raw": predicted_raw,
                **comparison,
            }
        )

        if tqdm is None and (idx == 1 or idx % 10 == 0 or idx == total):
            progress = 100.0 * idx / max(1, total)
            print(f"[{bar_label}] {idx}/{total} ({progress:.1f}%)")

    elapsed_s = perf_counter() - start_total

    return {
        "category": category_name,
        "elapsed_s": elapsed_s,
        "metrics": _aggregate_metrics(records),
        "records": records,
    }


def evaluate_approach_by_category(
    approach_name: str,
    predict_fn: Callable[[str], Any],
    categorized_cases: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict:
    run_start = perf_counter()
    by_category: dict[str, dict] = {}
    all_records: list[dict] = []

    for category_name, cases in categorized_cases.items():
        category_result = evaluate_category(
            approach_name=approach_name,
            category_name=category_name,
            predict_fn=predict_fn,
            cases=cases,
        )
        by_category[category_name] = category_result
        all_records.extend(category_result["records"])

    return {
        "approach": approach_name,
        "elapsed_s": perf_counter() - run_start,
        "metrics": _aggregate_metrics(all_records),
        "by_category": by_category,
    }


def _quality_score(metrics: Mapping[str, Any]) -> float:
    micro_f1 = float(metrics.get("micro_f1", 0.0))
    exact_match = float(metrics.get("exact_match_rate", 0.0))
    return 0.7 * micro_f1 + 0.3 * exact_match


def _tradeoff_score(metrics: Mapping[str, Any]) -> float:
    quality = _quality_score(metrics)
    latency = metrics.get("latency", {})
    p95_ms = float(latency.get("p95_ms", 0.0)) if isinstance(latency, Mapping) else 0.0
    return quality / (1.0 + (p95_ms / 100.0))


def _pick_winner(
    llm_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    score_fn: Callable[[Mapping[str, Any]], float],
) -> tuple[str, dict]:
    llm_score = score_fn(llm_metrics)
    baseline_score = score_fn(baseline_metrics)

    if baseline_score > llm_score + EPS:
        winner = "winner_baseline"
    elif llm_score > baseline_score + EPS:
        winner = "llm_parser"
    else:
        winner = "tie"

    return winner, {
        "llm_parser": llm_score,
        "winner_baseline": baseline_score,
    }


def _head_to_head_counts(llm_records: Sequence[Mapping[str, Any]], baseline_records: Sequence[Mapping[str, Any]]) -> dict:
    llm_better = 0
    baseline_better = 0
    tie = 0

    for llm_row, base_row in zip(llm_records, baseline_records):
        llm_f1 = float(llm_row.get("f1", 0.0))
        base_f1 = float(base_row.get("f1", 0.0))
        delta = base_f1 - llm_f1

        if delta > EPS:
            baseline_better += 1
        elif delta < -EPS:
            llm_better += 1
        else:
            tie += 1

    return {
        "llm_better_count": llm_better,
        "baseline_better_count": baseline_better,
        "tie_count": tie,
    }


def _build_category_summary(
    llm_category_result: Mapping[str, Any],
    baseline_category_result: Mapping[str, Any],
) -> dict:
    llm_metrics = llm_category_result.get("metrics", {})
    baseline_metrics = baseline_category_result.get("metrics", {})

    winner_by_quality, quality_scores = _pick_winner(llm_metrics, baseline_metrics, _quality_score)
    winner_by_tradeoff, tradeoff_scores = _pick_winner(llm_metrics, baseline_metrics, _tradeoff_score)

    llm_records = llm_category_result.get("records", [])
    baseline_records = baseline_category_result.get("records", [])

    return {
        "winner_by_quality": winner_by_quality,
        "winner_by_tradeoff": winner_by_tradeoff,
        "quality_scores": quality_scores,
        "tradeoff_scores": tradeoff_scores,
        "time_s": {
            "llm_parser": float(llm_category_result.get("elapsed_s", 0.0)),
            "winner_baseline": float(baseline_category_result.get("elapsed_s", 0.0)),
        },
        "differences": {
            "delta_micro_f1_baseline_minus_llm": float(baseline_metrics.get("micro_f1", 0.0)) - float(llm_metrics.get("micro_f1", 0.0)),
            "delta_exact_match_rate_baseline_minus_llm": float(baseline_metrics.get("exact_match_rate", 0.0)) - float(llm_metrics.get("exact_match_rate", 0.0)),
            "delta_mean_latency_ms_baseline_minus_llm": float(baseline_metrics.get("latency", {}).get("mean_ms", 0.0)) - float(llm_metrics.get("latency", {}).get("mean_ms", 0.0)),
            "delta_p95_latency_ms_baseline_minus_llm": float(baseline_metrics.get("latency", {}).get("p95_ms", 0.0)) - float(llm_metrics.get("latency", {}).get("p95_ms", 0.0)),
            "head_to_head": _head_to_head_counts(llm_records, baseline_records),
        },
    }


def main() -> None:
    _ensure_dir(OUTPUT_FOLDER)
    categorized_cases = load_test_cases_by_category(INPUT_TEST_QUERIES_FILE)
    total_cases = sum(len(bucket) for bucket in categorized_cases.values())

    baseline_load_start = perf_counter()
    baseline_parser, baseline_manifest = load_latest_exported_parser(WINNER_RESULTS_DIR)
    baseline_load_ms = (perf_counter() - baseline_load_start) * 1000.0

    baseline_model_name = str(baseline_manifest.get("model", "unknown"))
    baseline_folder_size = baseline_manifest.get("folder_size", "?")
    baseline_model_tag = f"{baseline_model_name}_{baseline_folder_size}"

    print("=" * 72)
    print("Query Parser Evaluation")
    print("=" * 72)
    print(f"Input file            : {INPUT_TEST_QUERIES_FILE}")
    print(f"Output directory      : {OUTPUT_FOLDER}")
    print(f"Total test cases      : {total_cases}")
    print(f"Categories            : {', '.join(categorized_cases.keys())}")
    print(f"LLM model             : {OLLAMA_MODEL}")
    print(f"Winner baseline model : {baseline_model_tag}")
    print(f"Winner load time      : {baseline_load_ms:.2f} ms")
    print("=" * 72)

    llm_result = evaluate_approach_by_category(
        approach_name="llm_parser",
        predict_fn=lambda query: parse_query(query, verbose=False),
        categorized_cases=categorized_cases,
    )
    llm_result["model_name"] = OLLAMA_MODEL
    _write_json(LLM_RESULTS_FILE, llm_result)

    baseline_result = evaluate_approach_by_category(
        approach_name="winner_baseline",
        predict_fn=baseline_parser.parse,
        categorized_cases=categorized_cases,
    )
    baseline_result["model_name"] = baseline_model_tag
    baseline_result["winner_model"] = {
        "results_dir": str(WINNER_RESULTS_DIR),
        "model_tag": baseline_model_tag,
        "model": baseline_model_name,
        "folder_size": baseline_folder_size,
        "load_time_ms": baseline_load_ms,
        "manifest": baseline_manifest,
    }
    _write_json(BASELINE_RESULTS_FILE, baseline_result)

    categories_summary: dict[str, dict] = {}
    for category_name in categorized_cases.keys():
        llm_category_result = llm_result["by_category"][category_name]
        baseline_category_result = baseline_result["by_category"][category_name]
        categories_summary[category_name] = _build_category_summary(
            llm_category_result=llm_category_result,
            baseline_category_result=baseline_category_result,
        )

    summary = {
        "models": {
            "llm_parser": OLLAMA_MODEL,
            "winner_baseline": baseline_model_tag,
        },
        "categories": categories_summary,
    }

    _write_json(SUMMARY_RESULTS_FILE, summary)

    print("\nEvaluation complete.")
    print(f"LLM results file      : {LLM_RESULTS_FILE}")
    print(f"Baseline results file : {BASELINE_RESULTS_FILE}")
    print(f"Summary file          : {SUMMARY_RESULTS_FILE}")


if __name__ == "__main__":
    main()
