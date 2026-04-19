from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .data_utils import (
    QueryExample,
    StructuredPair,
    flatten_structured_pairs,
    fold_pairs_by_key,
    plain_pairs,
)


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


def _pair_polarity_map(pairs: Set[StructuredPair]) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for polarity, key, value in pairs:
        mapping[(key, value)] = polarity
    return mapping


def summarize_latency(latencies_ms: Sequence[float]) -> Dict[str, float]:
    total_ms = sum(latencies_ms)
    return {
        "count": float(len(latencies_ms)),
        "mean_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "median_ms": statistics.median(latencies_ms) if latencies_ms else 0.0,
        "p95_ms": _percentile(latencies_ms, 0.95),
        "total_ms": total_ms,
        "throughput_qps": _safe_div(len(latencies_ms) * 1000.0, total_ms),
    }


def compute_token_metrics(
    gold_label_sequences: Sequence[Sequence[str]],
    pred_label_sequences: Sequence[Sequence[str]],
) -> Dict[str, object]:
    gold_flat: List[str] = []
    pred_flat: List[str] = []

    for gold_seq, pred_seq in zip(gold_label_sequences, pred_label_sequences):
        length = min(len(gold_seq), len(pred_seq))
        gold_flat.extend(list(gold_seq)[:length])
        pred_flat.extend(list(pred_seq)[:length])

    if not gold_flat:
        return {
            "token_accuracy": 0.0,
            "token_micro_precision": 0.0,
            "token_micro_recall": 0.0,
            "token_micro_f1": 0.0,
            "token_macro_precision": 0.0,
            "token_macro_recall": 0.0,
            "token_macro_f1": 0.0,
            "per_label": {},
        }

    labels = sorted({label for label in gold_flat + pred_flat if label != "O"})

    token_accuracy = accuracy_score(gold_flat, pred_flat)

    if labels:
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            gold_flat,
            pred_flat,
            labels=labels,
            average="micro",
            zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gold_flat,
            pred_flat,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    else:
        micro_p = micro_r = micro_f1 = 0.0
        macro_p = macro_r = macro_f1 = 0.0

    per_label: Dict[str, Dict[str, float]] = {}
    if labels:
        p_vals, r_vals, f_vals, supports = precision_recall_fscore_support(
            gold_flat,
            pred_flat,
            labels=labels,
            average=None,
            zero_division=0,
        )
        for idx, label in enumerate(labels):
            per_label[label] = {
                "precision": float(p_vals[idx]),
                "recall": float(r_vals[idx]),
                "f1": float(f_vals[idx]),
                "support": float(supports[idx]),
            }

    return {
        "token_accuracy": float(token_accuracy),
        "token_micro_precision": float(micro_p),
        "token_micro_recall": float(micro_r),
        "token_micro_f1": float(micro_f1),
        "token_macro_precision": float(macro_p),
        "token_macro_recall": float(macro_r),
        "token_macro_f1": float(macro_f1),
        "per_label": per_label,
    }


def evaluate_structured_predictions(
    examples: Sequence[QueryExample],
    predictions: Sequence[dict],
    latencies_ms: Sequence[float],
    predicted_label_sequences: Sequence[Sequence[str]] | None = None,
) -> Dict[str, object]:
    if len(examples) != len(predictions):
        raise ValueError("Examples and predictions must have identical lengths.")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_matches = 0

    negation_total = 0
    negation_correct = 0

    by_key_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "support": 0.0})

    per_query: List[dict] = []
    pred_token_sequences: List[List[str]] = []

    for idx, (example, prediction) in enumerate(zip(examples, predictions), start=1):
        gold_pairs = flatten_structured_pairs(example.include, example.exclude)
        pred_pairs = flatten_structured_pairs(
            prediction.get("include", {}),
            prediction.get("exclude", {}),
        )

        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if gold_pairs == pred_pairs:
            exact_matches += 1

        gold_plain = plain_pairs(gold_pairs)
        pred_plain = plain_pairs(pred_pairs)
        overlap_plain = gold_plain & pred_plain

        gold_polarity = _pair_polarity_map(gold_pairs)
        pred_polarity = _pair_polarity_map(pred_pairs)

        negation_total += len(overlap_plain)
        for key_value in overlap_plain:
            if gold_polarity.get(key_value) == pred_polarity.get(key_value):
                negation_correct += 1

        gold_by_key = fold_pairs_by_key(gold_pairs)
        pred_by_key = fold_pairs_by_key(pred_pairs)

        for key in sorted(set(gold_by_key) | set(pred_by_key)):
            key_gold = gold_by_key.get(key, set())
            key_pred = pred_by_key.get(key, set())

            key_tp = len(key_gold & key_pred)
            key_fp = len(key_pred - key_gold)
            key_fn = len(key_gold - key_pred)

            by_key_counts[key]["tp"] += key_tp
            by_key_counts[key]["fp"] += key_fp
            by_key_counts[key]["fn"] += key_fn
            by_key_counts[key]["support"] += len(key_gold)

        query_precision = _safe_div(tp, tp + fp)
        query_recall = _safe_div(tp, tp + fn)
        query_f1 = _f1(query_precision, query_recall)

        per_query.append(
            {
                "index": idx,
                "id": example.example_id,
                "query": example.query,
                "precision": query_precision,
                "recall": query_recall,
                "f1": query_f1,
                "exact_match": gold_pairs == pred_pairs,
                "gold_pair_count": len(gold_pairs),
                "pred_pair_count": len(pred_pairs),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "runtime_ms": latencies_ms[idx - 1] if idx - 1 < len(latencies_ms) else 0.0,
                "gold": {
                    "include": example.include,
                    "exclude": example.exclude,
                },
                "prediction": {
                    "include": prediction.get("include", {}),
                    "exclude": prediction.get("exclude", {}),
                },
            }
        )

        pred_labels = prediction.get("predicted_bio_labels", [])
        if isinstance(pred_labels, Sequence):
            pred_token_sequences.append(list(pred_labels))

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    micro_f1 = _f1(micro_precision, micro_recall)

    per_key: Dict[str, Dict[str, float]] = {}
    key_f1_values: List[float] = []

    for key in sorted(by_key_counts):
        key_tp = by_key_counts[key]["tp"]
        key_fp = by_key_counts[key]["fp"]
        key_fn = by_key_counts[key]["fn"]

        key_precision = _safe_div(key_tp, key_tp + key_fp)
        key_recall = _safe_div(key_tp, key_tp + key_fn)
        key_f1 = _f1(key_precision, key_recall)

        key_f1_values.append(key_f1)

        per_key[key] = {
            "precision": key_precision,
            "recall": key_recall,
            "f1": key_f1,
            "support": by_key_counts[key]["support"],
        }

    macro_f1 = statistics.mean(key_f1_values) if key_f1_values else 0.0

    token_metrics = compute_token_metrics(
        [example.bio_labels for example in examples],
        predicted_label_sequences if predicted_label_sequences is not None else pred_token_sequences,
    )

    latency_summary = summarize_latency(latencies_ms)

    return {
        "structured_micro_precision": micro_precision,
        "structured_micro_recall": micro_recall,
        "structured_micro_f1": micro_f1,
        "structured_macro_f1": macro_f1,
        "strict_exact_match_rate": _safe_div(exact_matches, len(examples)),
        "negation_accuracy": _safe_div(negation_correct, negation_total),
        "pair_tp": float(total_tp),
        "pair_fp": float(total_fp),
        "pair_fn": float(total_fn),
        "per_key": per_key,
        "token_metrics": token_metrics,
        "latency": latency_summary,
        "per_query": per_query,
    }


def compute_quality_score(metrics: Mapping[str, float]) -> float:
    micro_f1 = float(metrics.get("structured_micro_f1", 0.0))
    macro_f1 = float(metrics.get("structured_macro_f1", 0.0))
    negation_acc = float(metrics.get("negation_accuracy", 0.0))
    exact_match = float(metrics.get("strict_exact_match_rate", 0.0))

    return 0.45 * micro_f1 + 0.25 * macro_f1 + 0.20 * negation_acc + 0.10 * exact_match


def compute_tradeoff_score(quality_score: float, p95_latency_ms: float) -> float:
    penalty = 1.0 + (p95_latency_ms / 100.0)
    return quality_score / penalty
