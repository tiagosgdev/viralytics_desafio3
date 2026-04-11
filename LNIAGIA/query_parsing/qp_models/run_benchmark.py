from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .baselines.crf_model import CRFParser
from .baselines.data_utils import (
    QueryExample,
    build_leakage_report,
    discover_dataset_folders,
    ensure_dir,
    file_sha256,
    load_examples,
    normalize_filter_block,
    write_json,
)
from .baselines.exporter import export_winner_parser
from .baselines.evaluation_metrics import (
    compute_quality_score,
    compute_tradeoff_score,
    evaluate_structured_predictions,
)
from .baselines.label_projection import project_bio_labels
from .baselines.reporting import (
    plot_folder_metrics,
    plot_global_delta,
    plot_global_latency_vs_quality,
    plot_global_learning_curves,
    write_csv,
)
from .baselines.rule_based import RuleBasedParser


DEFAULT_FULL_OUTPUTS_DIR = Path("LNIAGIA/query_parsing_models/data_generation/full_outputs")
DEFAULT_RESULTS_DIR = Path("LNIAGIA/query_parsing_models/results")


def _to_relative_path_str(path: Path, base: Path | None = None) -> str:
    root = base or Path.cwd()

    try:
        return str(path.relative_to(root))
    except ValueError:
        pass

    try:
        return os.path.relpath(path, root)
    except ValueError:
        # Different drives on Windows can fail relpath.
        return str(path)


def _parse_sizes(raw: str) -> List[int]:
    sizes: List[int] = []

    parts: List[str]
    if isinstance(raw, str):
        parts = [raw]
    else:
        parts = [str(item) for item in raw]

    for part in parts:
        for chunk in part.split(","):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            sizes.append(int(cleaned))

    deduped: List[int] = []
    seen = set()
    for size in sizes:
        if size in seen:
            continue
        seen.add(size)
        deduped.append(size)

    return deduped


def _normalized_prediction(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {"include": {}, "exclude": {}}

    include = normalize_filter_block(payload.get("include", {}))
    exclude = normalize_filter_block(payload.get("exclude", {}))

    return {
        "include": include,
        "exclude": exclude,
    }


def _evaluate_parser(parser, test_examples: Sequence[QueryExample]) -> Tuple[dict, List[dict], List[float]]:
    predictions: List[dict] = []
    latencies: List[float] = []
    pred_labels: List[List[str]] = []

    for example in test_examples:
        start = perf_counter()
        raw_prediction = parser.parse(example.query)
        elapsed_ms = (perf_counter() - start) * 1000.0

        normalized = _normalized_prediction(raw_prediction)
        labels = project_bio_labels(example.tokens, normalized["include"], normalized["exclude"])

        predictions.append(
            {
                "include": normalized["include"],
                "exclude": normalized["exclude"],
                "predicted_bio_labels": labels,
            }
        )
        pred_labels.append(labels)
        latencies.append(elapsed_ms)

    metrics = evaluate_structured_predictions(
        examples=test_examples,
        predictions=predictions,
        latencies_ms=latencies,
        predicted_label_sequences=pred_labels,
    )

    return metrics, predictions, latencies


def _build_row(
    folder_size: int,
    folder_name: str,
    model: str,
    metrics: dict,
    fit_time_s: float,
    extra: dict | None = None,
) -> dict:
    token = metrics.get("token_metrics", {})
    latency = metrics.get("latency", {})

    quality_score = compute_quality_score(metrics)
    tradeoff_score = compute_tradeoff_score(quality_score, float(latency.get("p95_ms", 0.0)))

    row = {
        "folder_size": folder_size,
        "folder_name": folder_name,
        "model": model,
        "structured_micro_precision": float(metrics.get("structured_micro_precision", 0.0)),
        "structured_micro_recall": float(metrics.get("structured_micro_recall", 0.0)),
        "structured_micro_f1": float(metrics.get("structured_micro_f1", 0.0)),
        "structured_macro_f1": float(metrics.get("structured_macro_f1", 0.0)),
        "negation_accuracy": float(metrics.get("negation_accuracy", 0.0)),
        "strict_exact_match_rate": float(metrics.get("strict_exact_match_rate", 0.0)),
        "token_accuracy": float(token.get("token_accuracy", 0.0)),
        "token_micro_f1": float(token.get("token_micro_f1", 0.0)),
        "token_macro_f1": float(token.get("token_macro_f1", 0.0)),
        "latency_mean_ms": float(latency.get("mean_ms", 0.0)),
        "latency_median_ms": float(latency.get("median_ms", 0.0)),
        "latency_p95_ms": float(latency.get("p95_ms", 0.0)),
        "throughput_qps": float(latency.get("throughput_qps", 0.0)),
        "fit_time_s": float(fit_time_s),
        "quality_score": float(quality_score),
        "tradeoff_score": float(tradeoff_score),
    }

    if extra:
        row.update(extra)

    return row


def _trim_metrics(metrics: dict) -> dict:
    payload = dict(metrics)
    payload.pop("per_query", None)
    return payload


def _row_key(row: Mapping[str, Any]) -> Tuple[int, str]:
    return int(row.get("folder_size", 0)), str(row.get("model", "")).strip().lower()


def _print_ranked_summary(ranked_rows: Sequence[dict], limit: int = 10) -> None:
    print("\nTop candidates by tradeoff_score:")
    print(" rank | model      | size | tradeoff | quality | micro_f1 | p95_ms | fit_s")
    print("------|------------|------|----------|---------|----------|--------|------")

    shown = ranked_rows[:limit]
    for rank, row in enumerate(shown, start=1):
        print(
            " {:>4} | {:<10} | {:>4} | {:>8.4f} | {:>7.4f} | {:>8.4f} | {:>6.2f} | {:>4.1f}".format(
                rank,
                str(row.get("model", "")),
                int(row.get("folder_size", 0)),
                float(row.get("tradeoff_score", 0.0)),
                float(row.get("quality_score", 0.0)),
                float(row.get("structured_micro_f1", 0.0)),
                float(row.get("latency_p95_ms", 0.0)),
                float(row.get("fit_time_s", 0.0)),
            )
        )

    if len(ranked_rows) > limit:
        remaining = len(ranked_rows) - limit
        print(f"  ... {remaining} more row(s) available. Use --winner-rank N to pick any entry.")


def _select_winner_row(args: argparse.Namespace, ranked_rows: Sequence[dict]) -> dict | None:
    if args.no_export_winner:
        print("\nWinner export skipped (--no-export-winner).")
        return None

    if not ranked_rows:
        return None

    if args.winner_rank is not None:
        if args.winner_rank < 1 or args.winner_rank > len(ranked_rows):
            raise ValueError(
                f"--winner-rank must be between 1 and {len(ranked_rows)}, got {args.winner_rank}."
            )

        print(f"\nWinner rank selected by CLI: {args.winner_rank}")
        return dict(ranked_rows[args.winner_rank - 1])

    if not sys.stdin.isatty():
        print("\nNon-interactive terminal detected; exporting top-ranked entry by default.")
        return dict(ranked_rows[0])

    prompt = (
        f"\nSelect winner rank to export [1-{len(ranked_rows)}] "
        "(Enter=1, 'skip' to skip export): "
    )

    while True:
        raw = input(prompt).strip().lower()

        if raw in {"", "1"}:
            return dict(ranked_rows[0])

        if raw in {"skip", "s", "n", "no"}:
            return None

        try:
            rank = int(raw)
        except ValueError:
            print("Please enter a valid rank number or 'skip'.")
            continue

        if 1 <= rank <= len(ranked_rows):
            return dict(ranked_rows[rank - 1])

        print(f"Rank must be between 1 and {len(ranked_rows)}.")


def run_benchmark(args: argparse.Namespace) -> None:
    full_outputs_dir = Path(args.full_outputs_dir)
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    discovered = discover_dataset_folders(full_outputs_dir)
    requested_sizes = _parse_sizes(args.sizes)

    if not requested_sizes:
        raise ValueError("No dataset sizes were provided.")

    fixed_test_path: Path
    fixed_test_selector = ""
    selected_fixed_size: int | None = None
    if args.fixed_test_file:
        fixed_test_path = Path(args.fixed_test_file)
        if not fixed_test_path.exists():
            raise FileNotFoundError(f"Fixed test file does not exist: {fixed_test_path}")
        fixed_test_selector = "explicit --fixed-test-file"
    else:
        preferred_size = args.fixed_test_size if args.fixed_test_size is not None else max(requested_sizes)
        fixed_folder = discovered.get(preferred_size)

        if fixed_folder is None:
            available_sizes = sorted(discovered.keys())
            if not available_sizes:
                raise FileNotFoundError(
                    f"Unable to find any valid dataset folders in {full_outputs_dir}"
                )

            if args.fixed_test_size is None:
                requested_available = sorted(size for size in requested_sizes if size in discovered)
                fallback_size = requested_available[-1] if requested_available else available_sizes[-1]
                print(
                    f"[WARN] Largest requested test size {preferred_size} is unavailable; "
                    f"falling back to size {fallback_size} ({discovered[fallback_size].name})."
                )
                fixed_folder = discovered[fallback_size]
                selected_fixed_size = fallback_size
                fixed_test_selector = "largest available among requested sizes"
            else:
                fallback_size = available_sizes[-1]
                print(
                    f"[WARN] Requested --fixed-test-size={preferred_size} is unavailable; "
                    f"falling back to size {fallback_size} ({discovered[fallback_size].name})."
                )
                fixed_folder = discovered[fallback_size]
                selected_fixed_size = fallback_size
                fixed_test_selector = "fallback from explicit --fixed-test-size"
        else:
            selected_fixed_size = preferred_size
            if args.fixed_test_size is None:
                fixed_test_selector = "largest requested size"
            else:
                fixed_test_selector = "explicit --fixed-test-size"

        fixed_test_path = fixed_folder / "test.json"

    fixed_test_examples = load_examples(fixed_test_path)
    if args.max_test_samples > 0:
        fixed_test_examples = fixed_test_examples[: args.max_test_samples]

    fixed_test_sha = file_sha256(fixed_test_path)

    global_rows: List[dict] = []
    run_manifest: List[dict] = []
    trained_parsers: Dict[Tuple[int, str], object] = {}

    print("=" * 72)
    print("Query Parsing Baseline Benchmark")
    print("=" * 72)
    print(f"Full outputs directory : {_to_relative_path_str(full_outputs_dir)}")
    print(f"Fixed test set        : {_to_relative_path_str(fixed_test_path)}")
    if selected_fixed_size is not None:
        print(f"Fixed test source     : size={selected_fixed_size} ({fixed_test_selector})")
    else:
        print(f"Fixed test source     : {fixed_test_selector}")
    print(f"Fixed test size       : {len(fixed_test_examples)}")
    print(f"Fixed test SHA-256    : {fixed_test_sha}")
    print(f"Requested sizes       : {requested_sizes}")
    print("=" * 72)

    for size in requested_sizes:
        folder_path = discovered.get(size)
        if folder_path is None:
            message = f"Dataset folder for size {size} was not found under {full_outputs_dir}."
            if args.skip_missing:
                print(f"[SKIP] {message}")
                continue
            raise FileNotFoundError(message)

        print(f"\n--- Running folder size {size} ({folder_path.name}) ---")

        train_examples = load_examples(folder_path / "train.json")
        val_examples = load_examples(folder_path / "val.json")

        leakage = build_leakage_report(train_examples, val_examples, fixed_test_examples)

        folder_result_dir = results_dir / f"{size}_{folder_path.name}"
        ensure_dir(folder_result_dir)

        # Rule-based baseline.
        rule_parser = RuleBasedParser()
        start = perf_counter()
        rule_parser.fit(train_examples, val_examples)
        rule_fit_time = perf_counter() - start
        trained_parsers[(size, "rule_based")] = rule_parser

        rule_metrics, rule_predictions, rule_latencies = _evaluate_parser(rule_parser, fixed_test_examples)
        rule_row = _build_row(
            folder_size=size,
            folder_name=folder_path.name,
            model="rule_based",
            metrics=rule_metrics,
            fit_time_s=rule_fit_time,
            extra={
                "val_model_score": 0.0,
                "train_val_overlap_queries": leakage.get("overlap_query_count", 0),
            },
        )

        # CRF baseline.
        crf_parser = CRFParser(
            random_state=args.seed,
            max_iterations=args.crf_max_iterations,
        )
        start = perf_counter()
        crf_training_info = crf_parser.fit(train_examples, val_examples)
        crf_fit_time = perf_counter() - start
        trained_parsers[(size, "crf")] = crf_parser

        crf_metrics, crf_predictions, crf_latencies = _evaluate_parser(crf_parser, fixed_test_examples)
        crf_row = _build_row(
            folder_size=size,
            folder_name=folder_path.name,
            model="crf",
            metrics=crf_metrics,
            fit_time_s=crf_fit_time,
            extra={
                "val_model_score": float(crf_training_info.get("val_weighted_f1", 0.0)),
                "best_c1": crf_training_info.get("best_c1", ""),
                "best_c2": crf_training_info.get("best_c2", ""),
                "train_val_overlap_queries": leakage.get("overlap_query_count", 0),
            },
        )

        folder_rows = [rule_row, crf_row]
        global_rows.extend(folder_rows)

        # Save per-folder artifacts.
        write_json(folder_result_dir / "rule_based_per_query.json", rule_metrics.get("per_query", []))
        write_json(folder_result_dir / "crf_per_query.json", crf_metrics.get("per_query", []))

        write_json(
            folder_result_dir / "summary.json",
            {
                "folder_size": size,
                "folder_name": folder_path.name,
                "train_count": len(train_examples),
                "val_count": len(val_examples),
                "fixed_test_count": len(fixed_test_examples),
                "fixed_test_path": _to_relative_path_str(fixed_test_path),
                "fixed_test_sha256": fixed_test_sha,
                "leakage_report": leakage,
                "models": {
                    "rule_based": {
                        "fit_time_s": rule_fit_time,
                        "metrics": _trim_metrics(rule_metrics),
                    },
                    "crf": {
                        "fit_time_s": crf_fit_time,
                        "training_info": crf_training_info,
                        "metrics": _trim_metrics(crf_metrics),
                    },
                },
            },
        )

        write_csv(
            folder_result_dir / "model_metrics.csv",
            folder_rows,
            fieldnames=list(folder_rows[0].keys()),
        )

        if not args.disable_plots:
            plot_folder_metrics(
                folder_size=size,
                model_rows=folder_rows,
                latency_map={
                    "rule_based": rule_latencies,
                    "crf": crf_latencies,
                },
                per_key_map={
                    "rule_based": rule_metrics.get("per_key", {}),
                    "crf": crf_metrics.get("per_key", {}),
                },
                output_dir=folder_result_dir,
            )

        run_manifest.append(
            {
                "folder_size": size,
                "folder_name": folder_path.name,
                "result_dir": _to_relative_path_str(folder_result_dir),
                "rule_based_quality": rule_row["quality_score"],
                "crf_quality": crf_row["quality_score"],
                "rule_based_tradeoff": rule_row["tradeoff_score"],
                "crf_tradeoff": crf_row["tradeoff_score"],
            }
        )

        print(
            "  Rule-based  -> micro_f1={:.4f}, macro_f1={:.4f}, p95={:.2f}ms".format(
                rule_row["structured_micro_f1"],
                rule_row["structured_macro_f1"],
                rule_row["latency_p95_ms"],
            )
        )
        print(
            "  CRF         -> micro_f1={:.4f}, macro_f1={:.4f}, p95={:.2f}ms".format(
                crf_row["structured_micro_f1"],
                crf_row["structured_macro_f1"],
                crf_row["latency_p95_ms"],
            )
        )

    if not global_rows:
        raise RuntimeError("No benchmark runs were executed. Check folder availability and filters.")

    global_rows.sort(key=lambda item: (int(item["folder_size"]), str(item["model"])))

    global_csv_fields = list(global_rows[0].keys())
    write_csv(results_dir / "global_summary.csv", global_rows, fieldnames=global_csv_fields)

    ranked_rows = sorted(global_rows, key=lambda item: float(item["tradeoff_score"]), reverse=True)
    write_csv(results_dir / "ranking_by_tradeoff.csv", ranked_rows, fieldnames=global_csv_fields)

    _print_ranked_summary(ranked_rows)

    selected_winner = _select_winner_row(args, ranked_rows)
    winner_export: dict | None = None

    if selected_winner is not None:
        winner_key = _row_key(selected_winner)
        winner_parser = trained_parsers.get(winner_key)

        if winner_parser is None:
            print(
                f"\n[WARN] Unable to find trained parser for winner row {winner_key}; skipping export."
            )
        else:
            manifest_path = export_winner_parser(
                winner_parser,
                selected_row=selected_winner,
                results_dir=results_dir,
                full_outputs_dir=full_outputs_dir,
                fixed_test_path=fixed_test_path,
                fixed_test_sha256=fixed_test_sha,
                fixed_test_selector=fixed_test_selector,
                requested_sizes=requested_sizes,
            )

            winner_export = {
                "selected_row": selected_winner,
                "manifest_path": _to_relative_path_str(manifest_path),
            }

            print("\nWinner exported.")
            print(f"Export manifest: {_to_relative_path_str(manifest_path)}")

    benchmark_manifest = {
        "full_outputs_dir": _to_relative_path_str(full_outputs_dir),
        "fixed_test_path": _to_relative_path_str(fixed_test_path),
        "fixed_test_sha256": fixed_test_sha,
        "fixed_test_count": len(fixed_test_examples),
        "fixed_test_selector": fixed_test_selector,
        "fixed_test_selected_size": selected_fixed_size,
        "requested_sizes": requested_sizes,
        "executed_runs": run_manifest,
        "global_rows": global_rows,
        "winner_export": winner_export,
    }

    write_json(results_dir / "benchmark_manifest.json", benchmark_manifest)

    if not args.disable_plots:
        plot_global_learning_curves(global_rows, output_dir=results_dir)
        plot_global_latency_vs_quality(global_rows, output_dir=results_dir)
        plot_global_delta(global_rows, output_dir=results_dir)

    print("\nBenchmark complete.")
    print(f"Results written to: {_to_relative_path_str(results_dir)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark rule-based and CRF query parsers across dataset sizes.",
    )
    parser.add_argument(
        "--full-outputs-dir",
        type=str,
        default=str(DEFAULT_FULL_OUTPUTS_DIR),
        help="Directory containing dataset-size folders with train/val/test/stat files.",
    )
    parser.add_argument(
        "--sizes",
        "--size",
        nargs="+",
        default=["2000,4000,6000,8000,10000"],
        help="Dataset sizes to benchmark. Supports comma-separated or space-separated values.",
    )
    parser.add_argument(
        "--fixed-test-size",
        type=int,
        default=None,
        help="Optional folder size used to source fixed test set. If omitted, the largest requested size is used.",
    )
    parser.add_argument(
        "--fixed-test-file",
        type=str,
        default="",
        help="Optional path to explicit fixed test JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where benchmark outputs are written.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip dataset sizes that do not exist instead of failing.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Use only the first N fixed-test samples (0 means all).",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Disable PNG chart generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CRF training.",
    )
    parser.add_argument(
        "--crf-max-iterations",
        type=int,
        default=120,
        help="Maximum number of L-BFGS iterations per CRF fit.",
    )
    parser.add_argument(
        "--winner-rank",
        type=int,
        default=None,
        help="Optional rank to export directly from tradeoff ranking (1=best).",
    )
    parser.add_argument(
        "--no-export-winner",
        action="store_true",
        help="Disable winner model export after the benchmark.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
