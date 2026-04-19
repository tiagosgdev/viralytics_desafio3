# Metrics, Scoring, Benchmark Flow, and Export Overview

This document explains:

- what is evaluated
- how metrics are calculated
- how ranking is produced
- how winner artifacts are exported and loaded

Implementation sources:

- `run_benchmark.py`
- `baselines/evaluation_metrics.py`
- `baselines/exporter.py`
- `baselines/data_utils.py`

## 1. End-to-End Benchmark Flow

For each requested dataset size folder:

1. Load `train.json` and `val.json` from that folder.
2. Use one fixed test set for all folders (for fair comparison), selected by:
   - explicit `--fixed-test-file`, or
   - `--fixed-test-size`, or
   - largest requested size.
3. Train/finalize each baseline parser on folder train+val.
4. Parse all fixed test queries.
5. Project parser output back to BIO labels (`project_bio_labels`) for token-level metrics.
6. Compute structured metrics + token metrics + latency metrics.
7. Write per-folder outputs and append rows to global summary.

After all folders:

- Write `global_summary.csv`.
- Rank by `tradeoff_score` into `ranking_by_tradeoff.csv`.
- Optionally export winner parser artifact.

## 2. Structured Evaluation Unit

A prediction is flattened into structured pairs:

- `(include, key, value)`
- `(exclude, key, value)`

Metrics are based on set overlap between predicted pairs and gold pairs.

## 3. Core Structured Metrics

Let:

- $TP$ = true positive structured pairs
- $FP$ = false positive structured pairs
- $FN$ = false negative structured pairs

Then:

$$
\text{precision} = \frac{TP}{TP + FP}
$$

$$
\text{recall} = \frac{TP}{TP + FN}
$$

$$
F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

Reported structured metrics:

- `structured_micro_precision`
- `structured_micro_recall`
- `structured_micro_f1`
- `structured_macro_f1`

## 3.1 Structured Macro F1 (Per Key)

For each key (for example `color`, `type`, `fit`), key-level precision/recall/F1 are computed first, then averaged:

$$
\mathrm{structured\_macro\_f1} = \frac{1}{K} \sum_{k=1}^{K} F1_k
$$

Purpose:

- Prevent large/frequent keys from dominating the score.
- Reflect balanced behavior across attribute categories.

## 3.2 Micro vs Macro (quick intuition)

- **Micro** computes one global confusion count across all keys, then computes precision/recall/F1 from those totals. It gives more weight to frequent keys.
- **Macro** computes a score per key first, then averages keys equally. It gives the same weight to rare and frequent keys.

Micro formulas:

$$
\mathrm{micro\_precision} = \frac{\sum_{k=1}^{K} TP_k}{\sum_{k=1}^{K}(TP_k + FP_k)}
$$

$$
\mathrm{micro\_recall} = \frac{\sum_{k=1}^{K} TP_k}{\sum_{k=1}^{K}(TP_k + FN_k)}
$$

Why we keep both in this project:

- `structured_micro_f1` reflects overall end-to-end correctness on the real key distribution.
- `structured_macro_f1` ensures weak performance on rare keys is not hidden by common keys.

## 4. Strict Exact Match Rate

A query is an exact match only if the entire structured output is identical:

- same include keys/values
- same exclude keys/values
- same polarity placement

$$
\mathrm{strict\_exact\_match\_rate} = \frac{N_{\mathrm{exact\_queries}}}{N_{\mathrm{queries}}}
$$

Purpose:

- Measures true end-user correctness for full parse output.

## 5. Negation Accuracy

Negation accuracy is computed on overlapping plain `(key, value)` pairs between gold and prediction, then checks whether polarity (`include` vs `exclude`) matches.

$$
\mathrm{negation\_accuracy} = \frac{N_{\mathrm{correct\_polarity\_on\_overlap}}}{N_{\mathrm{overlap\_pairs}}}
$$

Purpose:

- Isolate include/exclude correctness once attribute detection is already aligned.

## 6. Token-Level BIO Metrics

Token metrics are computed from gold BIO labels vs projected predicted BIO labels:

- `token_accuracy`
- `token_micro_f1`
- `token_macro_f1`
- per-label precision/recall/F1/support

Purpose:

- Diagnose sequence-labeling quality and label-level failure modes.

## 7. Latency and Throughput Metrics

From per-query inference times (ms):

- `mean_ms`
- `median_ms`
- `p95_ms`
- `total_ms`
- `throughput_qps`

`p95_ms` is the 95th-percentile latency: 95% of queries finish at or below this value, and 5% are slower.

where:

$$
\mathrm{throughput\_qps} = \frac{N \cdot 1000}{\mathrm{total\_ms}}
$$

Purpose:

- Capture both typical latency and tail latency (p95).
- Support quality-speed tradeoff decisions.
- Use p95 specifically to track tail risk; user experience is usually impacted by slow outlier queries, not only by the mean.

## 8. Composite Scores Used for Ranking

## 8.1 Quality Score

From `compute_quality_score`:

$$
\mathrm{quality\_score} = 0.45 \cdot \mathrm{micro\_f1}
+ 0.25 \cdot \mathrm{macro\_f1}
+ 0.20 \cdot \mathrm{negation\_accuracy}
+ 0.10 \cdot \mathrm{exact\_match}
$$

Purpose:

- Emphasize structured correctness while still rewarding polarity quality and strict full-match behavior.

## 8.2 Tradeoff Score

From `compute_tradeoff_score`:

$$
\mathrm{tradeoff\_score} = \frac{\mathrm{quality\_score}}{1 + \frac{\mathrm{p95\_ms}}{100}}
$$

Purpose:

- Penalize high tail latency and prioritize deployable models.
- This is the primary ranking metric used for winner selection.

## 8.3 Why These Metrics and Formulas

Why this quality formula:

- `micro_f1` (0.45): primary signal of global structured correctness.
- `macro_f1` (0.25): protects against ignoring low-frequency keys.
- `negation_accuracy` (0.20): include/exclude polarity errors are critical in retrieval behavior.
- `exact_match` (0.10): rewards full-query correctness without over-dominating the score.

Why this tradeoff formula:

- The denominator `1 + p95_ms/100` adds a smooth, monotonic latency penalty.
- Using p95 (not only mean) penalizes unstable tail behavior that users feel most.
- Dividing by 100 normalizes milliseconds into a moderate penalty scale, so quality remains primary while latency still matters for ranking.

## 9. Export Overview

Winner export is handled by `export_winner_parser(...)`.

## 9.1 Selection Context

Export stores:

- selected row from global ranking
- fixed test path and SHA256
- selected sizes and full outputs path
- selection metric context (`tradeoff_score`)

## 9.2 Artifact Layout

Under `results/exported_parsers/`:

- `latest.json` pointer
- `<timestamp>_<model>_size<size>/manifest.json`
- `<timestamp>_<model>_size<size>/parser_state.json`
- `<timestamp>_<model>_size<size>/crf_model.pkl` (CRF only)

## 9.3 Why Split State and Model for CRF

- `parser_state.json` stores parser metadata and lookup tables in JSON.
- `crf_model.pkl` stores the trained CRF object.
- This keeps metadata inspectable and model loading explicit.

## 10. Loading Exported Winner in Runtime

`load_latest_exported_parser(results_dir)`:

1. Reads `latest.json`.
2. Resolves manifest path.
3. Loads state + optional CRF model object.
4. Reconstructs parser instance via `from_export_state`.

This enables production runtime to use the benchmark winner directly.

## 11. Common Output Files and Purpose

Per folder:

- `summary.json`: condensed folder-level report
- `model_metrics.csv`: one row per model with comparable metrics
- `*_per_query.json`: per-query debug records
- plots: quality bars, latency boxplot, per-key heatmap

Global:

- `global_summary.csv`: all rows across sizes/models
- `ranking_by_tradeoff.csv`: leaderboard
- `benchmark_manifest.json`: full run traceability
- global plots: learning curves, quality-vs-latency, CRF-minus-rule deltas

## 12. Notes

- The benchmark enforces fixed-test evaluation to reduce cross-folder comparison bias.
- Leakage previews are reported (`build_leakage_report`) to detect query/id overlap risks.
- `DEFAULT_FULL_OUTPUTS_DIR` and `DEFAULT_RESULTS_DIR` in `run_benchmark.py` currently use legacy `query_parsing_models` path strings. If needed, pass explicit CLI paths to match `query_parsing/qp_models`.