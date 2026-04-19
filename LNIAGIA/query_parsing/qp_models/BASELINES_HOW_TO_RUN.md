# Query Parsing Baselines: How To Run

This guide runs the two baseline models for structured query parsing:

- Rule-based spaCy parser
- CRF sequence-labeling parser

It benchmarks both models across dataset-size folders and produces per-folder and global comparison artifacts.

## Implementation docs

For architecture and metric details, see:

- `PARSER_CONSTRUCTION.md`
- `METRICS_EXPORT_OVERVIEW.md`

## 1. Install dependencies

From the repository root:

python -m pip install -r LNIAGIA/requirements.txt

If needed, install the spaCy English model:

python -m spacy download en_core_web_sm

## 2. (Optional) Generate size folders in one command

If 2000/4000/6000/8000/10000 folders are missing, generate them with:

python LNIAGIA/query_parsing/qp_models/data_generation/generate_size_sweep.py --sizes 2000,4000,6000,8000,10000

Notes:
- This uses the same full dataset generator used by your current pipeline.
- It creates folders under LNIAGIA/query_parsing/qp_models/data_generation/full_outputs.

## 3. Run the baseline benchmark

python -m LNIAGIA.query_parsing.qp_models.run_benchmark --sizes 2000,4000,6000,8000,10000

Useful options:
- --full-outputs-dir path/to/full_outputs
- --fixed-test-size 10000  (optional override)
- --fixed-test-file path/to/custom_test.json
- --max-test-samples 500
- --crf-max-iterations 120
- --disable-plots
- --results-dir LNIAGIA/query_parsing/qp_models/results
- --winner-rank 1  (auto-export ranking position without prompt)
- --no-export-winner  (skip winner export)

At the end of each run, the CLI prints a ranked table (sorted by tradeoff_score) and asks which rank to export.
If you just press Enter, rank 1 is exported.

Faster debug run example:

python -m LNIAGIA.query_parsing.qp_models.run_benchmark --sizes 2000,4000 --max-test-samples 300 --crf-max-iterations 80

## 4. Output structure

Results are written to:

LNIAGIA/query_parsing/qp_models/results

Main outputs:
- global_summary.csv
- ranking_by_tradeoff.csv
- benchmark_manifest.json
- learning_curves.png
- quality_vs_latency.png
- crf_minus_rule_delta.png
- exported_parsers/latest.json
- exported_parsers/<timestamp_model_size>/manifest.json
- exported_parsers/<timestamp_model_size>/parser_state.json
- exported_parsers/<timestamp_model_size>/crf_model.pkl (CRF only)

Per-folder outputs:
- summary.json
- model_metrics.csv
- rule_based_per_query.json
- crf_per_query.json
- quality_bars.png
- latency_boxplot.png
- per_key_f1_heatmap.png

## 5. Metrics included

Quality:
- Structured micro precision/recall/F1
- Structured macro F1 (per-key average)
- Negation accuracy
- Strict exact-match rate
- Token BIO metrics (accuracy, micro F1, macro F1)

Efficiency:
- Fit time (seconds)
- Inference latency (mean/median/p95 ms)
- Throughput (queries per second)

Ranking:
- quality_score
- tradeoff_score (quality adjusted by latency penalty)

## 6. Fair-comparison policy implemented

- One fixed common test set is used for all folders.
- Train and validation always come from the folder currently being benchmarked.
- A leakage report is saved per folder (query/id overlap against fixed test set).

## 7. Search app integration with exported winner

The interactive search app uses a hybrid parser flow:
- Initial query parsing: exported winner baseline (if available)
- Follow-up refinements: LLM (refine_query)

Parser mode is controlled directly in `LNIAGIA/search_app.py` via `PARSER_MODE`.
