# Codebase Inventory — Dead Code Audit

**Branch:** code-review  
**Goal:** Identify what's active, what's dead (especially YOLO), and what needs review before deletion.

---

## 1. Core Model Code (`src/`)

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `src/custom_model/model.py` | FashionNet CNN architecture (s/m/l scales) | ACTIVE | NO | Core edna model |
| `src/custom_model/dataset.py` | PyTorch Dataset — YOLO-format labels for FashionNet | ACTIVE | NO | Used by `train_custom.py` |
| `src/custom_model/loss.py` | CIoU + focal objectness + class BCE loss | ACTIVE | NO | Used by `train_custom.py` |
| `src/custom_model/postprocess.py` | NMS + grid decoding for FashionNet | ACTIVE | NO | Used by `fashionnet_detector.py` |
| `src/detection/fashionnet_detector.py` | Runtime FashionNet inference wrapper (edna) | ACTIVE | NO | Imported by `main.py` |
| `src/detection/detector.py` | BaseDetector ABC + FashionDetector (YOLO wrapper) + shared dataclasses | MIXED | NO | BaseDetector/dataclasses used everywhere. `FashionDetector` is cruella backend — needs cleanup not deletion. |
| `src/detection/yolo_world.py` | YOLOWorldDetector — zero-shot YOLO-World wrapper | DEAD | **YES** | YOLO — abandoned. Remove import in `main.py` too. |
| `src/detection/converter.py` | Converts DeepFashion2 → YOLO label format | DEAD | NEEDS REVIEW | Dataset already built. Safe if not rebuilding. |
| `src/detection/camera.py` | WebSocket camera stream | ACTIVE | NO | Used by `main.py` |
| `src/api/main.py` | FastAPI app — all routes | ACTIVE | NO | Entry point. Has YOLO imports to clean up. |
| `src/api/schemas.py` | Pydantic request/response models | ACTIVE | NO | |
| `src/api/personas.py` | Persona config (cruella=yolo, edna=fashionnet) | ACTIVE | NO | |
| `src/api/search_service.py` | Session management, query parsing, vector search | ACTIVE | NO | |
| `src/api/custom_text_parser.py` | Rule-based NL query parser for edna | ACTIVE | NO | |
| `src/recommendations/engine.py` | Rule-based recommendation engine | ACTIVE | NO | |
| `src/recommendations/catalogue.py` | JSON-backed store catalogue loader | ACTIVE | NO | |
| `src/utils/metrics.py` | Per-class AP, confusion matrix, IoU, match_predictions | ACTIVE | NO | |
| `src/utils/visualizer.py` | Draws confidence histograms | DEAD | **YES** | Not imported anywhere. 68 lines. |

---

## 2. Training Scripts (`scripts/training/`)

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `scripts/training/train_custom.py` | FashionNet training loop — active script | ACTIVE | NO | Primary training entry point |
| `scripts/training/train.py` | YOLOv8 fine-tuning script | DEAD | **YES** | YOLO — DELETE |

---

## 3. Evaluation Scripts (`scripts/evaluation/`)

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `scripts/evaluation/evaluate_custom.py` | FashionNet evaluation (mAP, F1, confusion matrix) | ACTIVE | NO | Primary eval script |
| `scripts/evaluation/visualize_results.py` | Plots from training/eval outputs | ACTIVE | NO | Produces `results/plots/` content |
| `scripts/evaluation/compare_models.py` | FashionNet vs YOLOv8 side-by-side comparison | UNCLEAR | NEEDS REVIEW | Thesis artifact. Keep for documentation. Decoder bug (see code_review.md H8). |
| `scripts/evaluation/evaluate.py` | YOLOv8 evaluation | DEAD | **YES** | YOLO — DELETE |
| `scripts/evaluation/evaluate_yolo_world.py` | YOLO-World zero-shot evaluation | DEAD | **YES** | YOLO — DELETE |

---

## 4. Data Scripts

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `scripts/data_prep/sample_balanced.py` | Balanced dataset sampler (84K images, 13 classes) | ACTIVE | NO | How the active training dataset was created |
| `scripts/data_prep/sample_dataset.py` | Older stratified sampler (no splits) | DEAD | **YES** | Superseded by `sample_balanced.py` |
| `scripts/data_prep/analyze_raw_dataset.py` | EDA figures for raw DeepFashion2 | ACTIVE | NO | Produces thesis figures |
| `scripts/app/start_full_app.py` | App launcher with dependency checks | ACTIVE | NO | Main way to start the app |
| `scripts/app/start_full_app.ps1` | PowerShell wrapper for Windows | ACTIVE | NO | |
| `scripts/README.md` | Index of all scripts | ACTIVE | NO | |

---

## 5. Notebooks

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `notebooks/01_EDA.ipynb` | EDA of sampled dataset | ACTIVE | NO | Thesis artifact |
| `notebooks/02_model_comparison.ipynb` | FashionNet vs YOLOv8 visual comparison | UNCLEAR | NEEDS REVIEW | YOLO comparison data — thesis history |

---

## 6. Docs

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `docs/organized/02_yolo_experiments/` | YOLO experiment results | DEAD | NEEDS REVIEW | Thesis history — keep or archive |
| `docs/organized/03_fashionnet_experiments/` | FashionNet pipeline fixes + results | ACTIVE | NO | |
| `docs/organized/04_edna/` | Edna plans, results, review | ACTIVE | NO | |
| All other `docs/` | Mix of analysis, experiment logs, figures | ACTIVE | NO | |

---

## 7. App Code

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `frontend/index.html` | Web UI (persona picker, camera, chat) | ACTIVE | NO | Served by `main.py` |
| `frontend/static/css/style.css` | Frontend styles | ACTIVE | NO | |
| `android_app/` | Native Android client (1 Kotlin file + Gradle) | UNCLEAR | NEEDS REVIEW | If Android dev not active, delete whole dir |
| `android_app/local.properties` | Local Windows SDK path | DEAD | **YES** | Machine-specific, should not be in git |
| `LNIAGIA/` | Semantic search: LLM parser, Qdrant, SQLite, NLP data gen | ACTIVE | NO | Powers cruella LLM text search via `search_service.py`. If cruella dropped → entire dir dead. |

---

## 8. Root-Level Files

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `README.md` | Project overview | ACTIVE | NO | Still references cruella/YOLO as active — needs update |
| `requirements.txt` | Python deps | ACTIVE | NO | Has `ultralytics` — can drop if YOLO fully removed |
| `Dockerfile` | Docker image | ACTIVE | NO | |
| `docker-compose.yml` | Docker Compose | ACTIVE | NO | `MODEL_WEIGHTS` points to YOLO weights — needs update |
| `tests.md` | 13K-line training results (YOLO + FashionNet mixed) | UNCLEAR | NEEDS REVIEW | Move to `docs/`? |
| `tests/test_detector.py` | Unit tests for YOLO-based FashionDetector | DEAD | NEEDS REVIEW | Replace with FashionNet tests |
| `tests/test_recommendations.py` | Unit tests for RecommendationEngine | ACTIVE | NO | |

---

## 9. Model Weights / Results

| Path | Purpose | Status | Delete? | Reason |
|------|---------|--------|---------|--------|
| `models/weights/` | Empty placeholder (*.pt gitignored) | ACTIVE | NO | |
| `results/plots/edna_1.2m/` | Plots for edna 1.2M experiment | ACTIVE | NO | Thesis figures |
| `results/plots/edna_1.3m/` | Plots for edna 1.3M experiment | ACTIVE | NO | Thesis figures |
| `results/test/edna_1.2m_test-set/` | Test-set metrics JSON for edna 1.2M | ACTIVE | NO | |
| `results/test/edna_1.3m_test-set/` | Test-set metrics JSON for edna 1.3M | ACTIVE | NO | |

---

## Clear Deletions (YOLO / confirmed dead)

| File | Action |
|------|--------|
| `src/detection/yolo_world.py` | DELETE + remove import in `main.py` |
| `src/utils/visualizer.py` | DELETE — not imported anywhere |
| `scripts/training/train.py` | DELETE — YOLO |
| `scripts/evaluation/evaluate.py` | DELETE — YOLO |
| `scripts/evaluation/evaluate_yolo_world.py` | DELETE — YOLO |
| `scripts/data_prep/sample_dataset.py` | DELETE — superseded |
| `android_app/local.properties` | DELETE — machine-specific |

---

## Needs Your Decision

| Item | Question |
|------|----------|
| `src/detection/converter.py` | Dataset already built — will you ever rebuild from raw? Keep or delete. |
| `src/detection/detector.py` | Contains YOLO `FashionDetector` + shared base classes. Strip YOLO class, keep base? |
| `LNIAGIA/` | Is cruella persona still needed? If no → entire dir dead. |
| `android_app/` | Is Android development continuing? If no → delete entire dir. |
| `scripts/evaluation/compare_models.py` | Keep as thesis artifact or delete? |
| `notebooks/02_model_comparison.ipynb` | Same — keep as thesis history or delete? |
| `docs/organized/02_yolo_experiments/` | Archive in docs or delete? |
| `tests/test_detector.py` | Replace with FashionNet tests or delete? |
| `tests.md` | Move to `docs/` or delete? |

---

## Files Needing Cleanup (not deletion)

| File | What to clean |
|------|--------------|
| `src/api/main.py` | Remove YOLO-World import (line 45), YOLO weight discovery (lines 82-100), `DETECTOR_BACKEND=yolo_world` branch (lines 160-162) |
| `docker-compose.yml` | Update `MODEL_WEIGHTS` env var from YOLO → edna weights path |
| `README.md` | Remove cruella/YOLO as active persona if dropping |
| `requirements.txt` | Drop `ultralytics` if YOLO fully removed |
