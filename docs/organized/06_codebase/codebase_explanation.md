# Viralytics / FashionSense — Codebase Explanation

## Quick Start

```powershell
.\scripts\start_full_app.ps1
```

---

## 1. What This Repository Does

This is a small research system combining:

1. A computer-vision pipeline for clothing detection
2. A lightweight recommendation engine for complementary garments
3. A browser-facing application layer (FastAPI + custom frontend)
4. A natural-language clothing search subsystem (`LNIAGIA/`)
5. Supporting experimentation code for dataset preparation, training, evaluation, and comparison

The repository mixes two AI paradigms:
- **Perception** — detect what a user is wearing from an image or camera stream
- **Retrieval / reasoning** — search or recommend products based on semantic or symbolic rules

---

## 2. Architectural Decomposition

The repository is best understood as five layers.

### 2.1 Runtime Application Layer

The deployable prototype:

- `src/api/main.py`
- `src/api/schemas.py`
- `src/detection/*.py`
- `src/recommendations/*.py`
- `frontend/index.html`

Responsibilities: serving the frontend, exposing HTTP and WebSocket endpoints, loading a
detector once at startup, handling camera/image inference, converting detections into
product recommendations.

### 2.2 Vision Model and Data Pipeline Layer

Supports creating and evaluating the detector:

- `src/detection/converter.py`
- `scripts/sample_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/evaluate_yolo_world.py`
- `scripts/analyze_raw_dataset.py`

Classic ML pipeline: `raw dataset → sampled subset → converted annotations → training → evaluation`.

### 2.3 Custom Research Model Layer

The from-scratch detector implementation:

- `src/custom_model/dataset.py`
- `src/custom_model/model.py`
- `src/custom_model/loss.py`
- `scripts/train_custom.py`
- `scripts/compare_models.py`

This layer exists because the project investigates a custom detector architecture and loss design,
not just consuming YOLOv8 as a black box.

### 2.4 Search Subsystem (`LNIAGIA/`)

A second mini-project inside the repository:

- `LNIAGIA/search_app.py`
- `LNIAGIA/llm_query_parser.py`
- `LNIAGIA/DB/models.py`
- `LNIAGIA/DB/SQLLite/DBManager.py`
- `LNIAGIA/DB/vector/*.py`

Goal: semantic clothing search from natural language, using an LLM for parsing, sentence
embeddings for retrieval, Qdrant for vector search, and metadata constraints for filtering.

### 2.5 Documentation and Experiment Artifacts

- `docs/*.md`, `docs/*.json`, `docs/*.png`
- `notebooks/*.ipynb`
- `runs/`, generated DB files, output JSON in `LNIAGIA/tests/output/`

Evidence, reports, and artifacts rather than active application logic.

---

## 3. Main Application Path

Core online path: `frontend → FastAPI → detector → recommendation engine → JSON/WebSocket response`

### 3.1 `src/api/main.py`

Orchestration module. Creates the FastAPI app, enables CORS, mounts static files, resolves
detector weights, creates singletons at startup, exposes endpoints for health, image detection,
live threshold updates, audio transcription, and camera streaming.

Key design choices:

**A) Weight discovery with `_find_weights()`**
Searches for fine-tuned weights in priority order: environment override → large model →
medium → small → nano → base `yolov8n.pt`. Decouples deployment from hard-coded filenames.

**B) Backend abstraction via `DETECTOR_BACKEND`**
Can choose between `FashionDetector` (fine-tuned YOLOv8) and `YOLOWorldDetector` (zero-shot).
This is a simple dependency injection pattern that makes the app inference-backend-agnostic.

**C) Startup-created shared objects**
Stores `detector`, `recommender`, `camera`, `whisper_model` as module-level globals initialized
at startup. Detection models are expensive to load; webcam logic needs shared state; Whisper
may trigger a one-time download.

**D) Whisper loading in a background task**
Speech-to-text is optional. Blocking startup on Whisper would delay API readiness. Background
loading preserves responsiveness for the critical path (image/camera detection).

**E) `/api/detect/image`**
Reads uploaded bytes → OpenCV decode → detect → extract categories → get recommendations →
draw annotations → return detections + base64 frame.

**F) `/api/conf` and `/api/conf/{value}`**
Live confidence-threshold control. Demonstrates the precision/recall tradeoff interactively.

**G) `/api/transcribe`**
Accepts browser audio (WebM/Opus), converts to WAV via `ffmpeg`, transcribes with Faster-Whisper.

**H) `/ws/camera`**
Delegates to `CameraStream.run_session`. Keeps transport-level code in the API layer and
session/state-machine logic in a dedicated class.

### 3.2 `src/api/schemas.py`

Pydantic models for response payloads. Provides typed contract between backend and client,
automatic serialization validation, and self-documenting API structure.

Notable weakness: `DetectionResponse` uses `List[Dict[str, Any]]` instead of a typed
`List[DetectionItem]`, reducing type strictness.

---

## 4. Detection Subsystem

### 4.1 `src/detection/detector.py`

Main abstraction boundary. Contains category definitions, visualization colors,
`Detection` and `DetectionResult` dataclasses, `BaseDetector`, and `FashionDetector`.

**Why the dataclasses matter:** `Detection` and `DetectionResult` decouple the rest of the
codebase from Ultralytics' raw output types. This is an adapter-pattern design — alternative
backends can be swapped in more easily and testing becomes simpler.

**`FashionDetector`** wraps the Ultralytics `YOLO` object: loads weights, stores inference
thresholds, runs prediction, parses boxes into project-native dataclasses.

### 4.2 `src/detection/yolo_world.py`

Zero-shot / open-vocabulary backend. Loads `yolov8s-worldv2.pt`, injects the 13 clothing
categories via `set_classes`, uses a lower confidence threshold (zero-shot detectors have
weaker confidence calibration on task-specific categories).

Note: the code temporarily relaxes SSL verification to allow CLIP-related downloads.
In production this should be replaced with proper certificate configuration.

### 4.3 `src/detection/camera.py`

Real-time session implementation. Implements a state machine:
- `CAPTURING`
- `ANALYSING`
- `RESULTS`

**Multi-frame accumulation:** during capture, confidence per class is averaged across frames
rather than trusting one frame. Reduces sensitivity to temporary misdetections and smooths
out flicker. The brief analysing phase communicates progress and prevents jarring transitions.

### 4.4 `src/detection/converter.py`

Converts DeepFashion2-style annotations into YOLO label format. Reads sampled annotations
from `index.json`, splits train/val, clamps boxes to image boundaries, converts
`[x1, y1, x2, y2]` to normalized YOLO format, writes `dataset.yaml`.

---

## 5. Recommendation Subsystem

### 5.1 `src/recommendations/catalogue.py`

Defines a mock catalogue via a `CatalogueItem` dataclass and a hard-coded product list.
Replaces infrastructure dependence with an in-memory fixture for prototyping.

### 5.2 `src/recommendations/engine.py`

Rule-based recommender. Maps detected categories to complementary categories via `OUTFIT_RULES`,
accumulates rule scores, samples catalogue items within those categories, returns top `k`.

Why rule-based: explainable, computationally trivial, appropriate when there is no interaction
history or user-profile data. This is a symbolic recommender layered on top of a perceptual model.

Note: the module docstring mentions embedding similarity as a strategy, but the current
implementation is purely rule-based.

---

## 6. Custom Detector Research Path

### 6.1 `src/custom_model/dataset.py`

Adapts YOLO-format annotations into a PyTorch `Dataset` and `DataLoader`. Uses Albumentations
(handles bounding box transformation correctly), supports `light`, `medium`, and `heavy`
augmentation modes, uses a custom `collate_fn` because each image has a different number of boxes.

### 6.2 `src/custom_model/model.py`

Implements: basic convolution blocks (`ConvBnRelu`), residual blocks (`ResBlock`), CSP-style
blocks (`CSPBlock`), multi-scale backbone (`FashionBackbone`), FPN-like neck (`FashionNeck`),
anchor-free detection head (`DetectionHead`), `FashionNet`, and `TinyFashionNet`.

Architecture is clearly inspired by YOLO-family designs: downsampling backbone, multi-scale
features (P3, P4, P5), top-down fusion, per-scale prediction heads. This is intentional —
adapting successful detector ideas rather than reinventing from zero.

**`TinyFashionNet`** exists for pipeline verification rather than accuracy — a cheap model
to validate code paths quickly on CPU before committing to long GPU runs.

### 6.3 `src/custom_model/loss.py`

Implements: CIoU box loss, focal binary cross-entropy for objectness, BCE class loss, and
target assignment across scales.

- **CIoU**: IoU-only loss gives poor gradients when boxes don't overlap well; CIoU adds
  center-distance and aspect-ratio penalties for smoother optimization
- **Focal BCE for objectness**: dense detectors suffer severe foreground/background imbalance;
  focal loss down-weights easy negatives
- **`build_targets()`**: the `multi_cell` option assigns each GT to neighboring cells when
  near boundaries, increasing positive signal density. This is one of the most impactful
  changes in the pipeline (see `03_fashionnet_experiments/fashionnet_pipeline_fixes.md`)

---

## 7. Utility Code

### 7.1 `src/utils/metrics.py`

Evaluation utilities from first principles: IoU computation, greedy prediction-to-GT matching,
per-class AP (VOC-style 101-point interpolated), confusion matrix generation and plotting,
textual detection report, inference benchmarking.

Implementing metrics from first principles makes the evaluation methodology visible and
reusable for models outside Ultralytics (FashionNet, YOLO-World).

### 7.2 `src/utils/visualizer.py`

Extends visualization beyond bare bounding boxes. Histogram and blended annotation views
help reason about confidence distribution and display quality.

---

## 8. Frontend

### 8.1 `frontend/index.html`

Contains HTML structure, a large embedded style block, and a large embedded script block.
Single-file portability simplifies demo deployment and reduces bundling complexity.

The UI supports two modes: camera scanning and chat/voice interaction.

**Known gap:** the page expects `/api/chat`, but `src/api/main.py` does not define that
route. The chat UI is a stub or unfinished integration point.

### 8.2 `frontend/static/js/app.js`

A second frontend implementation for the scan flow. Appears auxiliary or legacy — `index.html`
already contains a full inline script. The source of truth for the active scan flow is the
inline JS in `index.html`.

### 8.3 `frontend/static/css/style.css`

Supplementary styling: utility classes, responsive rules, badges, toasts, skeletons.
Suggests the frontend evolved over time.

---

## 9. Training and Evaluation Scripts

### 9.1 `scripts/sample_dataset.py`

Stratified dataset sampling from DeepFashion2 using pre-built CSV metadata. CSV-based
indexing avoids repeatedly parsing many raw annotation files. Class-balanced sampling
improves fairness across categories.

### 9.2 `scripts/train.py`

Fine-tunes YOLOv8 through the Ultralytics API. Encodes: model-scale selection, pretrained vs
from-scratch initialization, augmentation settings, optimizer/LR choices, early stopping.
This is the "engineering baseline" against which the custom detector is compared.

### 9.3 `scripts/train_custom.py`

Training loop for FashionNet. Exposes all experiment knobs: configurable loss weights,
augmentation intensity, multi-cell assignment, dropout, optimizer choice, grayscale ablations,
EMA, warmup, and scheduler variants. Not just a train loop — an experiment harness.

### 9.4 `scripts/evaluate.py`

Evaluates fine-tuned YOLOv8 using Ultralytics validation flow. Post hoc evaluation needed
for specific checkpoints or confidence settings.

### 9.5 `scripts/evaluate_yolo_world.py`

Evaluates zero-shot backend using the project's own metrics utilities. Necessary because
YOLO-World is used as a custom-configured detector, not a dataset-trained model.

### 9.6 `scripts/compare_models.py`

Compares custom FashionNet vs YOLOv8 (or custom vs custom). Metrics include per-class mAP,
overall mAP, inference speed, parameter count, weight size. The core comparison tool for
the experimental section.

### 9.7 `scripts/analyze_raw_dataset.py`

Exploratory data analysis on the original dataset. Analyzes class balance, box size, aspect
ratio, occlusion, and co-occurrence. Supports methodological rigor by providing empirical
justification for design decisions.

---

## 10. Tests

### 10.1 `tests/test_detector.py`

Validates basic detector behavior: return type, inference time, output shape, detection list
structure, bounding-box validity, confidence range. Sanity tests run on blank frames and base
weights — validates pipeline integrity more than semantic accuracy.

### 10.2 `tests/test_recommendations.py`

Tests: fallback behavior, `top_k`, required fields, non-duplication, expected dress/outwear
pairing, exclusion logic. Good unit tests because the rule-based recommender is deterministic
enough to verify meaningfully.

---

## 11. LNIAGIA Search Subsystem

### 11.1 `LNIAGIA/DB/models.py`

The ontology of the search system. Defines controlled vocabularies, field groups, realistic
generation constraints, brand/price distributions, and helper functions. Acts as domain schema,
generator configuration, and retrieval vocabulary source simultaneously.

### 11.2 `LNIAGIA/DB/SQLLite/DBManager.py`

Manages a simple SQLite database for item records. SQLite provides structured relational
storage, while Qdrant provides semantic retrieval — a common hybrid pattern.

### 11.3 `LNIAGIA/DB/vector/nl_mappings.py`

Maps compact symbolic values to richer natural-language descriptions and synonyms. Embedding
models work better with semantically rich text — `short_sleeve_top` becomes something like
"short sleeve top (t-shirt, tee)".

### 11.4 `LNIAGIA/DB/vector/description_generator.py`

Transforms structured catalog items into descriptive text for embedding. Bridges structured
data and semantic search through synthetic natural-language enrichment.

### 11.5 `LNIAGIA/DB/vector/VectorDBManager.py`

The retrieval engine. Handles embedding model loading (`BAAI/bge-base-en-v1.5`), Qdrant
collection management, vector indexing, plain semantic search, and filtered semantic search.

Uses `BGE_QUERY_PREFIX` because BGE models are optimized when queries include an instruction
prefix — a retrieval-quality optimization grounded in model-specific best practice.

**Strict vs non-strict search:** strict mode converts constraints into hard metadata filters;
non-strict mode retrieves broadly and penalizes mismatches. Real users often want negotiable
matches, not brittle exact filters.

Note: non-strict include-based soft boosting is currently commented out.

### 11.6 `LNIAGIA/llm_query_parser.py`

Uses Ollama with `qwen2.5:3b-instruct` to translate natural-language queries into structured
filters. LLM handles linguistic variability; controlled vocabulary constrains outputs;
validation cleans up invalid generations. A classic "LLM-to-symbolic-IR bridge."

### 11.7 `LNIAGIA/search_app.py`

CLI frontend for the search subsystem. Checks that the vector DB exists, loads the embedding
model, accepts user queries, invokes the LLM parser, asks whether to accept approximate
matches, runs filtered search, prints results.

---

## 12. Code Quality Observations

### Strengths

- Clear modular decomposition between API, detection, recommendations, custom model, and search
- Detector abstraction is well chosen and supports backend swapping cleanly
- Contains both engineering baselines and research-oriented custom implementations
- Evaluation and comparison utilities are unusually thoughtful for a student project
- Search subsystem shows strong awareness of hybrid symbolic + neural retrieval design

### Known Weaknesses

- `frontend/index.html` expects `/api/chat`, but the backend does not expose that route
- Frontend logic is split between inline JS and `app.js` — ambiguity about the source of truth
- Some Pydantic schemas are weaker than needed (nested models not fully enforced)
- `LNIAGIA/DB/models.py` is overloaded: schema + generator + business rules + search config
- Some conceptual claims (recommendation embeddings, soft include boosting) are broader than
  the currently active implementation

These are normal prototype-stage characteristics, not fatal flaws.

---

## 13. Technology Choices

| Technology | Why Used | Alternative |
|------------|----------|-------------|
| FastAPI | Async networking, schema-driven APIs, low boilerplate | Flask, Django |
| OpenCV + NumPy | Standard for CV prototypes, easy camera capture | PIL, imageio |
| Ultralytics YOLO | Strong baseline, fast iteration, easy model comparison | MMDetection, Detectron2 |
| Qdrant | Local deployment, metadata filtering, no external service needed | FAISS + custom layer |
| sentence-transformers (BGE) | Strong general-purpose retrieval embeddings | OpenAI embeddings |
| Ollama + qwen2.5:3b | Local LLM, no API key, handles linguistic variability | GPT-4, Mistral |

---

## 14. Recommended Reading Order

For a new researcher or evaluator:

1. `README.md`
2. `src/api/main.py`
3. `src/detection/detector.py`
4. `src/detection/camera.py`
5. `src/recommendations/engine.py`
6. `scripts/sample_dataset.py`
7. `scripts/train.py`
8. `src/custom_model/model.py`
9. `src/custom_model/loss.py`
10. `scripts/train_custom.py`
11. `LNIAGIA/SEARCH_OVERVIEW.md`
12. `LNIAGIA/DB/vector/VectorDBManager.py`
13. `LNIAGIA/llm_query_parser.py`

This order moves from deployed behavior → training methodology → secondary retrieval research.
