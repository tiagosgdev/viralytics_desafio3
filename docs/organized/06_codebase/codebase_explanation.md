# Codebase Architecture

## System Overview

This repository implements a fashion object detection and recommendation system, combining:

1. A computer-vision pipeline for clothing detection (both YOLOv8 fine-tuning and a custom detector)
2. A rule-based recommendation engine for complementary garments
3. A web application layer (FastAPI + browser frontend)
4. A natural-language clothing search subsystem (LNIAGIA)
5. Training, evaluation, and dataset preparation tooling

---

## Architectural Layers

### 1. Runtime Application Layer

The deployable prototype that serves the end-user experience.

| Module | Responsibility |
|--------|---------------|
| `src/api/main.py` | FastAPI app: HTTP/WebSocket endpoints, model loading, inference orchestration |
| `src/api/schemas.py` | Pydantic response models |
| `src/detection/detector.py` | Detection abstraction: `BaseDetector`, `FashionDetector` (YOLOv8 wrapper) |
| `src/detection/yolo_world.py` | Zero-shot YOLO-World backend via CLIP embeddings |
| `src/detection/camera.py` | Real-time camera session with multi-frame confidence accumulation |
| `src/recommendations/engine.py` | Rule-based complementary garment recommender |
| `frontend/index.html` | Single-file browser UI (HTML + CSS + JS) |

**Design decisions:**
- The detector is loaded once at startup and shared across requests (expensive to reload).
- `DETECTOR_BACKEND` abstracts over FashionDetector vs YOLOWorldDetector, making the app inference-backend-agnostic.
- Camera sessions average confidence across frames to reduce flicker and temporary misdetections.
- The recommender uses symbolic rules (`OUTFIT_RULES` mapping detected categories to complementary ones) rather than learned embeddings. This is appropriate for a prototype with no interaction history.

### 2. Custom Detector (FashionNet / edna)

A single-shot anchor-free detector built from scratch in pure PyTorch.

| Module | Responsibility |
|--------|---------------|
| `src/custom_model/model.py` | Architecture: ConvBnReLU, ResBlock, CSPBlock, Backbone, FPN Neck, DetectionHead |
| `src/custom_model/loss.py` | CIoU box loss, focal BCE objectness, BCE class loss, multi-scale target assignment |
| `src/custom_model/dataset.py` | YOLO-format dataset adapter with Albumentations augmentation pipeline |
| `src/custom_model/postprocess.py` | Grid decoding, NMS, confidence filtering |

**Architecture:** Input (3x640x640) -> Backbone (4 downsampling stages producing P3/P4/P5 at strides 8/16/32) -> Neck (bidirectional FPN with upsample + concat + fuse) -> Head (per-scale predictions: objectness + class + bbox). Three model scales:

| Scale | Parameters | Channel widths |
|-------|-----------|---------------|
| s | ~11.7M | 64-128-256-512 |
| m | ~25M | 96-192-384-768 |
| l | ~43M | 128-256-512-1024 |

**Why custom vs off-the-shelf YOLO?** The project goal was to understand and build a detection pipeline end-to-end -- loss functions, target assignment, post-processing, evaluation -- rather than consuming Ultralytics as a black box. YOLOv8 serves as the performance ceiling and comparison baseline.

### 3. Data Pipeline

| Module | Responsibility |
|--------|---------------|
| `scripts/sample_dataset.py` | Stratified sampling from DeepFashion2 using pre-built CSV metadata |
| `scripts/analyze_raw_dataset.py` | Exploratory data analysis: class balance, box sizes, occlusion |
| `src/detection/converter.py` | DeepFashion2 annotation format to YOLO label format |

### 4. Training and Evaluation

| Module | Responsibility |
|--------|---------------|
| `scripts/train.py` | YOLOv8 fine-tuning via Ultralytics API |
| `scripts/train_custom.py` | FashionNet training loop with full experiment configuration |
| `scripts/evaluate.py` | YOLOv8 evaluation via Ultralytics validation |
| `scripts/evaluate_custom.py` | FashionNet/edna evaluation with custom metrics |
| `scripts/compare_models.py` | Side-by-side model comparison (metrics, speed, size) |
| `src/utils/metrics.py` | IoU, AP, confusion matrix, matching -- implemented from first principles |

`train_custom.py` exposes all experiment knobs as CLI flags: loss weights, augmentation
intensity, multi-cell assignment, dropout, optimizer choice, learning rate schedule, EMA,
mosaic augmentation, and model scale. This makes it an experiment harness, not just a
training loop.

### 5. LNIAGIA Search Subsystem

A separate subsystem for natural-language clothing search.

| Module | Responsibility |
|--------|---------------|
| `LNIAGIA/search_app.py` | CLI search frontend |
| `LNIAGIA/llm_query_parser.py` | LLM-based query parsing (Ollama + qwen2.5:3b) |
| `LNIAGIA/DB/models.py` | Domain schema, controlled vocabularies, generation constraints |
| `LNIAGIA/DB/SQLLite/DBManager.py` | SQLite for structured item storage |
| `LNIAGIA/DB/vector/VectorDBManager.py` | Qdrant vector search with BGE embeddings |
| `LNIAGIA/DB/vector/description_generator.py` | Structured item to natural-language text |
| `LNIAGIA/DB/vector/nl_mappings.py` | Symbolic values to rich text for better embeddings |

**Design:** Hybrid retrieval combining structured SQL filters with semantic vector search.
The LLM translates natural-language queries ("find me a red summer dress under 50 euros")
into structured filters, which are then applied as Qdrant metadata constraints alongside
embedding-based similarity search.

---

## Technology Choices

| Technology | Rationale |
|------------|-----------|
| PyTorch | Standard framework for custom model research |
| Ultralytics YOLO | Strong baseline with minimal code; easy model comparison |
| FastAPI | Async networking, schema-driven APIs, WebSocket support |
| OpenCV + NumPy | Standard CV pipeline tooling |
| Qdrant | Local vector DB with metadata filtering, no external service needed |
| sentence-transformers (BGE) | Strong general-purpose retrieval embeddings |
| Ollama + qwen2.5:3b | Local LLM for query parsing, no API key required |
| Albumentations | Correct bounding box transformation during augmentation |

---

## Main Application Flow

```
Browser -> FastAPI (/api/detect/image or /ws/camera)
  -> Detector (FashionDetector or YOLOWorldDetector)
    -> Inference + post-processing
    -> Detection results (class, bbox, confidence)
  -> RecommendationEngine
    -> Rule-based complementary category lookup
    -> Catalogue item sampling
  -> JSON/WebSocket response with detections + recommendations
```

The camera path (`/ws/camera`) implements a state machine (CAPTURING -> ANALYSING -> RESULTS)
with multi-frame accumulation during capture for robust detection.
