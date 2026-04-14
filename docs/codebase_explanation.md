# Viralytics / FashionSense Codebase Explanation

## Quick start

- `.\scripts\start_full_app.ps1`


## 1. What this repository is trying to do

This repository is not a single-purpose script. It is a small research system that combines:

1. A computer-vision pipeline for clothing detection.
2. A lightweight recommendation engine that suggests complementary garments.
3. A browser-facing application layer built with FastAPI and a custom frontend.
4. A persona-selection layer that switches between two model stacks: `Cruella` and `Edna`.
5. A natural-language clothing search subsystem in `LNIAGIA/`.
6. Supporting experimentation code for dataset preparation, training, evaluation, and model comparison.

At a master's-project level, the repository is interesting because it mixes two distinct AI paradigms:

- `Perception`: detect what a user is wearing from an image or camera stream.
- `Retrieval / reasoning over fashion metadata`: search or recommend products based on semantic or symbolic rules.

That makes the codebase a hybrid applied-AI system rather than just a model-training repo.

## 2. Architectural decomposition

The repository is best understood as five layers.

### 2.1 Runtime application layer

This is the deployable prototype:

- `src/api/main.py`
- `src/api/schemas.py`
- `src/detection/*.py`
- `src/recommendations/*.py`
- `frontend/index.html`

This layer is responsible for:

- serving the frontend,
- exposing HTTP and WebSocket endpoints,
- loading multiple detector backends at startup,
- routing requests by selected persona,
- handling camera/image inference,
- converting detections into product recommendations,
- passing persona-aware state into chat refinement.

### 2.2 Vision model and data pipeline layer

This layer supports creating and evaluating the detector:

- `src/detection/converter.py`
- `scripts/sample_dataset.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/evaluate_yolo_world.py`
- `scripts/analyze_raw_dataset.py`

This is a classic ML pipeline:

`raw dataset -> sampled subset -> converted annotations -> training -> evaluation`.

### 2.3 Custom research model layer

This is the from-scratch detector implementation:

- `src/custom_model/dataset.py`
- `src/custom_model/model.py`
- `src/custom_model/loss.py`
- `scripts/train_custom.py`
- `scripts/compare_models.py`

This part exists because the project is not only consuming YOLOv8 as a black box; it also investigates a custom detector architecture and loss design.

### 2.4 Search subsystem (`LNIAGIA`)

This is effectively a second mini-project inside the repository:

- `LNIAGIA/search_app.py`
- `LNIAGIA/llm_query_parser.py`
- `LNIAGIA/DB/models.py`
- `LNIAGIA/DB/SQLLite/DBManager.py`
- `LNIAGIA/DB/vector/*.py`

Its goal is not visual detection. Its goal is semantic clothing search from natural language, using:

- an LLM for parsing,
- sentence embeddings for retrieval,
- Qdrant for vector search,
- metadata constraints for filtering.

### 2.5 Documentation and experiment artifacts

These are supporting outputs rather than core runtime code:

- `docs/*.md`
- `docs/*.json`
- `docs/*.png`
- `notebooks/*.ipynb`
- `runs/`
- generated DB files and output JSON files inside `LNIAGIA/tests/output/`

These files are still important academically, but they are mostly evidence, reports, and artifacts, not active application logic.

## 3. Main application path: how the deployed prototype works

The core online path is:

`frontend -> persona selection -> FastAPI -> persona-specific detector/parser -> recommendation/search response`.

### 3.1 `src/api/main.py`

This is the orchestration module. Its main role is not algorithmic novelty but system composition.

What it does:

- creates the FastAPI app,
- enables permissive CORS,
- mounts static files,
- resolves detector weights,
- creates singleton instances at startup,
- loads both the standard detector and the custom FashionNet detector,
- maintains persona-aware detector and camera registries,
- exposes endpoints for health, image detection, live threshold updates, audio transcription, and camera streaming.

Why this structure is used:

- FastAPI is a strong fit for ML prototypes because it gives typed endpoints, automatic docs, and async support with low ceremony.
- startup-time singleton loading avoids reloading the model per request, which would make inference unusably slow.
- WebSockets are appropriate for continuous camera interaction because the server can push frames/results incrementally.

Important design choices:

#### A) Weight discovery with `_find_weights()`

The file searches for fine-tuned weights in priority order:

- environment override,
- large model,
- medium,
- small,
- nano,
- and finally the base `yolov8n.pt`.

Why this is useful:

- it decouples deployment from hard-coded filenames,
- it allows the same app code to run across multiple experiment outcomes,
- it ensures the app still works even if no fine-tuned model is available.

Alternative:

- require a mandatory explicit config file or CLI flag.

Tradeoff:

- explicit config is cleaner and less implicit,
- but the current approach is friendlier for demos and rapid iteration.

#### B) Backend abstraction and persona routing

The current runtime architecture now has two overlapping ideas:

1. a generic backend abstraction for detector classes
2. a higher-level persona selection layer

At startup, the code prepares:

- a `Cruella` detector path, normally the trained YOLO detector
- an `Edna` detector path, normally the custom `FashionNetDetector`

This is stronger than the earlier single-detector startup design because the frontend can select a model family at runtime without restarting the app.

There is still a lower-level `DETECTOR_BACKEND` switch for the standard path, especially for YOLO-World experimentation, but the user-facing architectural concept is now persona-based rather than just "one detector backend."

#### C) Startup-created shared objects

The code stores:

- a default detector
- a persona-to-detector map
- a persona-to-camera map
- `recommender`
- `whisper_model`
- `search_service`

as module-level globals initialized during startup.

Why this is being used:

- detection models are large and expensive to load,
- webcam session logic needs shared state,
- Whisper may be slow to initialize and may trigger a one-time download.

Alternative:

- use `app.state` or a dependency-injection layer for all runtime services.

That would be more idiomatic FastAPI, but the current global registry pattern is still reasonable for a prototype and keeps the model-switching logic easy to follow.

#### D) Whisper loading in a background task

This is a notable system-design decision. The model is loaded asynchronously through `run_in_executor`.

Why:

- speech-to-text is optional functionality,
- blocking startup on Whisper would delay API readiness,
- a background load preserves responsiveness.

This is good engineering for an interactive demo: the critical path is image/camera detection, not voice chat.

#### E) `/api/detect/image` and `/api/mobile/scan`

These endpoints:

1. reads uploaded bytes,
2. decodes them with OpenCV,
3. resolves the selected persona,
4. runs the corresponding detector,
4. extracts unique categories,
5. gets recommendations,
6. draws annotations,
7. creates a persona-aware session,
8. returns structured detections plus a base64 frame.

Why OpenCV and base64:

- OpenCV is a natural choice because the detector already works on NumPy/OpenCV frames,
- base64 avoids dealing with separate binary image endpoints for the annotated preview.

Alternative:

- return raw boxes only and let the frontend draw overlays.

That would reduce bandwidth and make the frontend more flexible, but server-side annotation is simpler and guarantees consistent visualization.

#### F) `/api/conf` and `/api/conf/{value}`

These routes expose live confidence-threshold control.

Why this is pedagogically valuable:

- it surfaces an important detector hyperparameter to the user,
- it demonstrates the precision/recall tradeoff interactively,
- it helps explain why operating thresholds matter in applied ML systems.

#### G) `/api/transcribe`

This route accepts browser audio, stores it temporarily, converts it to WAV through `ffmpeg`, and transcribes it using Faster-Whisper.

Why it is implemented this way:

- browsers often record audio in WebM/Opus,
- Whisper-based libraries usually prefer PCM WAV or equivalent decoded audio.

The temporary-file approach is pragmatic and easy to debug, though not the most efficient.

Alternative:

- stream audio in-memory through `ffmpeg-python` or PyAV.

That would be more elegant, but significantly more complex.

#### H) `/ws/camera`

This endpoint now resolves the selected persona from the WebSocket query string and delegates the whole UX loop to the corresponding `CameraStream.run_session`.

That is a meaningful architectural improvement because it keeps the transport route stable while allowing different vision models to power the live scan flow.

### 3.2 `src/api/schemas.py`

This file defines Pydantic models for responses.

What it contributes:

- a typed contract between backend and client,
- automatic serialization validation,
- self-documenting API structure,
- explicit persona propagation through session, scan, and chat payloads.

Why Pydantic is being used:

- in FastAPI, Pydantic is the standard way to formalize API payloads.

One subtle weakness remains:

- `DetectionResponse` still uses `List[Dict[str, Any]]` instead of strict nested response models for detections and recommendations.

That reduces type strictness even though the persona-related contract is now clearer.

Alternative:

- use nested model classes directly.

That would be cleaner, safer, and better for generated docs.

## 4. Detection subsystem

### 4.1 `src/detection/detector.py`

This is the main abstraction boundary for detection.

It contains:

- category definitions,
- visualization colors,
- `Detection` and `DetectionResult` dataclasses,
- `BaseDetector`,
- `FashionDetector`.

#### Why the dataclasses matter

`Detection` and `DetectionResult` make the rest of the codebase independent of Ultralytics' raw output types.

This is a strong architectural decision because it creates an internal representation layer.

Benefits:

- the rest of the app is not tightly coupled to the YOLO library,
- alternative backends can be swapped in more easily,
- testing becomes simpler because results can be mocked with plain Python objects.

This is a textbook example of adapter-pattern thinking.

#### `BaseDetector`

This abstract base class defines the contract:

- subclasses must implement `detect(frame)`,
- all detectors inherit the `draw()` helper.

Why this is useful:

- both fine-tuned YOLOv8 and YOLO-World can be used interchangeably,
- code elsewhere can rely on polymorphism.

Alternative:

- skip the abstract class and just rely on duck typing.

That would still work in Python, but the current design is clearer for maintainability and for educational exposition.

#### `FashionDetector`

This class wraps the Ultralytics `YOLO` object.

What it does:

- loads weights,
- stores inference thresholds,
- runs prediction,
- parses Ultralytics boxes into project-native dataclasses.

Why this wrapper exists:

- to hide library-specific details from the rest of the system,
- to provide a stable project API regardless of detector backend,
- to centralize threshold/image-size configuration.

Alternative:

- call `YOLO.predict()` directly from the API route.

That would create tighter coupling and duplicate parsing logic. The wrapper is better.

### 4.2 `src/detection/fashionnet_detector.py`

This is a major new runtime bridge in the codebase.

Before this addition, the custom FashionNet model existed primarily as a research artifact for training and evaluation. The repository now contains an adapter that allows that model family to participate directly in the deployed application.

What it does:

- loads a FashionNet checkpoint,
- resolves configuration from checkpoint metadata when available,
- preprocesses OpenCV frames into the tensor format expected by FashionNet,
- runs the custom model,
- postprocesses raw outputs through the custom decoding/NMS pipeline,
- converts the results back into the same `Detection` / `DetectionResult` abstraction used elsewhere.

Why this file matters architecturally:

- it upgrades FashionNet from "offline experiment" to "runtime backend",
- it preserves the detector interface contract,
- it allows the frontend persona switch to map to a genuinely different vision stack.

This is a strong example of adapter-pattern thinking: a custom research model is made deployable by writing a translation layer rather than rewriting the rest of the app.

### 4.3 `src/detection/yolo_world.py`

This is the zero-shot/open-vocabulary backend.

Conceptually, it answers the research question:

"Can we perform clothing detection without fashion-specific fine-tuning?"

What it changes relative to `FashionDetector`:

- loads `yolov8s-worldv2.pt`,
- injects the project's 13 clothing categories through `set_classes`,
- uses a lower confidence threshold,
- parses results into the same internal dataclasses.

Why the lower threshold is justified:

- zero-shot detectors usually have weaker confidence calibration on task-specific categories than fully fine-tuned models.

Important note:

- the code temporarily relaxes SSL verification to allow the required CLIP-related download when configuring YOLO-World classes.

That is a pragmatic workaround, but from a production-security standpoint it is not ideal.

Alternative:

- pre-download dependencies in a controlled environment,
- or configure certificates correctly instead of bypassing verification.

From a research prototype perspective, the current approach prioritizes reproducibility under messy local environments.

### 4.4 `src/detection/camera.py`

This file implements the real-time session UX.

This is not just "camera code"; it is a state machine:

- `CAPTURING`
- `ANALYSING`
- `RESULTS`

Why this is good design:

- it imposes structure on what could otherwise become ad hoc WebSocket logic,
- it maps technical processing onto a comprehensible user experience.

#### Why accumulate detections over multiple frames

During capture, the code averages confidence per class across frames instead of trusting one frame.

This is an important applied-vision choice.

Benefits:

- reduces sensitivity to temporary misdetections,
- smooths out flicker,
- makes results more robust under small pose or lighting changes.

Alternative:

- use only the last frame,
- use temporal tracking,
- use a voting mechanism,
- use weighted exponential smoothing.

The current averaging approach is simple and sensible for a short scan window.

#### Why there is a separate analysing phase

The pause is only about half a second, but it matters UX-wise.

It:

- communicates progress,
- makes the system feel deliberate rather than abruptly jumping,
- creates time for recommendation generation without a jarring transition.

This is a subtle example of human-centered systems design.

### 4.5 `src/detection/converter.py`

This file converts DeepFashion2-style annotations into YOLO labels.

Why it exists:

- DeepFashion2 annotations are not directly in the Ultralytics training format,
- so the project needs a preprocessing bridge.

What it does:

- reads sampled annotations from `index.json`,
- splits train/val,
- clamps boxes to image boundaries,
- converts `[x1, y1, x2, y2]` into normalized YOLO format,
- writes `dataset.yaml`.

Why clamping and filtering invalid boxes matters:

- raw annotation pipelines often contain edge-case boxes,
- invalid boxes can poison training or crash downstream tools.

Alternative:

- use a library-specific dataset importer,
- or train directly in COCO-like format if the framework supports it.

But for Ultralytics YOLO, explicit YOLO-format conversion is the most straightforward route.

## 5. Recommendation subsystem

### 5.1 `src/recommendations/catalogue.py`

This layer is no longer best understood as a hard-coded Python fixture.

The recommendation catalogue is now designed around an editable external data source:

- `data/mock_store_catalogue_template.json`

Why this is important:

- it makes the mock store dynamic rather than code-embedded,
- it allows future stores to replace the catalogue without editing Python,
- it aligns the recommendation layer more closely with the attributes used by the search/parser subsystem.

The catalogue is therefore becoming an application-facing content layer rather than just a developer convenience.

### 5.2 `src/recommendations/engine.py`

This is a rule-based recommender.

Its logic:

1. map detected categories to complementary categories via `OUTFIT_RULES`,
2. accumulate rule scores,
3. sample catalogue items within those categories,
4. return the top `k`.

Why rule-based recommendations are used:

- they are explainable,
- computationally trivial,
- easy to debug,
- appropriate when there is no interaction history or user-profile data.

From an academic point of view, this is a symbolic recommender layered on top of a perceptual model.

That is a valid design choice because the recommendation problem here is not personalized recommendation at scale; it is contextual completion of an outfit.

Alternative approaches:

- collaborative filtering,
- content-based embedding similarity,
- graph-based outfit compatibility,
- learned stylistic compatibility scoring,
- CLIP-based multimodal recommendation.

Why those are not used here:

- they require more data, more modeling complexity, and often a real product corpus.

The present engine is intentionally interpretable and demonstrable.

One important note:

- the module docstring mentions embedding similarity as a strategy, but the current implementation is purely rule-based.

So the code reflects a simplified prototype rather than the full conceptual roadmap.

## 6. Custom detector research path

### 6.1 `src/custom_model/dataset.py`

This file adapts YOLO-format annotations into a PyTorch `Dataset` and `DataLoader`.

Core responsibilities:

- reading image/label pairs,
- applying Albumentations transforms,
- converting variable-length annotations into a collated target tensor.

Why Albumentations is used:

- it is strong for object-detection augmentation,
- especially because it can transform bounding boxes consistently with the image.

Why there are `light`, `medium`, and `heavy` augmentation modes:

- they support experimentation with regularization strength.

This is a practical research feature: augmentation intensity can materially affect small custom detectors trained from scratch.

Why a custom `collate_fn` is needed:

- each image has a different number of boxes,
- so default tensor stacking would fail.

This is standard for detection pipelines.

### 6.2 `src/custom_model/model.py`

This is the most research-oriented file in the repo.

It implements:

- basic convolution blocks,
- residual blocks,
- CSP-style blocks,
- a multi-scale backbone,
- an FPN-like neck,
- an anchor-free detection head,
- `FashionNet`,
- `TinyFashionNet`.

#### Why the architecture looks like this

Although it is described as custom, it is clearly inspired by modern one-stage detectors such as YOLO-family designs:

- downsampling backbone,
- multi-scale features,
- top-down fusion,
- per-scale prediction heads.

That is a sensible choice. Reinventing every design principle from zero would be academically weaker than intentionally adapting successful detector ideas.

#### `ConvBnRelu`, `ResBlock`, `CSPBlock`

These are the reusable structural primitives.

Why they exist:

- modularity,
- reduced repetition,
- clearer architectural semantics.

The CSP block is especially meaningful because it reflects awareness of compute/representation tradeoffs found in modern detectors.

#### `FashionBackbone`

This produces feature maps at three scales:

- P3,
- P4,
- P5.

Why multi-scale features are essential:

- clothing items vary in spatial scale,
- and some categories may appear as small localized regions while others span most of the person.

Alternative:

- single-scale detection,
- transformer-only encoder,
- pretrained backbone like ResNet/EfficientNet.

The current design prioritizes educational transparency and end-to-end control over raw performance.

#### `FashionNeck`

This is effectively an FPN/PAN-style fusion mechanism.

Why it is used:

- deep features have semantic richness but poor spatial precision,
- shallow features have better localization but weaker semantics,
- combining them improves detection performance across object sizes.

This is standard but important detector engineering.

#### `DetectionHead`

The head predicts:

- center offsets,
- width/height,
- objectness,
- class logits.

Why this matters:

- it turns dense feature maps into candidate detections,
- and does so in an anchor-free style.

Alternative:

- anchor-based heads,
- transformer decoder heads,
- center-based formulations like FCOS/CenterNet variants.

Anchor-free is a reasonable design because it reduces anchor-tuning complexity.

#### `TinyFashionNet`

This exists for pipeline verification rather than accuracy.

That is a very useful research engineering practice: maintain a cheap model that lets you validate code paths quickly on CPU before committing to long runs.

### 6.3 `src/custom_model/loss.py`

This file implements the custom detection loss.

Major components:

- CIoU box loss,
- focal binary cross-entropy for objectness,
- BCE class loss,
- target assignment across scales.

Why CIoU is used:

- IoU-only loss gives poor gradients when boxes do not overlap well,
- CIoU adds center-distance and aspect-ratio penalties,
- making optimization smoother.

Why focal BCE is used for objectness:

- dense detectors suffer severe foreground/background imbalance,
- most grid cells contain no object,
- focal loss down-weights easy negatives.

Why `build_targets()` is important

This function defines how ground truth is mapped to detection cells. In practice, target assignment is one of the most consequential pieces of a detector.

The optional `multi_cell` behavior is especially notable:

- it assigns a ground-truth object to neighboring cells when near boundaries,
- increasing positive signal density.

This is a practical approximation of richer assignment strategies.

Alternative assignment methods:

- anchor matching,
- dynamic label assignment such as SimOTA,
- center sampling,
- Hungarian matching.

Those are stronger or more modern in some contexts, but much harder to implement cleanly in a student project.

## 7. Utility code

### 7.1 `src/utils/metrics.py`

This file implements evaluation utilities from first principles.

Why this is academically useful:

- it makes the evaluation methodology visible,
- rather than fully outsourcing metrics to a framework.

Functions include:

- IoU computation,
- greedy prediction-to-ground-truth matching,
- per-class AP,
- confusion matrix generation and plotting,
- textual detection report,
- inference benchmarking.

The AP implementation is VOC-style 101-point interpolated AP. That is a legitimate and interpretable metric choice, though not identical to COCO's more exhaustive metric suite.

Alternative:

- rely entirely on Ultralytics' internal validation metrics.

The advantage of the custom metrics module is transparency and reuse for models outside Ultralytics, such as YOLO-World or FashionNet.

### 7.2 `src/utils/visualizer.py`

This module extends visualization beyond bare bounding boxes.

Why it exists:

- visualization is a debugging instrument in computer vision,
- not just a cosmetic feature.

The histogram and blended annotation views help reason about confidence distribution and display quality.

## 8. Frontend design and behavior

### 8.1 `frontend/index.html`

This file contains most of the active frontend implementation inline:

- HTML structure,
- a large embedded style block,
- a large embedded script block.

This is a deliberate prototype-oriented tradeoff.

Why this approach may have been chosen:

- single-file portability,
- easier demo deployment,
- reduced bundling complexity,
- simpler student iteration.

For a production frontend this would be too monolithic, but for a thesis prototype it is understandable.

The UI now supports three conceptual layers:

1. a landing screen for persona selection,
2. a camera-scanning mode,
3. a chat/voice refinement mode.

The camera experience is tightly aligned with the backend WebSocket state machine.

Important update:

- the page now uses a working `/api/chat` backend route,
- persona selection is stored client-side and propagated into scan/chat/session calls,
- the theme changes visually when the user selects `Cruella` or `Edna`,
- the user can return to the landing screen and reset the active session via a dedicated header control.

So the frontend should now be understood as a functioning integrated client rather than a partial interface stub.

### 8.2 `frontend/static/css/style.css`

This stylesheet is now a primary part of the runtime UI, not just supplementary decoration.

It encodes:

- the overall visual identity,
- responsive layout behavior,
- recommendation modal styling,
- persona-specific theming,
- interaction-state styling for scan/chat/recommendation components.

That matters because the current frontend no longer behaves like a neutral utility UI. It presents two distinct model personas visually as well as computationally.

## 9. Training and evaluation scripts

### 9.1 `scripts/sample_dataset.py`

This script performs stratified dataset sampling from DeepFashion2 using pre-built CSV metadata.

Why this matters:

- the raw dataset is large,
- full-scale experimentation may be expensive or unnecessary for a master's prototype,
- class-balanced sampling improves fairness across categories.

Why CSV-based indexing is smart:

- parsing many raw annotation files repeatedly is slow,
- a consolidated dataframe makes sampling dramatically faster.

Alternative:

- use the entire dataset,
- or build a more sophisticated sampler with weighting by occlusion/scale.

The current method is a good compromise between practicality and statistical coverage.

### 9.2 `scripts/train.py`

This script fine-tunes YOLOv8 through the Ultralytics API.

What it encodes:

- model-scale selection,
- pretrained vs from-scratch initialization,
- augmentation settings,
- optimizer and learning-rate choices,
- early stopping.

Why YOLOv8 fine-tuning is being used:

- strong baseline,
- fast iteration,
- high quality with relatively little custom engineering.

The script is effectively the "engineering baseline" against which the custom detector is compared.

### 9.3 `scripts/train_custom.py`

This is the training loop for FashionNet.

Why this file is important:

- it operationalizes the custom architecture,
- exposes experiment knobs,
- handles scheduling, EMA, checkpoints, and history logging.

Noteworthy research features:

- configurable loss weights,
- augmentation intensity,
- multi-cell assignment,
- dropout,
- optimizer choice,
- grayscale ablations,
- EMA,
- warmup and scheduler variants.

This is exactly the kind of script one expects in an experimental thesis repo: it is not just a train loop, but an experiment harness.

### 9.4 `scripts/evaluate.py`

This evaluates fine-tuned YOLOv8 models using the built-in Ultralytics validation flow and reports per-class AP.

Why it exists even though Ultralytics already validates during training:

- post hoc evaluation is often needed on specific checkpoints or confidence settings,
- and it creates a clearer reporting path for thesis tables.

### 9.5 `scripts/evaluate_yolo_world.py`

This evaluates the zero-shot backend using the project's own metrics utilities rather than Ultralytics' task-specific validation.

Why this is necessary:

- YOLO-World is used here as a custom-configured detector rather than a dataset-trained model,
- so a custom evaluation loop provides consistent comparison against the fine-tuned and custom models.

### 9.6 `scripts/compare_models.py`

This is one of the academically strongest scripts in the repo.

It compares:

- custom FashionNet,
- YOLOv8 baseline,
- or even custom-vs-custom checkpoints.

Metrics include:

- per-class mAP,
- overall mAP,
- inference speed,
- parameter count,
- weight size.

This is exactly what a comparative experimental section in a dissertation needs.

### 9.7 `scripts/analyze_raw_dataset.py`

This performs exploratory data analysis on the original dataset and produces figures.

Why this matters academically:

- dataset properties strongly influence model behavior,
- analyzing class balance, box size, aspect ratio, occlusion, and co-occurrence gives empirical justification for design decisions.

In other words, this script supports methodological rigor, not just convenience.

## 10. Test files

### 10.1 `tests/test_detector.py`

These tests validate basic detector behavior:

- return type,
- inference time existence,
- output shape,
- detection list structure,
- bounding-box validity,
- confidence range.

These are sanity tests rather than deep correctness tests.

Why that still matters:

- for ML systems, many failures are interface failures rather than theorem-level logical bugs,
- so smoke tests are valuable.

One limitation:

- the tests run on blank frames and base weights,
- so they validate pipeline integrity more than semantic accuracy.

### 10.2 `tests/test_recommendations.py`

These test:

- fallback behavior,
- `top_k`,
- required fields,
- non-duplication,
- expected dress/outwear pairing,
- exclusion logic.

These are good unit tests because the recommendation engine is deterministic enough in structure to verify meaningfully.

## 11. The `LNIAGIA` search subsystem in detail

This subsystem is conceptually separate from the image-based app.

It solves a different problem:

"Given a natural-language clothing request, retrieve suitable items from a structured catalog."

### 11.1 `LNIAGIA/DB/models.py`

This file is the ontology of the search system.

It defines:

- controlled vocabularies,
- field groups,
- realistic generation constraints,
- brand/price distributions,
- helper functions,
- the canonical set of filterable fields.

Why this file is foundational:

- it acts as domain schema,
- generator configuration,
- and retrieval vocabulary source all at once.

In database terms, it is part schema definition, part synthetic-data prior, and part business-rule layer.

This is powerful, but also creates tight coupling: many other modules depend on it as a single source of truth.

Alternative:

- split schema, generation rules, and search config into separate files.

That would improve separation of concerns, but the current single-file ontology is easier to navigate in a student project.

### 11.2 `LNIAGIA/DB/SQLLite/DBManager.py`

This manages a simple SQLite database for item records.

Why it exists alongside the vector DB:

- SQLite provides structured relational storage,
- Qdrant provides semantic retrieval.

This is a common hybrid pattern:

- relational storage for exact records,
- vector storage for semantic similarity.

### 11.3 `LNIAGIA/DB/vector/nl_mappings.py`

This maps compact symbolic values to richer natural-language descriptions and synonyms.

Why this is clever:

- embedding models work better with semantically rich text than with terse categorical tokens,
- so `short_sleeve_top` becomes something more linguistically meaningful like "short sleeve top (t-shirt, tee)".

This improves retrieval quality without changing the structured metadata.

### 11.4 `LNIAGIA/DB/vector/description_generator.py`

This transforms structured catalog items into descriptive text for embedding.

Why it is being used:

- vector search quality is only as good as the text representation being embedded,
- rich descriptions help the embedding model capture style, material, audience, and occasion semantics.

This is a standard retrieval trick: use synthetic natural-language enrichment to bridge structured data and semantic search.

### 11.5 `LNIAGIA/DB/vector/VectorDBManager.py`

This is the retrieval engine.

It handles:

- embedding model loading,
- Qdrant collection management,
- vector indexing,
- plain semantic search,
- filtered semantic search,
- strict vs soft exclusion behavior.

Why BGE is used:

- `BAAI/bge-base-en-v1.5` is a strong general-purpose embedding model for retrieval.

Why Qdrant local storage is used:

- easy local deployment,
- metadata filtering support,
- no external service requirement.

Why the `BGE_QUERY_PREFIX` matters:

- BGE models are optimized when queries are phrased with an instruction prefix,
- so this is a retrieval-quality optimization grounded in model-specific best practice.

#### Strict vs non-strict search

This is the central design idea.

- strict mode converts include/exclude constraints into hard metadata filters,
- non-strict mode retrieves more broadly and then penalizes mismatches.

This is a good information-retrieval design because real users often want a negotiable match, not a brittle exact filter.

One caveat:

- the code currently penalizes exclusions but leaves include-based soft boosting commented out.

So the non-strict mode is partially implemented relative to its conceptual ambition.

### 11.6 `LNIAGIA/llm_query_parser.py`

This uses Ollama with `qwen2.5:3b-instruct` to translate natural-language queries into structured filters.

Why this architecture is appealing:

- the LLM handles linguistic variability,
- the controlled vocabulary constrains outputs,
- validation cleans up invalid generations.

This is a classic "LLM-to-symbolic-IR bridge."

Why validation matters:

- LLMs are generative and not guaranteed to obey schemas perfectly,
- downstream retrieval needs clean, valid field values.

The parser therefore acts as a probabilistic front end, while `_validate()` converts it into a more deterministic system component.

In the current integrated application, this LLM path is specifically associated with the `Cruella` persona.

### 11.7 `LNIAGIA/search_app.py`

This is the CLI frontend for the search subsystem.

It:

- checks that the vector DB exists,
- loads the embedding model,
- accepts user queries,
- invokes the LLM parser,
- asks whether the user accepts approximate matches,
- runs filtered search,
- prints results.

From a system-design perspective, this file is the user-facing glue for the retrieval experiment.

### 11.8 `src/api/custom_text_parser.py`

This file introduces a non-LLM parsing path for the integrated app.

Its purpose is not to outperform the LLM parser linguistically; its purpose is to give the `Edna` persona a distinct text-processing behavior that is local, deterministic, and based on handcrafted matching rules.

What it does:

- parses messages against known mappings and vocabulary,
- handles simple include/exclude detection,
- supports generic garment synonyms,
- performs lightweight refinement merging.

Why this matters:

- it creates a meaningful model-family distinction between the two personas,
- it allows the app to demonstrate a custom NLP-style path instead of always routing through the LLM,
- it offers a controllable fallback when the project does not yet expose a separate trained text model checkpoint at runtime.

## 12. Code quality observations and implicit design tradeoffs

### 12.1 Strengths

- There is clear modular decomposition between API, detection, recommendations, custom model, and search.
- The detector abstraction is well chosen and supports backend swapping cleanly.
- The repository contains both engineering baselines and research-oriented custom implementations.
- Evaluation and comparison utilities are unusually thoughtful for a student project.
- The search subsystem shows strong awareness of hybrid symbolic + neural retrieval design.

### 12.2 Weaknesses or mismatches

- Some Pydantic schemas are weaker than they could be because nested models are not fully enforced.
- `LNIAGIA/DB/models.py` is powerful but overloaded; it acts as schema, generator, business rules, and search config simultaneously.
- The `Edna` text path is currently implemented through a custom deterministic parser adapter rather than a separately deployed trained text model artifact.
- Some conceptual claims in comments or docs are broader than the currently active implementation, especially around recommendation embeddings and soft include boosting.

These are normal prototype-stage characteristics, not fatal flaws.

## 13. Why the chosen technologies make sense

### 13.1 FastAPI

Used because it provides:

- asynchronous networking,
- schema-driven APIs,
- low boilerplate,
- clean Python integration.

Alternative:

- Flask,
- Django,
- Starlette directly.

FastAPI is the best fit among these for an ML demo API.

### 13.2 OpenCV + NumPy

Used because:

- detector input/output is image-array centric,
- camera capture and JPEG encoding are easy,
- the ecosystem is standard for CV prototypes.

Alternative:

- PIL only,
- imageio,
- browser-side capture with canvas overlays.

OpenCV remains the pragmatic choice.

### 13.3 Ultralytics YOLO

Used because:

- it gives a strong baseline quickly,
- training and inference APIs are streamlined,
- model variants are easy to compare.

Alternative:

- MMDetection,
- Detectron2,
- torchvision detection models.

Ultralytics optimizes for speed of experimentation, which aligns with the project.

### 13.4 Qdrant + sentence-transformers + Ollama

This stack makes sense for a local semantic search prototype because it is:

- open-source friendly,
- locally runnable,
- relatively easy to compose.

Alternative:

- Elasticsearch + dense vectors,
- FAISS + custom metadata layer,
- hosted embeddings plus a hosted vector DB.

The current choices optimize for independence and reproducibility.

## 14. How to read the repo efficiently

If a new researcher or evaluator wanted to understand the repo quickly, the best order is:

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

That order moves from deployed behavior to training methodology to secondary retrieval research.

## 15. Final interpretation

The repository should be understood as a hybrid master's-level applied AI project with two complementary themes:

- `visual understanding of clothing` through detection,
- `semantic understanding of clothing` through search and recommendation.

Its strongest qualities are:

- modular detector abstraction,
- meaningful evaluation tooling,
- a transparent custom model path,
- and a credible hybrid search architecture.

Its main limitations are typical of an evolving research prototype:

- some UI/backend mismatch,
- some unfinished integration surfaces,
- and a few files that carry both active logic and experimental residue.

Even with those limitations, the code clearly demonstrates thoughtful design decisions rather than random assembly. The repository is using widely accepted engineering patterns for ML systems, while also exposing enough internal implementation detail to support academic explanation, comparison, and critique.
