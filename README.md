# FashionSense
> Master's Project | Computer Vision + Semantic Fashion Search

FashionSense is now a single integrated application with two runtime personas:

- `Cruella`: trained YOLO-based outfit detection + LLM-powered text refinement
- `Edna`: custom FashionNet outfit detection + local custom text parsing

The app starts on a landing screen where the user chooses which persona to use. After that, the user can:

- scan an outfit with the live camera flow,
- get initial store recommendations,
- refine the search through chat or voice,
- switch back to the persona chooser and reset the active session.

---

## Quick Start

Web:

- `.\scripts\start_full_app.ps1`

LAN / mobile testing:

- `.\scripts\start_full_app.ps1 -BindHost 0.0.0.0 -BindPort 8000`

Android app:

- open `android_app/` in Android Studio
- build `Build > Build Bundle(s) / APK(s) > Build APK(s)`

---

## What The App Does

The runtime system combines:

1. clothing detection from a live camera or uploaded image
2. rule-based initial complementary recommendations
3. semantic follow-up search through natural language
4. voice transcription for chat input
5. a JSON-backed mock store catalogue for recommendation details

The current web experience has:

- a landing screen with `Cruella` and `Edna`
- separate `Camera` and `Chat` tabs
- a recommendation modal with item details
- a side carousel of store recommendations
- persona-specific visual themes

---

## Persona Modes

### Cruella

- Vision backend: trained YOLO detector
- Text backend: LNIAGIA conversation flow with the LLM parser
- Theme: dark red / black

### Edna

- Vision backend: custom `FashionNet`
- Text backend: local custom parser path
- Theme: current light neutral palette

Notes:

- `Cruella` is the more LLM-centric path.
- `Edna` is the more custom-model-centric path.
- If the custom FashionNet weights are missing, `Edna` falls back to the standard vision backend.

---

## Runtime Architecture

High-level runtime flow:

`Landing screen -> persona selection -> camera scan -> recommendations -> chat refinement`

Backend components:

- `src/api/main.py`
- `src/api/search_service.py`
- `src/api/personas.py`
- `src/api/custom_text_parser.py`
- `src/detection/detector.py`
- `src/detection/fashionnet_detector.py`
- `src/detection/camera.py`
- `src/recommendations/engine.py`

Frontend components:

- `frontend/index.html`
- `frontend/static/css/style.css`

Integrated search subsystem:

- `LNIAGIA/search_app.py`
- `LNIAGIA/llm_query_parser.py`
- `LNIAGIA/DB/vector/*`

---

## Project Structure

```text
viralytics_desafio3/
|
|-- android_app/                    # Native Android client
|-- data/
|   |-- mock_store_catalogue_template.json
|   `-- sample_dataset/
|-- docs/
|   `-- codebase_explanation.md
|-- frontend/
|   |-- index.html
|   `-- static/
|       `-- css/style.css
|-- LNIAGIA/                        # Semantic search subsystem
|-- models/
|   `-- weights/
|       |-- fashionnet/
|       |-- yolov8n_fashion/
|       `-- yolov8s_fashion/
|-- scripts/
|   |-- start_full_app.py
|   |-- start_full_app.ps1
|   |-- train.py
|   |-- train_custom.py
|   |-- evaluate.py
|   |-- evaluate_custom.py
|   `-- compare_models.py
`-- src/
    |-- api/
    |   |-- main.py
    |   |-- schemas.py
    |   |-- search_service.py
    |   |-- personas.py
    |   `-- custom_text_parser.py
    |-- custom_model/
    |-- detection/
    |   |-- detector.py
    |   |-- fashionnet_detector.py
    |   |-- yolo_world.py
    |   `-- camera.py
    `-- recommendations/
        |-- engine.py
        `-- catalogue.py
```

---

## Main API Surface

Important routes:

- `GET /`
- `GET /health`
- `POST /api/detect/image`
- `POST /api/mobile/scan`
- `POST /api/session/start`
- `GET /api/session/{session_id}`
- `POST /api/chat`
- `POST /api/chat/warmup`
- `POST /api/transcribe`
- `GET /api/conf`
- `POST /api/conf/{value}`
- `WS /ws/camera`

Important runtime behavior:

- scan/image/chat/session requests now carry a `persona`
- the camera WebSocket also uses the selected `persona`
- sessions store persona information
- chat results and recommendations are persona-aware

---

## Model Backends

### Vision

Available runtime paths:

- `FashionDetector` in `src/detection/detector.py`
- `FashionNetDetector` in `src/detection/fashionnet_detector.py`
- `YOLOWorldDetector` in `src/detection/yolo_world.py`

Current persona mapping:

| Persona | Vision backend |
|---------|----------------|
| `cruella` | trained YOLO |
| `edna` | FashionNet |

### Text

Available runtime paths:

- LLM-based parser / conversation flow in `LNIAGIA/`
- local custom parser in `src/api/custom_text_parser.py`

Current persona mapping:

| Persona | Text backend |
|---------|--------------|
| `cruella` | LLM conversation search |
| `edna` | custom local parser |

---

## Mobile / Android

The repository includes a native Android client in `android_app/`.

Current Android direction:

- native APK client
- separate scan and chat tabs
- default LAN server target
- recommendation cards and detail dialog

The Android app uses the backend running on the PC as the server.

---

## Catalogue

Recommendations are now backed by the editable JSON catalogue:

- `data/mock_store_catalogue_template.json`

This file is intended to be:

- easy to edit manually
- replaceable per store
- extensible with new attributes

Recommendation details shown in the UI are populated from this data source.

---

## Training / Evaluation Scripts

YOLO path:

- `scripts/train.py`
- `scripts/evaluate.py`

Custom FashionNet path:

- `scripts/train_custom.py`
- `scripts/evaluate_custom.py`
- `scripts/compare_models.py`

These support the two main research directions in the repository:

- strong fine-tuned baseline with YOLO
- from-scratch detector experimentation with FashionNet

---

## Notes

- The app launcher checks for integrated-search dependencies before starting.
- Voice transcription depends on Whisper plus `ffmpeg`.
- The current frontend and backend are aligned around `/api/chat`; this is no longer just a placeholder.
- The app now has a working persona-selection layer, so documentation or older notes referring to a single startup detector are outdated.
