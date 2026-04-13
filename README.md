# 👗 FashionSense — Real-Time Clothing Detection System
> Master's Project | Deep Learning & Neural Networks

A real-time clothing detection system using YOLOv8 fine-tuned on DeepFashion2,
with a live camera feed, REST API, and store recommendation engine.

---

## Project Structure

```
fashion-detector/
│
├── data/
│   ├── raw/                        # Original DeepFashion2 (17GB, not committed)
│   └── sample_dataset/             # Stratified sample (~3-5GB)
│       ├── images/
│       └── annos/
│
├── models/
│   └── weights/                    # Saved .pt model checkpoints
│
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py             # Base detector ABC + YOLOv8 wrapper
│   │   ├── yolo_world.py           # YOLO-World zero-shot detector
│   │   ├── camera.py               # Real-time camera pipeline
│   │   └── converter.py            # DeepFashion2 → YOLO format
│   ├── recommendations/
│   │   ├── __init__.py
│   │   ├── engine.py               # Rule-based + embedding recommendation engine
│   │   └── catalogue.py            # Mock store catalogue
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   └── schemas.py              # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── visualizer.py           # Bounding box drawing utilities
│       └── metrics.py              # Evaluation helpers
│
├── frontend/
│   ├── index.html                  # Main dashboard UI
│   └── static/
│       └── css/style.css
│
├── scripts/
│   ├── sample_dataset.py           # Stratified dataset sampler
│   ├── train.py                    # Model training script
│   ├── evaluate.py                 # Evaluation script (fine-tuned model)
│   └── evaluate_yolo_world.py      # Evaluation script (YOLO-World zero-shot)
│
├── notebooks/
│   └── 01_EDA.ipynb                # Exploratory Data Analysis
│
├── tests/
│   ├── test_detector.py
│   └── test_recommendations.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quick Start

- `.\scripts\start_full_app.ps1` - run web
- `.\scripts\start_full_app.ps1 -BindHost 0.0.0.0 -BindPort 8000` - run mobile

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sample the dataset (run once)
python scripts/sample_dataset.py --data_dir data/raw --output_dir data/sample_dataset --n_per_class 500

# 3. Convert annotations to YOLO format
python -c "from src.detection.converter import DeepFashion2ToYOLO; DeepFashion2ToYOLO('data/sample_dataset').convert()"

# 4. Train the model
python scripts/train.py --epochs 50 --model yolov8s --device 0

# 5. Launch the API + camera
python -m uvicorn src.api.main:app --reload

# 5b. Or launch with YOLO-World zero-shot (no fine-tuning needed)
DETECTOR_BACKEND=yolo_world uvicorn src.api.main:app --reload

# 6. Open the dashboard
open frontend/index.html

# 7. Evaluate YOLO-World zero-shot on the validation set
python scripts/evaluate_yolo_world.py
```

### `train.py` flags (YOLOv8 fine-tuning)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | `yolov8s` | YOLOv8 variant: `yolov8n` (nano), `yolov8s` (small), `yolov8m` (medium), `yolov8l` (large) |
| `--epochs` | int | `50` | Number of training epochs |
| `--batch` | int | `16` | Batch size |
| `--imgsz` | int | `640` | Input image size (pixels) |
| `--data` | str | `data/sample_dataset/yolo/dataset.yaml` | Path to dataset YAML |
| `--output_dir` | str | `models/weights` | Parent directory for training runs |
| `--workers` | int | `2` | Number of DataLoader workers |
| `--device` | str | `0` | Hardware target (see table below) |
| `--no-pretrained` | flag | off | Train from scratch (random weights, no COCO pretraining). Appends `_scratch` to run name |
| `--wandb` | flag | off | Enable Weights & Biases logging |

**Output:** Results are saved to `models/weights/{model}_fashion/` (or `{model}_fashion_scratch` with `--no-pretrained`). If the folder already exists, YOLO auto-increments the name (e.g. `yolov8s_fashion2`).

### `train_custom.py` flags (FashionNet from scratch)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--epochs` | int | `50` | Number of training epochs |
| `--batch` | int | `8` | Batch size (use 4-8 on CPU, 16+ on GPU) |
| `--imgsz` | int | `640` | Input image size (pixels) |
| `--lr` | float | `1e-3` | Learning rate |
| `--device` | str | auto | Hardware target: `cpu`, `cuda`, `mps`, or empty for auto-detect |
| `--workers` | int | `0` | DataLoader workers (0 = main process, safest) |
| `--data` | str | `data/sample_dataset/yolo` | Path to YOLO directory containing `images/` and `labels/` |
| `--output` | str | `models/weights/fashionnet` | Output directory for checkpoints |
| `--resume` | str | — | Path to a `.pt` checkpoint to resume training from |
| `--fast` | flag | off | Use TinyFashionNet (fewer channels) for quick testing |
| `--max_samples` | int | `0` | Cap dataset size for quick testing (0 = use all) |

**Output:** Saves `best.pt`, `last.pt`, and `history.json` to the `--output` directory. **Warning:** re-running with the same `--output` path overwrites previous results — use a different path to preserve them.

### `--device` values

| Value | Hardware | Example |
|-------|----------|---------|
| `0` | First NVIDIA GPU (CUDA) | `--device 0` |
| `0,1` | Multiple NVIDIA GPUs | `--device 0,1` |
| `mps` | Apple Silicon GPU (Mac M1/M2/M3) | `--device mps` |
| `cpu` | CPU only (slowest) | `--device cpu` |

---

## Categories Detected (13 classes)

| ID | Category | ID | Category |
|----|----------|----|----------|
| 1 | Short Sleeve Top | 8 | Trousers |
| 2 | Long Sleeve Top | 9 | Skirt |
| 3 | Short Sleeve Outwear | 10 | Short Sleeve Dress |
| 4 | Long Sleeve Outwear | 11 | Long Sleeve Dress |
| 5 | Vest | 12 | Vest Dress |
| 6 | Sling | 13 | Sling Dress |
| 7 | Shorts | | |

---

## Detector Backends

The API supports two detection backends, selected via the `DETECTOR_BACKEND` environment variable:

| Backend | Env value | Model | Needs training? | mAP@50 |
|---------|-----------|-------|-----------------|--------|
| YOLOv8 (default) | `yolov8` | Fine-tuned `best.pt` | Yes | **0.767** |
| YOLO-World | `yolo_world` | `yolov8s-worldv2.pt` | No (zero-shot) | 0.146 |

```bash
# Fine-tuned YOLOv8 (default)
uvicorn src.api.main:app --reload

# YOLO-World zero-shot
DETECTOR_BACKEND=yolo_world uvicorn src.api.main:app --reload
```

---

## Architecture

```
Camera (OpenCV) → YOLOv8 Inference → Detection Results
                                           ↓
                              Recommendation Engine
                                           ↓
                              FastAPI REST Endpoint
                                           ↓
                              Browser Dashboard (Live Feed)
```
