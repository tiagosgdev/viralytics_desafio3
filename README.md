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
│   │   ├── detector.py             # YOLOv8 inference wrapper
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
│       ├── css/style.css
│       └── js/app.js
│
├── scripts/
│   ├── sample_dataset.py           # Stratified dataset sampler
│   ├── train.py                    # Model training script
│   └── evaluate.py                 # Evaluation script
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

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sample the dataset (run once)
python scripts/sample_dataset.py --data_dir data/raw --output_dir data/sample_dataset --n_per_class 500

# 3. Convert annotations to YOLO format
python -c "from src.detection.converter import DeepFashion2ToYOLO; DeepFashion2ToYOLO('data/sample_dataset').convert()"

# 4. Train the model
python scripts/train.py --epochs 50 --model yolov8s --device 0
python scripts/train.py --epochs 16 --model yolov8s

# 5. Launch the API + camera
python -m uvicorn src.api.main:app --reload

# 6. Open the dashboard
open frontend/index.html
```

### `--device` flag options

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
