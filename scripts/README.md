# Scripts

The repository scripts are now grouped by purpose instead of living in one flat folder.

## Runtime

- `scripts/app/start_full_app.py`: backend launcher with search-stack checks
- `scripts/app/start_full_app.ps1`: PowerShell wrapper for local startup

## Data Preparation

- `scripts/data_prep/analyze_raw_dataset.py`: raw DeepFashion2 analysis and figures
- `scripts/data_prep/sample_dataset.py`: smaller sample dataset builder
- `scripts/data_prep/sample_balanced.py`: balanced train/val/test dataset builder

## Training

- `scripts/training/train.py`: YOLOv8 fine-tuning
- `scripts/training/train_custom.py`: FashionNet training

## Evaluation

- `scripts/evaluation/evaluate.py`: YOLOv8 evaluation
- `scripts/evaluation/evaluate_custom.py`: FashionNet evaluation
- `scripts/evaluation/evaluate_yolo_world.py`: YOLO-World zero-shot evaluation
- `scripts/evaluation/compare_models.py`: side-by-side model comparison
- `scripts/evaluation/visualize_results.py`: plots from training/evaluation outputs
