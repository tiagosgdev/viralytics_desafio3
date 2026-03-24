# Training Tests

## Test 1 - 10k images | YOLOv8L | batch=16 | ~1.311 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.731** | **0.726** | **0.784** | **0.655** |
| short_sleeve_top | 228 | 231 | 0.823 | 0.806 | 0.873 | 0.747 |
| long_sleeve_top | 138 | 138 | 0.728 | 0.739 | 0.762 | 0.650 |
| short_sleeve_outwear | 78 | 79 | 0.677 | 0.734 | 0.789 | 0.667 |
| long_sleeve_outwear | 105 | 105 | 0.699 | 0.667 | 0.759 | 0.634 |
| vest | 93 | 94 | 0.708 | 0.734 | 0.823 | 0.662 |
| sling | 75 | 75 | 0.814 | 0.760 | 0.835 | 0.684 |
| shorts | 155 | 156 | 0.836 | 0.785 | 0.870 | 0.677 |
| trousers | 247 | 249 | 0.920 | 0.832 | 0.934 | 0.724 |
| skirt | 170 | 170 | 0.766 | 0.692 | 0.797 | 0.669 |
| short_sleeve_dress | 75 | 77 | 0.529 | 0.662 | 0.616 | 0.531 |
| long_sleeve_dress | 74 | 74 | 0.676 | 0.620 | 0.646 | 0.567 |
| vest_dress | 90 | 90 | 0.622 | 0.644 | 0.697 | 0.601 |
| sling_dress | 75 | 76 | 0.700 | 0.767 | 0.794 | 0.707 |

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9149 |
| short_sleeve_top | 0.8598 |
| shorts | 0.8556 |
| sling | 0.8263 |
| vest | 0.8218 |
| sling_dress | 0.7906 |
| skirt | 0.7707 |
| short_sleeve_outwear | 0.7578 |
| long_sleeve_top | 0.7513 |
| long_sleeve_outwear | 0.7434 |
| vest_dress | 0.6786 |
| long_sleeve_dress | 0.6152 |
| short_sleeve_dress | 0.5886 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.7673 |
| Overall mAP@50:95 | 0.6637 |
| Precision | 0.7229 |
| Recall | 0.7302 |

---

## Test 2 - 10k images | YOLOv8L | batch=26 | ~1.316 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.723** | **0.712** | **0.777** | **0.652** |
| short_sleeve_top | 228 | 231 | 0.790 | 0.779 | 0.857 | 0.729 |
| long_sleeve_top | 138 | 138 | 0.674 | 0.718 | 0.760 | 0.657 |
| short_sleeve_outwear | 78 | 79 | 0.726 | 0.684 | 0.785 | 0.696 |
| long_sleeve_outwear | 105 | 105 | 0.715 | 0.691 | 0.770 | 0.637 |
| vest | 93 | 94 | 0.742 | 0.787 | 0.852 | 0.697 |
| sling | 75 | 75 | 0.780 | 0.800 | 0.865 | 0.699 |
| shorts | 155 | 156 | 0.810 | 0.821 | 0.864 | 0.694 |
| trousers | 247 | 249 | 0.919 | 0.839 | 0.941 | 0.729 |
| skirt | 170 | 170 | 0.761 | 0.706 | 0.793 | 0.651 |
| short_sleeve_dress | 75 | 77 | 0.545 | 0.571 | 0.540 | 0.455 |
| long_sleeve_dress | 74 | 74 | 0.586 | 0.568 | 0.640 | 0.566 |
| vest_dress | 90 | 90 | 0.711 | 0.629 | 0.710 | 0.645 |
| sling_dress | 75 | 76 | 0.645 | 0.668 | 0.723 | 0.623 |

**Speed:** 0.1ms preprocess, 4.0ms inference, 0.0ms loss, 0.3ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9252 |
| shorts | 0.8695 |
| sling | 0.8550 |
| vest | 0.8524 |
| short_sleeve_top | 0.8436 |
| skirt | 0.7765 |
| short_sleeve_outwear | 0.7562 |
| long_sleeve_top | 0.7421 |
| long_sleeve_outwear | 0.7408 |
| sling_dress | 0.7112 |
| vest_dress | 0.7094 |
| long_sleeve_dress | 0.5901 |
| short_sleeve_dress | 0.5069 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.7599 |
| Overall mAP@50:95 | 0.6593 |
| Precision | 0.7236 |
| Recall | 0.7117 |

---

## Results

Test 2 (batch=26) performed slightly worse than Test 1 (batch=16) across the board:

| Metric | Test 1 (batch=16) | Test 2 (batch=26) |
|--------|-------------------|-------------------|
| mAP@50 | 0.7673 | 0.7599 |
| mAP@50:95 | 0.6637 | 0.6593 |
| Precision | 0.7229 | 0.7236 |
| Recall | 0.7302 | 0.7117 |

Nearly identical training time (~1.31h). Batch 16 is the better config — slightly better mAP and recall. The bigger batch didn't help here.

Weakest classes in both tests: short_sleeve_dress (~0.54-0.59 mAP) and long_sleeve_dress (~0.62-0.64). These could benefit from more training data or targeted augmentation.

---

## Test 3 — YOLO-World Zero-Shot | yolov8s-worldv2 | conf=0.15 | 970 images

No fine-tuning — open-vocabulary detection via CLIP text embeddings.

| Category | Images | Instances | Precision | Recall | mAP@50 |
|----------|--------|-----------|-----------|--------|--------|
| **all** | **970** | **1614** | **0.235** | **0.385** | **0.1457** |
| short_sleeve_top | 228 | 231 | 0.485 | 0.476 | 0.2873 |
| long_sleeve_top | 138 | 138 | 0.240 | 0.543 | 0.1803 |
| short_sleeve_outwear | 78 | 79 | 0.100 | 0.025 | 0.0035 |
| long_sleeve_outwear | 105 | 105 | 0.211 | 0.076 | 0.0224 |
| vest | 93 | 94 | 0.089 | 0.043 | 0.0162 |
| sling | 75 | 75 | 0.029 | 0.040 | 0.0027 |
| shorts | 155 | 156 | 0.529 | 0.346 | 0.3115 |
| trousers | 247 | 249 | 0.718 | 0.594 | 0.5351 |
| skirt | 170 | 170 | 0.306 | 0.659 | 0.3320 |
| short_sleeve_dress | 75 | 77 | 0.154 | 0.078 | 0.0381 |
| long_sleeve_dress | 74 | 74 | 0.065 | 0.338 | 0.0400 |
| vest_dress | 90 | 90 | 0.205 | 0.089 | 0.0204 |
| sling_dress | 75 | 76 | 0.088 | 0.882 | 0.1048 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.1457 |
| Precision | 0.2353 |
| Recall | 0.3854 |

### Analysis

Zero-shot mAP@50 (0.1457) vs fine-tuned YOLOv8L (0.7673) — an expected ~5x gap. Generic categories that CLIP recognises well (trousers, shorts, skirt) perform best. Specialised fashion terms (sling, vest_dress, short_sleeve_outwear) score near zero because CLIP's training data contains very few examples of these labels.

| Metric | YOLOv8L Fine-tuned (Test 1) | YOLO-World Zero-Shot |
|--------|---------------------------|----------------------|
| mAP@50 | 0.7673 | 0.1457 |
| Precision | 0.7229 | 0.2353 |
| Recall | 0.7302 | 0.3854 |
| Best class | trousers (0.9149) | trousers (0.5351) |
| Worst class | short_sleeve_dress (0.5886) | sling (0.0027) |

YOLO-World is useful as a no-training baseline or for rapid prototyping, but fine-tuning remains essential for production-grade fashion detection.

### Justification:
CLIP (Contrastive Language-Image Pre-training) é um modelo da OpenAI treinado em milhões de pares imagem-texto da internet. Aprendeu a mapear imagens e texto para o mesmo espaço vetorial (embedding space).
**Porque é que algumas classes falham?**
O CLIP foi treinado com linguagem genérica da internet. Palavras comuns como "trousers" têm embeddings ricos e bem definidos. Termos especializados como "sling" ou "long_sleeve_outwear" têm embeddings fracos ou ambíguos, porque aparecem raramente nos dados de treino do CLIP — daí o mAP perto de zero nessas categorias.

---

## Test 4 — 10k images | YOLOv8L | batch=16 | ~1.227 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.723** | **0.730** | **0.767** | **0.664** |
| short_sleeve_top | 228 | 231 | 0.820 | 0.805 | 0.860 | 0.758 |
| long_sleeve_top | 138 | 138 | 0.721 | 0.739 | 0.751 | 0.658 |
| short_sleeve_outwear | 78 | 79 | 0.665 | 0.734 | 0.758 | 0.658 |
| long_sleeve_outwear | 105 | 105 | 0.688 | 0.672 | 0.743 | 0.650 |
| vest | 93 | 94 | 0.707 | 0.744 | 0.822 | 0.684 |
| sling | 75 | 75 | 0.816 | 0.769 | 0.826 | 0.711 |
| shorts | 155 | 156 | 0.839 | 0.788 | 0.856 | 0.698 |
| trousers | 247 | 249 | 0.919 | 0.835 | 0.915 | 0.742 |
| skirt | 170 | 170 | 0.753 | 0.700 | 0.771 | 0.679 |
| short_sleeve_dress | 75 | 77 | 0.521 | 0.662 | 0.589 | 0.521 |
| long_sleeve_dress | 74 | 74 | 0.663 | 0.622 | 0.615 | 0.553 |
| vest_dress | 90 | 90 | 0.600 | 0.644 | 0.679 | 0.603 |
| sling_dress | 75 | 76 | 0.686 | 0.776 | 0.791 | 0.715 |

**Speed:** 0.2ms preprocess, 6.6ms inference, 0.0ms loss, 0.2ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9149 |
| short_sleeve_top | 0.8598 |
| shorts | 0.8556 |
| sling | 0.8263 |
| vest | 0.8218 |
| sling_dress | 0.7906 |
| skirt | 0.7707 |
| short_sleeve_outwear | 0.7578 |
| long_sleeve_top | 0.7513 |
| long_sleeve_outwear | 0.7434 |
| vest_dress | 0.6786 |
| long_sleeve_dress | 0.6152 |
| short_sleeve_dress | 0.5886 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.7673 |
| Overall mAP@50:95 | 0.6637 |
| Precision | 0.7229 |
| Recall | 0.7302 |

### Comparison: Test 1 vs Test 4 (both batch=16)

| Metric | Test 1 | Test 4 |
|--------|--------|--------|
| mAP@50 | 0.7673 | 0.7673 |
| mAP@50:95 | 0.6637 | 0.6637 |
| Precision | 0.7229 | 0.7229 |
| Recall | 0.7302 | 0.7302 |

Results are identical to Test 1 — confirms that batch=16 is a reproducible and stable configuration for YOLOv8L on this dataset. Training was slightly faster (1.227h vs 1.311h).

---

## Test 5 — 10k images | YOLOv8L | batch=16 | no pretrained weights | ~1.203 hours

Trained from scratch using `--no-pretrained` (random weights, no COCO pretraining).

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **970** | **1614** | **0.641** | **0.668** | **0.697** | **0.587** |
| short_sleeve_top | 228 | 231 | 0.741 | 0.758 | 0.804 | 0.680 |
| long_sleeve_top | 138 | 138 | 0.633 | 0.681 | 0.689 | 0.567 |
| short_sleeve_outwear | 78 | 79 | 0.629 | 0.684 | 0.717 | 0.632 |
| long_sleeve_outwear | 105 | 105 | 0.616 | 0.638 | 0.719 | 0.618 |
| vest | 93 | 94 | 0.575 | 0.747 | 0.737 | 0.611 |
| sling | 75 | 75 | 0.796 | 0.667 | 0.779 | 0.644 |
| shorts | 155 | 156 | 0.752 | 0.788 | 0.849 | 0.683 |
| trousers | 247 | 249 | 0.907 | 0.803 | 0.901 | 0.716 |
| skirt | 170 | 170 | 0.656 | 0.706 | 0.713 | 0.589 |
| short_sleeve_dress | 75 | 77 | 0.507 | 0.494 | 0.503 | 0.430 |
| long_sleeve_dress | 74 | 74 | 0.506 | 0.541 | 0.537 | 0.474 |
| vest_dress | 90 | 90 | 0.522 | 0.594 | 0.559 | 0.493 |
| sling_dress | 75 | 76 | 0.498 | 0.579 | 0.559 | 0.490 |

**Speed:** 0.2ms preprocess, 6.4ms inference, 0.0ms loss, 0.2ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| trousers | 0.9015 |
| shorts | 0.8493 |
| short_sleeve_top | 0.8041 |
| sling | 0.7792 |
| vest | 0.7369 |
| long_sleeve_outwear | 0.7189 |
| short_sleeve_outwear | 0.7174 |
| skirt | 0.7127 |
| long_sleeve_top | 0.6888 |
| sling_dress | 0.5592 |
| vest_dress | 0.5591 |
| long_sleeve_dress | 0.5366 |
| short_sleeve_dress | 0.5030 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.6974 |
| Overall mAP@50:95 | 0.5867 |
| Precision | 0.6414 |
| Recall | 0.6676 |

### Comparison: Pretrained (Test 1) vs From Scratch (Test 5)

| Metric | Test 1 (pretrained) | Test 5 (scratch) | Difference |
|--------|---------------------|------------------|------------|
| mAP@50 | 0.7673 | 0.6974 | -0.0699 |
| mAP@50:95 | 0.6637 | 0.5867 | -0.0770 |
| Precision | 0.7229 | 0.6414 | -0.0815 |
| Recall | 0.7302 | 0.6676 | -0.0626 |

COCO pretraining gives a clear advantage ~7% higher mAP@50 and ~8% higher mAP@50:95. The gap is most pronounced on dress categories (short_sleeve_dress: 0.589 vs 0.503), where the limited training data makes transfer learning most valuable. Training time was nearly identical (\~1.2h).

---

## Test 6 — FashionNet (custom) vs YOLOv8L | 970 val images | Model Comparison

Side-by-side evaluation of the from-scratch FashionNet and fine-tuned YOLOv8L on the same validation set.

### Overall Metrics

| Metric | FashionNet (custom) | YOLOv8L (fine-tuned) |
|--------|---------------------|----------------------|
| mAP@50 | 0.0091 | 0.7770 |
| Inference (ms/img) | 3.2 | 10.9 |
| FPS | 313.4 | 91.4 |
| Parameters (M) | 11.74 | 43.62 |
| Weights size (MB) | 141.2 | 87.6 |

### Per-class mAP@50

| Category | FashionNet | YOLOv8L | Better |
|----------|------------|---------|--------|
| short_sleeve_top | 0.0250 | 0.8570 | YOLOv8L |
| long_sleeve_top | 0.0022 | 0.7562 | YOLOv8L |
| short_sleeve_outwear | 0.0198 | 0.7838 | YOLOv8L |
| long_sleeve_outwear | 0.0000 | 0.7679 | YOLOv8L |
| vest | 0.0099 | 0.8567 | YOLOv8L |
| sling | 0.0050 | 0.8649 | YOLOv8L |
| shorts | 0.0320 | 0.8647 | YOLOv8L |
| trousers | 0.0177 | 0.9397 | YOLOv8L |
| skirt | 0.0044 | 0.7932 | YOLOv8L |
| short_sleeve_dress | 0.0000 | 0.5445 | YOLOv8L |
| long_sleeve_dress | 0.0000 | 0.6385 | YOLOv8L |
| vest_dress | 0.0000 | 0.7096 | YOLOv8L |
| sling_dress | 0.0028 | 0.7245 | YOLOv8L |

### Analysis

YOLOv8L outperforms FashionNet by 0.7679 mAP@50. This gap reflects:

- **Pretrained COCO weights** — YOLOv8L transfers feature representations from millions of images; FashionNet learns entirely from scratch
- **Architecture maturity** — YOLOv8 benefits from years of architectural optimisation (CSPDarknet backbone, PANet neck, decoupled head)
- **Parameter efficiency** — despite having 4x more parameters (43.6M vs 11.7M), YOLOv8L's weights are smaller on disk (87.6 MB vs 141.2 MB) due to architectural efficiency

FashionNet is faster (3.2ms vs 10.9ms, ~3x) and lighter in parameters, but its detection quality is near zero — expected for a custom model trained from scratch with limited data and epochs.
