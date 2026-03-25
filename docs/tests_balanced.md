# Tests for the models trained on the balanced_dataset

## Test 1 — YOLOv8M | 50 epochs | batch=16 | ~7.429 hours

| Category | Images | Instances | Precision | Recall | mAP@50 | mAP@50:95 |
|----------|--------|-----------|-----------|--------|--------|-----------|
| **all** | **11186** | **12632** | **0.558** | **0.678** | **0.575** | **0.521** |
| short_sleeve_top | 1138 | 1143 | 0.318 | 0.535 | 0.293 | 0.269 |
| long_sleeve_top | 1145 | 1152 | 0.432 | 0.632 | 0.408 | 0.367 |
| long_sleeve_outwear | 1147 | 1157 | 0.692 | 0.775 | 0.702 | 0.644 |
| vest | 1138 | 1149 | 0.632 | 0.711 | 0.641 | 0.559 |
| shorts | 1165 | 1171 | 0.523 | 0.687 | 0.547 | 0.464 |
| trousers | 1146 | 1158 | 0.394 | 0.595 | 0.400 | 0.351 |
| skirt | 1143 | 1148 | 0.446 | 0.657 | 0.438 | 0.392 |
| short_sleeve_dress | 1116 | 1121 | 0.623 | 0.710 | 0.693 | 0.646 |
| long_sleeve_dress | 1128 | 1143 | 0.642 | 0.731 | 0.723 | 0.684 |
| vest_dress | 1145 | 1158 | 0.619 | 0.693 | 0.664 | 0.609 |
| sling_dress | 1120 | 1132 | 0.818 | 0.729 | 0.816 | 0.741 |

**Speed:** 0.1ms preprocess, 4.0ms inference, 0.0ms loss, 0.2ms postprocess per image

### Per-class mAP@50

| Category | mAP@50 |
|----------|--------|
| sling_dress | 0.8156 |
| long_sleeve_dress | 0.7235 |
| long_sleeve_outwear | 0.7021 |
| short_sleeve_dress | 0.6931 |
| vest_dress | 0.6641 |
| vest | 0.6412 |
| shorts | 0.5465 |
| skirt | 0.4381 |
| long_sleeve_top | 0.4077 |
| trousers | 0.4002 |
| short_sleeve_top | 0.2935 |

| Metric | Value |
|--------|-------|
| Overall mAP@50 | 0.5750 |
| Overall mAP@50:95 | 0.5207 |
| Precision | 0.5581 |
| Recall | 0.6778 |

### Analysis

Results are significantly worse than the sample_dataset tests (mAP@50: 0.575 vs 0.767). Key factors:

- **Smaller model** — YOLOv8M (25.8M params) vs YOLOv8L (43.6M params) used in previous tests
- **Harder validation set** — 11,186 val images vs 970, with a more uniform class distribution
- **Weakest classes** — short_sleeve_top (0.293) and trousers (0.400) dropped the most, likely due to higher visual confusion in the balanced set

---
