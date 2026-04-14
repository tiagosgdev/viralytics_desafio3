# Dataset Analysis

Source: DeepFashion2 dataset, processed from raw annotations into YOLO format.

---

## Raw Dataset Summary

**Total items: 364,676** across 13 clothing categories.

| Category | Count | % of Total | Median Box Area | Median Aspect Ratio | % Heavy Occlusion (occ=3) |
|---|---:|---:|---:|---:|---:|
| short_sleeve_top | 84,201 | 23.1% | 0.1941 | 0.96 | 4.3% |
| trousers | 64,973 | 17.8% | 0.1608 | 0.61 | 7.6% |
| long_sleeve_top | 42,030 | 11.5% | 0.2029 | 0.90 | 1.9% |
| shorts | 40,783 | 11.2% | 0.1142 | 1.10 | 12.4% |
| skirt | 37,357 | 10.2% | 0.1571 | 0.97 | 6.6% |
| vest_dress | 21,301 | 5.8% | 0.3139 | 0.60 | 12.6% |
| short_sleeve_dress | 20,338 | 5.6% | 0.3338 | 0.66 | 3.3% |
| vest | 18,208 | 5.0% | 0.1523 | 0.83 | 8.2% |
| long_sleeve_outwear | 15,468 | 4.2% | 0.3546 | 0.75 | 0.7% |
| long_sleeve_dress | 9,384 | 2.6% | 0.3474 | 0.68 | 3.6% |
| sling_dress | 7,641 | 2.1% | 0.2892 | 0.57 | 8.1% |
| sling | 2,307 | 0.6% | 0.2001 | 0.82 | 12.7% |
| short_sleeve_outwear | 685 | 0.2% | 0.3097 | 0.75 | 1.2% |

---

## Balanced Dataset

**Total items: 84,051** across **11 classes**.

### Excluded Classes

Two classes were excluded due to insufficient sample counts for stratified balancing:

- `sling` — only 2,307 items (not enough for representative train/val/test)
- `short_sleeve_outwear` — only 685 items

### Sampling Strategy

Each of the 11 remaining classes was sampled down to **7,641 items**, stratified by occlusion
level to preserve the original occlusion distribution within each class.

### Occlusion Distribution per Class

| Category | occ1 (visible) | occ2 (partial) | occ3 (heavy) |
|---|---:|---:|---:|
| short_sleeve_top | 4,753 | 2,562 | 326 |
| long_sleeve_top | 5,013 | 2,479 | 149 |
| long_sleeve_outwear | 4,923 | 2,662 | 56 |
| vest | 4,134 | 2,877 | 630 |
| shorts | 2,293 | 4,397 | 951 |
| trousers | 1,546 | 5,513 | 582 |
| skirt | 2,582 | 4,558 | 501 |
| short_sleeve_dress | 4,116 | 3,275 | 250 |
| long_sleeve_dress | 4,582 | 2,782 | 277 |
| vest_dress | 3,418 | 3,260 | 963 |
| sling_dress | 4,541 | 2,484 | 616 |

---

## Train / Val / Test Split

Split ratio: **70% train / 15% val / 15% test** (split by image, not annotation).

| Category | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| short_sleeve_top | 5,349 | 1,143 | 1,149 | 7,641 |
| long_sleeve_top | 5,340 | 1,152 | 1,149 | 7,641 |
| long_sleeve_outwear | 5,343 | 1,157 | 1,141 | 7,641 |
| vest | 5,349 | 1,149 | 1,143 | 7,641 |
| shorts | 5,333 | 1,171 | 1,137 | 7,641 |
| trousers | 5,350 | 1,158 | 1,133 | 7,641 |
| skirt | 5,359 | 1,148 | 1,134 | 7,641 |
| short_sleeve_dress | 5,367 | 1,121 | 1,153 | 7,641 |
| long_sleeve_dress | 5,338 | 1,143 | 1,160 | 7,641 |
| vest_dress | 5,343 | 1,158 | 1,140 | 7,641 |
| sling_dress | 5,356 | 1,132 | 1,153 | 7,641 |
| **TOTAL** | **58,827** | **12,632** | **12,592** | **84,051** |

---

## Notes on Dataset Design Decisions

- Stratified occlusion sampling ensures the model trains on realistic distributions of
  visibility — not just fully visible garments.
- The 70/15/15 split is by image (not instance), so a single multi-garment image is
  entirely in one split, preventing label leakage.
- sling_dress has the fewest raw images (7,641), which is why it is the cap value for
  all other classes.
