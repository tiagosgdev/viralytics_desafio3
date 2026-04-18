# FashionNet Training Process

## What Training Actually Is

The model starts with random weights — every number in every layer is initialised to a small random value. With random weights, every prediction the model makes is wrong. Training is the process of fixing that.

The core idea: repeatedly show the model labelled examples (images where we already know where the clothes are and what category they are), measure how wrong the model's prediction was (this measurement is called the **loss**), figure out which direction each weight should move to make the prediction less wrong (this is called the **gradient**), then move all weights a small amount in that direction. Repeat this thousands of times until the model's predictions are good.

The full loop per batch of images is:

```
data → forward pass → loss → backward pass → optimizer step → repeat
```

**Batch**: a group of images processed together in one iteration. Instead of processing one image at a time (slow) or all images at once (runs out of memory), images are grouped into batches of e.g. 8 or 16.

**Epoch**: one full pass through the entire training dataset. A training run of 50 epochs means the model sees every training image 50 times total.

**Weight / Parameter**: a single learnable number inside the model. FashionNet-s has ~11.7 million of these. Training adjusts all of them simultaneously.

---

## Step 0 — Setup (runs once, before the loop)

Code location: `scripts/training/train_custom.py`, `main()` function, line 207.

### Dataset and DataLoader

`FashionDataset` (`src/custom_model/dataset.py`, line 100) pairs each image file with its label file. Label files are plain text, one line per clothing item:

```
class_id  cx  cy  w  h
```

Where `cx`, `cy` are the centre of the bounding box, `w` and `h` are width and height — all expressed as fractions between 0 and 1 relative to the image size. This is called **YOLO format** (named after the YOLO family of detectors that popularised it).

`DataLoader` (a PyTorch built-in) wraps the dataset. It handles:
- Shuffling the order of images each epoch so the model does not memorise the order
- Grouping images into batches
- Running image loading and augmentation in background processes (workers) so the GPU is not idle waiting for data

Every time the training loop asks for the next batch, the DataLoader returns:
- `images`: shape `(B, 3, 640, 640)` — B images, 3 colour channels (red/green/blue), 640×640 pixels. Values are normalised (not raw 0–255 integers).
- `targets`: shape `(N, 6)` — all ground-truth boxes across all B images flattened into one list. Each row: `[batch_index, class_id, cx, cy, w, h]`. The batch index tells the loss which image in the batch this box belongs to.

Code: `build_dataloaders()` in `src/custom_model/dataset.py` line 306. Called at `scripts/training/train_custom.py` line 248.

---

### Model

`FashionNet` (`src/custom_model/model.py`, line 278) is created and moved to the compute device (GPU or CPU).

**Kaiming initialisation** (`src/custom_model/model.py` line 177): weights in convolutional layers are not set to zero or random uniform values — they are set using a formula that accounts for the number of inputs to each layer. This keeps the signal from exploding or vanishing as it passes through many layers at the start of training. BatchNorm layers start with scale (gamma) = 1, shift (beta) = 0, meaning they initially do nothing.

Code: `scripts/training/train_custom.py` line 261.

---

### Loss Function

`FashionNetLoss` (`src/custom_model/loss.py`, line 176) is instantiated once. It is a callable object — calling it with predictions and targets computes the loss. It holds the loss weight constants (lambda_box, lambda_obj, lambda_cls) but has no learnable parameters of its own.

Code: `scripts/training/train_custom.py` line 267.

---

### Optimizer

The optimizer holds references to all model parameters and is responsible for actually updating them after each backward pass.

Two options (`scripts/training/train_custom.py` line 278):

**AdamW** (default): Adam with decoupled weight decay. Adam stands for Adaptive Moment Estimation. It keeps a running average of past gradients (momentum) and a running average of past squared gradients (velocity) per parameter. This lets it adapt the effective learning rate individually for each weight. Parameters that receive large gradients get a smaller effective step; rarely updated parameters get a larger effective step.

**Weight decay**: a penalty added each step that pulls every weight slightly toward zero. Prevents weights from growing very large (overfitting). In AdamW, weight decay is applied separately from the gradient update, which is more correct than the original Adam formulation.

**Momentum** (in both Adam and SGD): instead of following the raw gradient each step, momentum keeps a running average of past gradients and follows that average. This smooths out noisy updates — one bad batch does not send the weights in a wildly wrong direction.

**Learning rate (lr)**: how large a step to take each update. Too large → weights overshoot and training diverges. Too small → training is slow. This is why a scheduler is used.

Code: `scripts/training/train_custom.py` line 284.

---

### Scheduler

Controls how the learning rate changes over training. `OneCycleLR` (default, `scripts/training/train_custom.py` line 306):

```
Epoch:  0%         10%                              100%
LR:     low   →   max_lr   →   cosine decay   →    ~0
```

Starts at a low learning rate, quickly ramps up to `max_lr` over the first 10% of training steps (`pct_start=0.1`), then follows a cosine curve down to near zero. The ramp-up prevents large destructive updates when weights are still random. The cosine decay allows fine-tuning adjustments late in training without overshooting.

**Cosine decay**: the learning rate follows a cosine curve from high to low, which is smoother than linear decay — it drops slowly at first, faster in the middle, then slowly again at the end.

`CosineAnnealingLR` is the alternative scheduler (if `--cos_lr` flag used). Steps once per epoch instead of once per batch.

Code: `scripts/training/train_custom.py` line 306.

---

### EMA (Exponential Moving Average)

`ModelEMA` (`scripts/training/train_custom.py` line 119): maintains a shadow copy of all model weights. After every batch update, the shadow copy is updated as:

```
shadow_weight = decay × shadow_weight + (1 - decay) × real_weight
```

With `decay = 0.9999`, the shadow copy changes very slowly — it is an average of the model weights over the last ~10,000 steps. This averaging smooths out the noise from individual bad batches and typically produces more stable, better-generalising weights than the raw model. The EMA copy is what gets saved to `best.pt` and used at inference time.

Code: `scripts/training/train_custom.py` line 315.

---

## Step 1 — Data Loading and Augmentation

Code: `src/custom_model/dataset.py`, `__getitem__` line 251, `_mosaic4` line 171, `get_train_transforms` line 31.

Every time the DataLoader requests an image, `FashionDataset.__getitem__` runs.

### Mosaic Augmentation (if enabled)

Before any other transforms, four random images are combined into one. Each image is letterbox-resized (resized to fit inside 640×640 while preserving aspect ratio, with grey padding), then one quadrant of each is taken and placed into a single canvas. Bounding boxes are adjusted to match their new positions on the combined canvas.

**Why**: gives the model more varied training scenes in each sample. A single 640×640 canvas now contains clothing from four different images and locations. Helps the model learn from densely packed, varied scenes.

**Letterbox resize**: resize an image so the longest edge becomes the target size, then pad the shorter edge with a neutral grey (pixel value 114) to make it square. Preserves the original aspect ratio — no squashing distortion.

Code: `src/custom_model/dataset.py` line 171.

### Augmentation Pipeline

Applied via `albumentations`, a library specialising in image augmentations that correctly transform bounding boxes alongside the image.

Steps in order:

1. **LongestMaxSize** (line 41): resize image so longest edge = 640. Preserves aspect ratio.
2. **PadIfNeeded** (line 42): pad to 640×640 with grey (constant border). Same as letterbox.
3. **Spatial transforms** (vary by level — `light`, `medium`, `heavy`):
   - Horizontal flip: mirrors the image left-right. Bounding boxes are mirrored accordingly.
   - Affine: rotate, scale, and translate the image slightly. Teaches the model that clothing looks the same at different angles and positions.
   - These transforms move pixels — bounding boxes must move with them.
4. **Colour transforms**:
   - ColorJitter: randomly changes brightness, contrast, saturation, and hue. Teaches the model that the same garment can appear under different lighting.
   - Grayscale (small probability): occasionally converts to greyscale. Teaches some robustness to colour.
   - GaussNoise (heavy level only): adds random pixel noise. Teaches robustness to camera noise.
   - CoarseDropout (heavy level only): randomly blacks out small rectangular patches. Forces the model to not rely on any single region.
5. **Normalize** (line 47): subtracts the ImageNet mean and divides by ImageNet standard deviation per channel. Converts pixel values from the 0–255 range to approximately –2 to +2. This is standard practice — neural networks train more stably when inputs are centred around zero.
   - Mean: `[0.485, 0.456, 0.406]` (one value per colour channel: red, green, blue)
   - Standard deviation: `[0.229, 0.224, 0.225]`
6. **ToTensorV2**: converts the numpy array (height × width × channels) to a PyTorch tensor (channels × height × width). PyTorch expects channels first.

### Collate Function

`collate_fn` (`src/custom_model/dataset.py` line 286): called by the DataLoader after it has loaded a full batch. Each image may have a different number of bounding boxes. This function stacks all images into one tensor `(B, 3, 640, 640)` and flattens all boxes into one `(N, 6)` tensor, prepending a batch index to each box row so the loss knows which image each box belongs to.

---

## Step 2 — Forward Pass

Code: `train_one_epoch()` line 157 in `train_custom.py`. Model code: `src/custom_model/model.py`.

```python
preds = model(images)
```

The batch of images travels through three sections of the model in sequence.

### Backbone (`FashionBackbone`, `src/custom_model/model.py` line 126)

The backbone is a convolutional neural network that extracts visual features from the image at progressively smaller spatial resolutions.

**Spatial resolution**: how many grid positions the feature map has. A 640×640 input processed through a stride-2 convolution becomes 320×320. Each position in the output "sees" a larger area of the original image.

**Feature map**: the output of a layer. At early layers, each position encodes simple things like edges and colour gradients. At later layers, each position encodes more abstract things like "there is a collar here" or "this is a trouser leg shape".

**Stride**: how many pixels the convolution kernel moves per step. Stride 2 halves the spatial size.

**Channels**: the depth of a feature map. Each channel is one learned filter's response across the whole spatial grid. More channels = more different features captured.

Steps:

1. **Stem** (line 143): two convolutions. Reduces 640×640 → 320×320. Channels: 3 (RGB) → 32 → 64.
2. **Stage 1** (line 149): reduces 320×320 → 160×160. Applies a CSP block.
3. **Stage 2** (line 155): reduces 160×160 → 80×80. Output is called **P3** (pyramid level 3). This is the highest-resolution output — captures small details, good for detecting small objects.
4. **Stage 3** (line 161): reduces 80×80 → 40×40. Output is called **P4**. Medium scale.
5. **Stage 4** (line 167): reduces 40×40 → 20×20. Output is called **P5**. Lowest resolution but richest semantic features — deep in the network, sees a large area of the image. Good for detecting large objects.

**CSP Block** (Cross Stage Partial, `src/custom_model/model.py` line 102): splits the input channels in half, sends one half through a stack of residual blocks, then concatenates both halves and fuses them. This reduces computation while preserving representational power.

**Residual Block** (`src/custom_model/model.py` line 77): two convolutions with a skip connection — the input is added directly to the output: `output = conv2(conv1(x)) + x`. If the network wants to not change the input, it only needs to learn to output zero from the convolutions. This makes it easier for gradients to flow backwards through many layers during training (the gradient can travel through the skip connection directly), preventing the **vanishing gradient** problem where gradients shrink to near zero and weights stop updating in early layers.

The backbone returns three tensors: P3, P4, P5.

---

### Neck — Feature Pyramid Network (`FashionNeck`, `src/custom_model/model.py` line 195)

Problem: P3 has high spatial resolution (good for small objects) but shallow features (limited semantic meaning). P5 has rich semantic features but coarse resolution (bad for precise localisation). The neck fuses them so all three scales benefit from both.

**Top-down pass** — flows semantic information from deep to shallow:
- P5 → reduce channels → upsample to 40×40 → concatenate with P4 → CSP block → fused P4
- fused P4 → reduce channels → upsample to 80×80 → concatenate with P3 → CSP block → fused P3

**Upsample**: increase spatial resolution by repeating values (nearest-neighbour interpolation). The opposite of stride-2 convolution.

**Bottom-up pass** — refines after top-down:
- fused P3 → stride-2 conv (40×40) → concatenate with fused P4 → CSP → refined P4
- refined P4 → stride-2 conv (20×20) → concatenate with P5 lateral → CSP → refined P5

After both passes, all three feature maps carry both fine spatial detail and rich semantic meaning.

Code: `src/custom_model/model.py` line 224 (`FashionNeck.forward`).

---

### Detection Heads (`DetectionHead`, `src/custom_model/model.py` line 244)

Three separate heads, one per scale. Each operates independently on its feature map.

Each head applies two more convolutions then a final 1×1 convolution that maps each spatial position to `5 + num_classes` output values:

- **cx offset**: how far the predicted box centre is from the left edge of this grid cell (0 to 1)
- **cy offset**: how far from the top edge (0 to 1)
- **w**: predicted box width in grid units
- **h**: predicted box height in grid units
- **objectness**: is there an object centred in this cell?
- **class scores**: one score per clothing category

These are raw **logits** — unbounded numbers that have not yet been converted to probabilities. The sigmoid function converts them to probabilities (0 to 1) but this happens inside the loss function, not here.

**Logit**: the raw output of a linear layer before any activation function. Can be any positive or negative number. A logit of 0 corresponds to probability 0.5 after sigmoid.

Outputs:
- `head_p3(fused_p3)` → shape `(B, 5+NC, 80, 80)` — 6400 predictions per image
- `head_p4(fused_p4)` → shape `(B, 5+NC, 40, 40)` — 1600 predictions per image
- `head_p5(fused_p5)` → shape `(B, 5+NC, 20, 20)` — 400 predictions per image

Total: 8400 spatial positions per image, each predicting one potential object. The vast majority (>99%) of these will be background — no clothing at that location.

---

## Step 3 — Loss Computation

```python
loss, components = criterion(preds, targets)
```

Code: `FashionNetLoss.forward()` in `src/custom_model/loss.py` line 203.

The loss tells the model how wrong it was. It is a single number. Larger = worse. After this step, `.backward()` uses this number to compute gradients.

### build_targets() — Mapping Ground Truth onto the Grid

Code: `src/custom_model/loss.py` line 95.

Before any loss can be computed, the ground-truth boxes (which are in normalised 0–1 coordinates relative to the whole image) must be mapped onto the prediction grid of each scale.

For each scale (80×80, 40×40, 20×20):

1. Multiply normalised coordinates by grid size: `cx_grid = cx × gs` (e.g., `0.45 × 80 = 36.0`)
2. The owning grid cell is the integer part: `gi = int(36.0) = 36`, `gj = int(cy_grid)`
3. Four arrays are created, all zeros initially, all of shape matching the grid:
   - `obj_mask`: 1 where there is an object, 0 everywhere else
   - `noobj_mask`: 1 everywhere except where there is an object (inverse of obj_mask)
   - `target_box`: the target box stored as an offset from the cell corner: `[cx - gi, cy - gj, w_grid, h_grid]`
   - `target_cls`: a vector of zeros with 0.95 at the true class index (not 1.0 — see label smoothing below)

**Why offset from cell corner**: the model predicts where within its assigned cell the box centre is (0 to 1). A value of 0.5 means the centre of the cell. This is decoded back to absolute coordinates during inference.

**Label smoothing**: instead of setting the target class to exactly 1.0, it is set to 0.95. This prevents the model from becoming overconfident. A model that is forced to reach exactly 1.0 pushes its logits toward infinity, which is numerically unstable and leads to overfitting. With 0.95, there is always some gradient even for correct predictions.

**multi_cell** (`src/custom_model/loss.py` line 147): if the box centre falls within 0.5 grid units of a cell boundary, the adjacent cell also gets assigned this target. With ~5 ground-truth boxes per image and 8400 cells, only 0.06% of cells are positive. Multi-cell increases this to roughly 0.1–0.15%, providing 2–3× more learning signal per batch.

---

### Box Loss — Where is the object?

Code: `src/custom_model/loss.py` line 241.

Only computed at cells where `obj_mask = 1` (where there is actually an object). Background cells are ignored.

Extracts the predicted `[cx_offset, cy_offset, w, h]` at positive cells and compares to the target box using **CIoU** (Complete Intersection over Union).

**IoU (Intersection over Union)**: the ratio of the overlap area between two boxes divided by their combined area. Ranges 0 (no overlap) to 1 (perfect overlap). A standard measure of detection quality.

**CIoU** adds two penalty terms on top of plain IoU:
- Centre distance penalty: boxes whose centres are far apart are penalised even if they do not overlap at all
- Aspect ratio penalty: penalises predictions that have the wrong width-to-height ratio

```
CIoU = IoU - (centre_distance² / enclosing_diagonal²) - aspect_ratio_penalty
```

Loss is `(1 - CIoU).mean()` — so perfect predictions give 0 loss.

**Why CIoU and not plain IoU**: plain IoU has zero gradient when boxes do not overlap (common at the start of training when all predictions are wrong). CIoU always has a gradient signal, making it much better for training from scratch.

Code: `bbox_iou()` in `src/custom_model/loss.py` line 28.

---

### Class Loss — What category is it?

Code: `src/custom_model/loss.py` line 253.

Also only computed at positive cells.

Uses **Binary Cross-Entropy (BCE)** independently for each class, not a single softmax over all classes.

**Binary Cross-Entropy**: for each class independently, asks "is this class present: yes or no?" Each class is treated as a separate binary prediction. This is called **multi-label classification** — a single cell can predict multiple classes simultaneously (for example, if an image shows a complete outfit with trousers and a top both visible in that cell's area).

**Softmax** (not used here) would force the model to pick exactly one class with probabilities summing to 1. BCE does not have this constraint.

---

### Objectness Loss — Is there anything here?

Code: `src/custom_model/loss.py` line 261.

Computed at ALL 8400 cells, not just positives. This is where the major class imbalance problem lives.

Uses **Focal BCE** (`src/custom_model/loss.py` line 78):

```
standard_bce = -[target × log(p) + (1-target) × log(1-p)]
focal_weight = (1 - probability_of_correct_class)^gamma
focal_bce    = alpha × focal_weight × standard_bce
```

`gamma = 1.5`, `alpha = 0.25`.

**The problem it solves**: with ~8400 cells and ~5 objects per image, 99.94% of cells are background. If all cells contribute equally, the loss is dominated by the vast background. The model learns to always predict "no object" and the loss still looks low. No clothing ever gets detected.

**How focal loss solves it**: the `focal_weight` factor is small when the model is already confident about a prediction (easy examples) and large when the model is uncertain (hard examples). Background cells where the model correctly says "no object" with high confidence get near-zero weight. The hard cases — where the model is confused — get full weight. This forces the model to focus on learning difficult examples rather than coasting on easy background.

Divided by batch size B (line 261): this normalises the gradient magnitude. Without this, the ~8400-cell objectness gradient would overwhelm the ~5-cell box and class gradients.

---

### Final Sum

```python
total = 5.0 × loss_box + 1.0 × loss_obj + 0.5 × loss_cls
```

The three component losses are combined with fixed weights. The weights were chosen to balance gradient magnitudes:

- **5.0 × box**: box regression is the hardest task to learn from scratch. Without high weight, the model prioritises classification and objectness, and bounding box quality stays poor. The original default was 0.05 — 100× lower — and was a major bug that prevented any learning.
- **1.0 × obj**: objectness must be learned strongly since it gates whether box and class gradients are ever applied.
- **0.5 × cls**: classification is easier once the model has learned where objects are. Lower weight prevents it from dominating early training.

The function also returns a dictionary of the three individual component losses. These are printed during training and logged to `history.json` — critical for diagnosing training problems (e.g., if `obj` is not decreasing, the model is not finding objects at all).

---

## Step 4 — Backward Pass

```python
loss.backward()
```

Code: `scripts/training/train_custom.py` line 159.

PyTorch automatically tracked every mathematical operation performed during the forward pass in a **computation graph** — a record of how the loss value was computed from the model weights.

`loss.backward()` traverses this graph in reverse (from loss back to weights), applying the **chain rule of calculus** at every operation. At each learnable parameter (every convolution weight, every BatchNorm scale and shift), it computes:

```
∂loss / ∂weight
```

This is the **gradient** of that weight — a number telling us: "if this weight increases by a small amount, the loss changes by this much." A large positive gradient means increasing the weight makes the loss worse. To reduce the loss, we should decrease this weight.

Gradients are stored in `parameter.grad` for each parameter. They **accumulate** — this is why `optimizer.zero_grad()` must be called at the start of each batch. Without it, gradients from previous batches add to the current one, producing incorrect updates.

---

## Step 5 — Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

Code: `scripts/training/train_custom.py` line 161.

Before applying the gradient update, compute the **global gradient norm** — a single number summarising the magnitude of all gradients across all parameters. If this number exceeds 10.0, scale all gradients down proportionally so the norm equals exactly 10.0.

**Why this is necessary**: early in training, with random weights and badly wrong predictions, gradients can be enormous (this is called **exploding gradients**). One bad batch could send all weights to extreme values and permanently break training. Clipping puts a hard ceiling on the update magnitude each step, making training robust to outlier batches.

The threshold of 10.0 is generous — normal training gradients are far below this. It only activates on genuinely abnormal batches.

---

## Step 6 — Optimizer Step

```python
optimizer.step()
```

Code: `scripts/training/train_custom.py` line 162.

This is where weights actually change. For AdamW, the update per parameter is:

```
m = 0.9 × m + 0.1 × gradient          (momentum: smoothed gradient history)
v = 0.999 × v + 0.001 × gradient²     (velocity: smoothed squared gradient history)

m_corrected = m / (1 - 0.9^step)      (bias correction — important in early steps)
v_corrected = v / (1 - 0.999^step)

new_weight = weight
           - lr × m_corrected / (√v_corrected + 1e-8)   (gradient step)
           - lr × 0.01 × weight                          (weight decay)
```

**Momentum** (`m`): instead of following the raw gradient, follow a smoothed average of past gradients. This prevents the update from reacting wildly to one noisy batch. If the gradient has been consistently pointing in the same direction, momentum amplifies the step. If the gradient is noisy/inconsistent, momentum dampens it.

**Velocity** (`v`): tracks how large the gradients have been recently per parameter. Parameters with historically large gradients get a smaller effective learning rate (they are already updating aggressively). Parameters that rarely get gradient get a larger effective learning rate. This is the "adaptive" part of Adam.

**Bias correction**: in the early steps, `m` and `v` start at zero and are biased toward zero. The correction terms account for this.

**Weight decay** (the last line): every step, the weight is pulled slightly toward zero regardless of gradient. This acts as regularisation — it prevents any individual weight from growing very large, which tends to indicate overfitting.

---

## Step 7 — Scheduler Step

```python
batch_scheduler.step()
```

Code: `scripts/training/train_custom.py` line 168.

Updates the learning rate for the next batch. With `OneCycleLR`, this follows a precomputed curve. No computation happens in the model — this purely changes the `lr` value inside the optimizer.

The learning rate is the most sensitive hyperparameter in training. Too high → weights overshoot good values and oscillate or diverge. Too low → training converges very slowly or gets stuck in poor local minima. The one-cycle schedule navigates this by being aggressive in the middle of training and careful at the end.

---

## Step 8 — EMA Update

```python
ema.update(model)
```

Code: `scripts/training/train_custom.py` line 165. `ModelEMA.update()` at line 128.

After every optimizer step, the EMA shadow copy is updated:

```python
decay = min(0.9999, (1 + updates) / (10 + updates))   # ramps up from ~0.1 to 0.9999
for each weight:
    ema_weight = decay × ema_weight + (1 - decay) × real_weight
```

The decay warmup (`(1 + updates) / (10 + updates)`) means that in the very first steps, the EMA moves fast (low decay). As training progresses, it moves increasingly slowly, averaging over more and more past steps. This prevents the EMA from being dominated by the poor early weights.

The EMA model is never used during the forward/backward pass — only the real model trains. The EMA is used only for validation and saved to checkpoints.

---

## Step 9 — Validation

Code: `validate()` function, `scripts/training/train_custom.py` line 192.

After every epoch, the model (or EMA model) is evaluated on the validation set — images it has never trained on.

Key differences from training:

```python
model.eval()        # switch model to evaluation mode
torch.no_grad()     # disable gradient tracking
```

**`model.eval()`**: changes behaviour of two layer types:
- **BatchNorm**: during training, normalises using the current batch's mean/variance. During eval, uses the running mean/variance accumulated over all training batches. This gives consistent results regardless of batch size.
- **Dropout**: during training, randomly zeroes some activations. During eval, uses all activations (but scales them).

**`torch.no_grad()`**: tells PyTorch not to build the computation graph. Since no `.backward()` will be called, there is no need to store all the intermediate values needed for gradient computation. This saves significant memory and makes inference faster.

The same loss is computed but no backward pass is performed — gradients are not computed, weights are not updated. The validation loss is purely a measurement.

**Why validation matters**: training loss measures how well the model fits the training data. Validation loss measures how well it generalises to new data. If training loss keeps falling but validation loss starts rising, the model is **overfitting** — memorising training examples rather than learning general features. The checkpoint saved as `best.pt` is chosen by lowest validation loss, not training loss.

---

## Step 10 — Checkpointing

Code: `save_checkpoint()` at `scripts/training/train_custom.py` line 104. Called at lines 368 and 373.

Saved every epoch to `models/weights/fashionnet/`:

**`last.pt`**: always overwritten with the current epoch's state. Used to resume training if interrupted.

**`best.pt`**: only saved when validation loss is lower than any previous epoch. This is the model used for deployment and inference.

**`config.json`**: all training hyperparameters (learning rate, batch size, augmentation level, loss weights, etc.). Used by `FashionNetDetector` at inference to reconstruct the correct model architecture.

**`history.json`**: one row per epoch with all loss values and timing. Used by the evaluation scripts to produce training curve plots.

What is saved in each `.pt` file:

```python
{
    "epoch":     epoch_number,
    "model":     model.state_dict(),      # all weights
    "optimizer": optimizer.state_dict(),  # momentum/velocity state per parameter
    "scheduler": scheduler.state_dict(),  # current position in LR schedule
    "metrics":   {loss values},
    "ema":       ema.state_dict(),        # EMA weights (used at inference)
}
```

**Why save optimizer state**: if training is resumed from a checkpoint, the optimizer needs its momentum and velocity history. Without it, the optimizer restarts cold — as if it were the first step — which can destabilise training.

---

## Full Loop Summary

```
SETUP (once):
  build dataset + dataloader          src/custom_model/dataset.py line 306
  create model                        src/custom_model/model.py line 278
  create loss function                src/custom_model/loss.py line 176
  create optimizer                    train_custom.py line 284
  create scheduler                    train_custom.py line 306
  create EMA shadow copy              train_custom.py line 315

FOR each epoch:
  FOR each batch:
    1. load + augment images          dataset.py __getitem__ line 251
    2. collate into tensors           dataset.py collate_fn line 286
    3. zero_grad()                    train_custom.py line 156
    4. forward pass through model     model.py FashionNet.forward line 308
         → backbone: P3, P4, P5      model.py FashionBackbone.forward line 182
         → neck: fuse scales          model.py FashionNeck.forward line 224
         → heads: 8400 predictions   model.py DetectionHead.forward line 270
    5. build_targets()                loss.py line 95
    6. compute box + obj + cls loss   loss.py FashionNetLoss.forward line 203
    7. loss.backward()                train_custom.py line 159
    8. clip gradients (max 10.0)      train_custom.py line 161
    9. optimizer.step()               train_custom.py line 162
    10. ema.update()                  train_custom.py line 165
    11. scheduler.step()              train_custom.py line 168

  validate on val set                 train_custom.py validate() line 192
  save last.pt                        train_custom.py line 368
  if best val_loss → save best.pt     train_custom.py line 371
  append to history.json              train_custom.py line 377
```
