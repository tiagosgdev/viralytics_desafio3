"""
src/custom_model/loss.py
─────────────────────────
Detection loss for FashionNet.

Three components (standard in detection literature):
  1. Box loss      — CIoU loss on predicted vs ground-truth boxes
  2. Objectness    — Binary cross-entropy (focal-weighted to handle
                     the massive class imbalance between background
                     cells and object cells)
  3. Class loss    — Binary cross-entropy per class (multi-label,
                     since a single image can contain multiple garment types)

Total loss = λ_box * L_box + λ_obj * L_obj + λ_cls * L_cls
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# IoU utilities
# ─────────────────────────────────────────────────────────────────────────────

def bbox_iou(pred: torch.Tensor, target: torch.Tensor,
             ciou: bool = True) -> torch.Tensor:
    """
    Compute IoU (or CIoU) between predicted and target boxes.
    Both tensors: (N, 4) in (cx, cy, w, h) format.

    CIoU adds a penalty for aspect ratio difference and centre distance,
    which gives smoother gradients during training than plain IoU.
    """
    pw, ph = pred[:, 2],   pred[:, 3]
    tw, th = target[:, 2], target[:, 3]

    # Convert to corner format
    p_x1 = pred[:, 0]   - pw / 2;  p_y1 = pred[:, 1]   - ph / 2
    p_x2 = pred[:, 0]   + pw / 2;  p_y2 = pred[:, 1]   + ph / 2
    t_x1 = target[:, 0] - tw / 2;  t_y1 = target[:, 1] - th / 2
    t_x2 = target[:, 0] + tw / 2;  t_y2 = target[:, 1] + th / 2

    inter_w = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0)
    inter_h = (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
    inter   = inter_w * inter_h

    p_area = pw * ph
    t_area = tw * th
    union  = p_area + t_area - inter + 1e-7
    iou    = inter / union

    if not ciou:
        return iou

    # CIoU penalty terms
    enclose_w = (torch.max(p_x2, t_x2) - torch.min(p_x1, t_x1)).clamp(0)
    enclose_h = (torch.max(p_y2, t_y2) - torch.min(p_y1, t_y1)).clamp(0)
    c2        = enclose_w ** 2 + enclose_h ** 2 + 1e-7

    rho2 = ((pred[:, 0] - target[:, 0]) ** 2 +
            (pred[:, 1] - target[:, 1]) ** 2)

    v   = (4 / math.pi ** 2) * (torch.atan(tw / (th + 1e-7)) -
                                  torch.atan(pw / (ph + 1e-7))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return iou - (rho2 / c2 + v * alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Focal loss helper
# ─────────────────────────────────────────────────────────────────────────────

def focal_bce(pred: torch.Tensor, target: torch.Tensor,
              gamma: float = 1.5, alpha: float = 0.25) -> torch.Tensor:
    """
    Focal BCE — down-weights easy background examples.
    Critical for object detection where ~95% of grid cells are background.
    """
    bce  = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t  = torch.exp(-bce)
    loss = alpha * (1 - p_t) ** gamma * bce
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Target builder
# ─────────────────────────────────────────────────────────────────────────────

def build_targets(
    preds:       List[torch.Tensor],
    targets:     torch.Tensor,
    img_size:    int = 640,
    num_classes: int = 13,
    multi_cell:  bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Assign ground-truth boxes to grid cells across all 3 prediction scales.

    Parameters
    ----------
    preds       : list of 3 prediction tensors [(B,5+NC,80,80), (B,5+NC,40,40), (B,5+NC,20,20)]
    targets     : (N, 6) — each row: [batch_idx, class_id, cx, cy, w, h]  (coords normalised 0-1)
    multi_cell  : if True, assign each GT to the center cell + adjacent cells
                  when the center is near a boundary (within 0.5 grid units).
                  Increases positive training signal by ~2-3x.

    Returns
    -------
    Per scale: (obj_mask, noobj_mask, target_box, target_cls)
    """
    device     = preds[0].device
    strides    = [img_size // p.shape[-1] for p in preds]   # [8, 16, 32]
    grid_sizes = [p.shape[-1] for p in preds]               # [80, 40, 20]

    results = []
    B       = preds[0].shape[0]

    for stride, gs, pred in zip(strides, grid_sizes, preds):
        obj_mask    = torch.zeros(B, gs, gs,           device=device)
        noobj_mask  = torch.ones( B, gs, gs,           device=device)
        target_box  = torch.zeros(B, gs, gs, 4,        device=device)
        target_cls  = torch.zeros(B, gs, gs, num_classes, device=device)

        if targets.shape[0] > 0:
            # Scale normalised coords to this grid
            scaled = targets.clone()
            scaled[:, 2] *= gs   # cx in grid units
            scaled[:, 3] *= gs   # cy in grid units
            scaled[:, 4] *= gs   # w
            scaled[:, 5] *= gs   # h

            for t in scaled:
                bi   = int(t[0].item())
                cls  = int(t[1].item())
                cx, cy, w, h = t[2], t[3], t[4], t[5]
                gi, gj = int(cx.item()), int(cy.item())   # grid cell

                # Build list of cells to assign this GT to
                cells = [(gi, gj)]

                if multi_cell:
                    fx, fy = cx.item() - gi, cy.item() - gj
                    if fx < 0.5 and gi > 0:
                        cells.append((gi - 1, gj))
                    elif fx >= 0.5 and gi < gs - 1:
                        cells.append((gi + 1, gj))
                    if fy < 0.5 and gj > 0:
                        cells.append((gi, gj - 1))
                    elif fy >= 0.5 and gj < gs - 1:
                        cells.append((gi, gj + 1))

                for ci, cj in cells:
                    if 0 <= ci < gs and 0 <= cj < gs:
                        obj_mask[bi, cj, ci]       = 1
                        noobj_mask[bi, cj, ci]     = 0
                        target_box[bi, cj, ci]     = torch.stack([cx - ci, cy - cj, w, h])
                        if cls < num_classes:
                            target_cls[bi, cj, ci, cls] = 0.95

        results.append((obj_mask, noobj_mask, target_box, target_cls))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main loss class
# ─────────────────────────────────────────────────────────────────────────────

class FashionNetLoss(nn.Module):
    """
    Combined detection loss for FashionNet.

    λ_box = 5.0   — strong box regression signal (critical for mAP)
    λ_obj = 1.0   — high weight; objectness is the hardest part to learn
    λ_cls = 0.5   — moderate; classification on top of correct localisation
    """

    def __init__(self,
                 num_classes: int = 13,
                 lambda_box:  float = 5.0,
                 lambda_obj:  float = 1.0,
                 lambda_cls:  float = 0.5,
                 img_size:    int = 640,
                 multi_cell:  bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box  = lambda_box
        self.lambda_obj  = lambda_obj
        self.lambda_cls  = lambda_cls
        self.img_size    = img_size
        self.multi_cell  = multi_cell
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(
        self,
        preds:   List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        preds   : output of FashionNet.forward()
        targets : (N, 6) [batch_idx, cls, cx, cy, w, h] normalised

        Returns
        -------
        total_loss, {'box': float, 'obj': float, 'cls': float}
        """
        device = preds[0].device
        loss_box = torch.tensor(0., device=device)
        loss_obj = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)

        scale_targets = build_targets(preds, targets,
                                      self.img_size, self.num_classes,
                                      self.multi_cell)

        for pred, (obj_mask, noobj_mask, tgt_box, tgt_cls) in zip(preds, scale_targets):
            B, C, gs, _ = pred.shape

            # Reshape: (B, 5+NC, gs, gs) → (B, gs, gs, 5+NC)
            p = pred.permute(0, 2, 3, 1)

            p_xy  = torch.sigmoid(p[..., :2])    # predicted offset within cell
            p_wh  = p[..., 2:4]                   # log-space width/height
            p_obj = p[..., 4]                      # objectness logit
            p_cls = p[..., 5:]                     # class logits

            obj_mask_bool = obj_mask.bool()

            # ── Box loss (only on positive cells) ─────────────────────
            if obj_mask_bool.any():
                pred_boxes = torch.cat([p_xy[obj_mask_bool],
                                        p_wh[obj_mask_bool]], dim=-1)
                tgt_boxes  = tgt_box[obj_mask_bool]
                iou        = bbox_iou(pred_boxes, tgt_boxes, ciou=True)
                loss_box   = loss_box + (1 - iou).mean()

                # IoU-aware objectness: use CIoU as soft target instead of 1.0
                obj_mask[obj_mask_bool] = iou.detach().clamp(0)

                # ── Class loss (only on positives) ───────────────────
                loss_cls = loss_cls + self.bce(
                    p_cls[obj_mask_bool], tgt_cls[obj_mask_bool]
                )

            # ── Objectness loss (all cells, focal-weighted) ───────────
            loss_obj = loss_obj + focal_bce(p_obj, obj_mask)

        total = (self.lambda_box * loss_box
                 + self.lambda_obj * loss_obj
                 + self.lambda_cls * loss_cls)

        return total, {
            'box': loss_box.item(),
            'obj': loss_obj.item(),
            'cls': loss_cls.item(),
        }