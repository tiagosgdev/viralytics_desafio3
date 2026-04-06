"""
src/custom_model/postprocess.py
───────────────────────────────
Post-processing pipeline for FashionNet raw output tensors.

Converts the 3 raw grid-level prediction tensors into usable detection boxes
with confidence scores and class IDs, then applies NMS.

Grid decoding is the exact inverse of build_targets() in loss.py:
  - build_targets stores offsets as (cx - cell_i, cy - cell_j, w, h) in grid units
  - channels 0-1: sigmoid → cell offset (0..1), add grid offset, divide by gs → normalised
  - channels 2-3: raw w/h in grid units (no exp), clamp ≥ 0, divide by gs → normalised
  - channel  4:   sigmoid → objectness
  - channels 5+:  sigmoid → class scores
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops


def decode_predictions(
    raw_preds: List[torch.Tensor],
    img_size: int = 640,
    conf_thresh: float = 0.25,
    num_classes: int = 13,
) -> List[torch.Tensor]:
    """
    Decode FashionNet's 3 raw output tensors into detection candidates.

    Parameters
    ----------
    raw_preds : list of 3 tensors, each (B, 5+NC, gs, gs)
    img_size  : input image size (assumes square)
    conf_thresh : minimum confidence (obj * max_cls) to keep
    num_classes : number of object classes

    Returns
    -------
    List of length B, each a (D, 6) tensor: [cx, cy, w, h, confidence, class_id]
    in normalised coordinates (0..1).
    """
    device = raw_preds[0].device
    B = raw_preds[0].shape[0]

    per_image: List[List[torch.Tensor]] = [[] for _ in range(B)]

    for pred in raw_preds:
        _, C, gs, _ = pred.shape

        # (B, C, gs, gs) → (B, gs, gs, C)
        p = pred.permute(0, 2, 3, 1)

        # Grid offsets: grid_x[j, i] = i, grid_y[j, i] = j
        gy, gx = torch.meshgrid(
            torch.arange(gs, device=device, dtype=torch.float32),
            torch.arange(gs, device=device, dtype=torch.float32),
            indexing="ij",
        )  # both (gs, gs)

        # Decode
        p_xy = torch.sigmoid(p[..., :2])          # cell offsets (0..1)
        p_wh = p[..., 2:4].clamp(min=0)           # w/h in grid units
        p_obj = torch.sigmoid(p[..., 4])           # objectness
        p_cls = torch.sigmoid(p[..., 5:])          # class scores

        for bi in range(B):
            # Combined confidence = obj * max_cls
            cls_scores, cls_ids = p_cls[bi].max(dim=-1)   # (gs, gs)
            conf = p_obj[bi] * cls_scores                 # (gs, gs)

            mask = conf >= conf_thresh
            if not mask.any():
                continue

            # Decode centres → normalised coords
            cx_norm = (p_xy[bi, ..., 0][mask] + gx[mask]) / gs
            cy_norm = (p_xy[bi, ..., 1][mask] + gy[mask]) / gs

            # Decode w/h → normalised coords
            w_norm = p_wh[bi, ..., 0][mask] / gs
            h_norm = p_wh[bi, ..., 1][mask] / gs

            dets = torch.stack([
                cx_norm,
                cy_norm,
                w_norm,
                h_norm,
                conf[mask],
                cls_ids[mask].float(),
            ], dim=1)  # (D, 6)

            per_image[bi].append(dets)

    # Concatenate all scales per image
    results: List[torch.Tensor] = []
    for bi in range(B):
        if per_image[bi]:
            results.append(torch.cat(per_image[bi], dim=0))
        else:
            results.append(torch.zeros((0, 6), device=device))

    return results


def nms(detections: torch.Tensor, iou_thresh: float = 0.45) -> torch.Tensor:
    """
    Per-class Non-Maximum Suppression.

    Parameters
    ----------
    detections : (D, 6) tensor [cx, cy, w, h, confidence, class_id]
                 in normalised coordinates
    iou_thresh : IoU threshold for suppression

    Returns
    -------
    Filtered (K, 6) tensor after NMS.
    """
    if detections.shape[0] == 0:
        return detections

    # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
    cx, cy, w, h = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    boxes_xyxy = torch.stack([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    ], dim=1)

    scores = detections[:, 4]
    class_ids = detections[:, 5]

    # Class-offset trick: shift boxes by class so classes don't suppress each other
    offsets = class_ids * 10000.0
    boxes_shifted = boxes_xyxy + offsets.unsqueeze(1)

    keep = tv_ops.nms(boxes_shifted, scores, iou_thresh)
    return detections[keep]


def postprocess(
    raw_preds: List[torch.Tensor],
    img_size: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    num_classes: int = 13,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """
    Full post-processing pipeline: decode → NMS → top-K.

    Parameters
    ----------
    raw_preds   : list of 3 raw output tensors from FashionNet
    img_size    : input image size
    conf_thresh : confidence threshold for filtering
    iou_thresh  : NMS IoU threshold
    num_classes : number of classes
    max_det     : maximum detections per image

    Returns
    -------
    List of length B, each a (K, 6) tensor [cx, cy, w, h, confidence, class_id]
    in normalised coordinates, sorted by confidence descending.
    """
    decoded = decode_predictions(raw_preds, img_size, conf_thresh, num_classes)

    results: List[torch.Tensor] = []
    for dets in decoded:
        dets = nms(dets, iou_thresh)
        # Sort by confidence descending
        if dets.shape[0] > 0:
            order = dets[:, 4].argsort(descending=True)
            dets = dets[order[:max_det]]
        results.append(dets)

    return results
