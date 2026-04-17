from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.custom_model.model import FashionNet, TinyFashionNet
from src.custom_model.postprocess import postprocess
from src.detection.detector import (
    BaseDetector,
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    CATEGORY_NAMES_11,
    Detection,
    DetectionResult,
)


class FashionNetDetector(BaseDetector):
    def __init__(
        self,
        weights: str,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        device: str = "",
    ):
        self.weights = str(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = self._pick_device(device)
        self.model, self.num_classes, self.grayscale = self._load_model(Path(self.weights))
        self._names = CATEGORY_NAMES_11 if self.num_classes == 11 else CATEGORY_NAMES

    def detect(self, frame: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = self._prepare_input(frame)

        with torch.no_grad():
            preds = self.model(tensor)
            dets = postprocess(
                preds,
                img_size=self.imgsz,
                conf_thresh=self.conf_thres,
                iou_thresh=self.iou_thres,
                num_classes=self.num_classes,
                max_det=200,
            )[0]

        detections: list[Detection] = []
        for det in dets.cpu().tolist():
            cx, cy, w, h, conf, class_id = det
            x1 = (cx - w / 2) * self.imgsz
            y1 = (cy - h / 2) * self.imgsz
            x2 = (cx + w / 2) * self.imgsz
            y2 = (cy + h / 2) * self.imgsz

            x1 = int(max(0, min(frame.shape[1], (x1 - pad_x) / scale)))
            y1 = int(max(0, min(frame.shape[0], (y1 - pad_y) / scale)))
            x2 = int(max(0, min(frame.shape[1], (x2 - pad_x) / scale)))
            y2 = int(max(0, min(frame.shape[0], (y2 - pad_y) / scale)))
            class_id = int(class_id)

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._names.get(class_id, f"class_{class_id}"),
                    confidence=float(conf),
                    bbox=[x1, y1, x2, y2],
                    color=CATEGORY_COLORS.get(class_id, (0, 255, 0)),
                )
            )

        inference_ms = (time.perf_counter() - t0) * 1000
        return DetectionResult(
            detections=detections,
            frame_shape=frame.shape,
            inference_ms=inference_ms,
            timestamp=time.time(),
        )

    def _prepare_input(self, frame: np.ndarray) -> tuple[torch.Tensor, float, int, int]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(self.imgsz / max(height, 1), self.imgsz / max(width, 1))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        if self.grayscale:
            gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        return tensor, scale, pad_x, pad_y

    def _load_model(self, weights_path: Path) -> tuple[torch.nn.Module, int, bool]:
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        config_path = weights_path.parent / "config.json"

        num_classes = 13
        is_fast = False
        grayscale = False
        dropout = 0.0
        scale = "s"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            num_classes = config.get("num_classes_resolved", num_classes)
            is_fast = config.get("fast", False)
            grayscale = config.get("grayscale", False)
            dropout = config.get("dropout", 0.0)
            scale = config.get("model_scale", "s")
        else:
            head_weight = ckpt.get("model", {}).get("head_p3.pred.weight")
            if head_weight is not None:
                num_classes = int(head_weight.shape[0] - 5)

        if is_fast:
            model = TinyFashionNet(num_classes=num_classes)
        else:
            model = FashionNet(num_classes=num_classes, dropout=dropout, scale=scale)

        if "ema" in ckpt and ckpt["ema"]:
            model.load_state_dict(ckpt["ema"])
        else:
            model.load_state_dict(ckpt["model"])

        model.to(self.device).eval()
        return model, num_classes, grayscale

    @staticmethod
    def _pick_device(requested: str) -> torch.device:
        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
