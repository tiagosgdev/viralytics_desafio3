"""
src/custom_model/model.py
─────────────────────────
FashionNet — custom CNN detector built from scratch in pure PyTorch.
No Ultralytics, no pretrained weights, no borrowed architectures.

Architecture overview:
  Input (3 × 640 × 640)
      │
  [Backbone]  — custom CNN with residual blocks, 4 downsampling stages
      │  P3, P4, P5 at strides 8, 16, 32
  [Neck]      — bidirectional FPN: upsample + concat + fuse
      │  fused_p3, fused_p4, fused_p5
  [Head]      — per-scale prediction: objectness + class + bbox
      │
  Output: list of raw prediction tensors (decoded in loss / postprocess)

Model scales (--model_scale flag):
  s:  ~11.7M params   (64→128→256→512,  CSP n=1,2,3,2)
  m:  ~25M  params   (96→192→384→768,  CSP n=2,3,4,3)
  l:  ~43M  params   (128→256→512→1024, CSP n=3,4,6,3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


NUM_CLASSES = 13   # DeepFashion2 categories

# ─────────────────────────────────────────────────────────────────────────────
# Scale configurations — controls model width and depth
# ────────��─────────────���──────────────────────────────────────────────────────

SCALE_CONFIGS = {
    "s": {
        "stem":       (32, 64),
        "stages":     (64, 128, 256, 512),
        "csp_depths": (1, 2, 3, 2),
        "neck_depth": 1,
    },
    "m": {
        "stem":       (48, 96),
        "stages":     (96, 192, 384, 768),
        "csp_depths": (2, 3, 4, 3),
        "neck_depth": 2,
    },
    "l": {
        "stem":       (64, 128),
        "stages":     (128, 256, 512, 1024),
        "csp_depths": (3, 4, 6, 3),
        "neck_depth": 2,
    },
}


# ───────���──────────────────────────��──────────────────────────────��───────────
# Building blocks
# ─────��───────────────────────────────────────────────��───────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → LeakyReLU — the basic unit used everywhere."""
    def __init__(self, in_ch: int, out_ch: int,
                 k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.03, eps=1e-3),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """
    Residual block: two 3x3 convs + skip connection.
    Skip uses 1x1 conv projection when channel count changes.
    These are critical when training from scratch — they prevent
    vanishing gradients in deeper stages.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.c1 = ConvBnRelu(in_ch,  out_ch, k=3, s=stride, p=1)
        self.c2 = ConvBnRelu(out_ch, out_ch, k=3, s=1,      p=1)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.c2(self.c1(x)) + self.skip(x))


class CSPBlock(nn.Module):
    """
    Cross Stage Partial block — splits channels, processes one half through
    residual blocks, then concatenates. Reduces computation while maintaining
    representational power. A key design pattern we implement from scratch.
    """
    def __init__(self, in_ch: int, out_ch: int, n: int = 1):
        super().__init__()
        mid = out_ch // 2
        self.conv1 = ConvBnRelu(in_ch, mid, k=1, s=1, p=0)   # branch 1
        self.conv2 = ConvBnRelu(in_ch, mid, k=1, s=1, p=0)   # branch 2 — bypasses blocks
        self.blocks = nn.Sequential(*[ResBlock(mid, mid) for _ in range(n)])
        self.fuse   = ConvBnRelu(mid * 2, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.blocks(self.conv1(x))   # processed branch
        b2 = self.conv2(x)                # identity branch
        return self.fuse(torch.cat([b1, b2], dim=1))


# ──────────���────────────────────────��────────────────────────────────────��────
# Backbone
# ────────────────���─────────────────────────────────��──────────────────────────

class FashionBackbone(nn.Module):
    """
    4-stage CNN backbone outputting multi-scale feature maps.

    Input:  (B, 3, 640, 640)
    Output: P3, P4, P5 at strides 8, 16, 32

    Channel widths and CSP depths are controlled by the scale config.
    """
    def __init__(self, scale: str = "s"):
        super().__init__()
        cfg = SCALE_CONFIGS[scale]
        s1_ch, s2_ch = cfg["stem"]
        c1, c2, c3, c4 = cfg["stages"]
        n1, n2, n3, n4 = cfg["csp_depths"]

        # Stem: 640 -> 320
        self.stem = nn.Sequential(
            ConvBnRelu(3,     s1_ch, k=3, s=2, p=1),
            ConvBnRelu(s1_ch, s2_ch, k=3, s=1, p=1),
        )

        # Stage 1: 320 -> 160
        self.stage1 = nn.Sequential(
            ConvBnRelu(s2_ch, c1, k=3, s=2, p=1),
            CSPBlock(c1, c1, n=n1),
        )

        # Stage 2: 160 -> 80  -> P3
        self.stage2 = nn.Sequential(
            ConvBnRelu(c1, c2, k=3, s=2, p=1),
            CSPBlock(c2, c2, n=n2),
        )

        # Stage 3: 80 -> 40   -> P4
        self.stage3 = nn.Sequential(
            ConvBnRelu(c2, c3, k=3, s=2, p=1),
            CSPBlock(c3, c3, n=n3),
        )

        # Stage 4: 40 -> 20   -> P5
        self.stage4 = nn.Sequential(
            ConvBnRelu(c3, c4, k=3, s=2, p=1),
            CSPBlock(c4, c4, n=n4),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.stem(x)
        x  = self.stage1(x)
        p3 = self.stage2(x)   # 80x80
        p4 = self.stage3(p3)  # 40x40
        p5 = self.stage4(p4)  # 20x20
        return p3, p4, p5


# ───────────────────────────────────���────────────────────────────���────────────
# Neck (Feature Pyramid Network)
# ──────��──────────────────────────────────────────────────────────────────────

class FashionNeck(nn.Module):
    """
    Bidirectional FPN neck: top-down fusion followed by bottom-up refinement.

    Fuses deep semantic features (P5) with shallower spatial features (P4, P3)
    via upsampling + concatenation. Channel widths and CSP depth are controlled
    by the scale config.
    """
    def __init__(self, scale: str = "s"):
        super().__init__()
        cfg = SCALE_CONFIGS[scale]
        _, c2, c3, c4 = cfg["stages"]   # P3=c2, P4=c3, P5=c4
        nd = cfg["neck_depth"]

        # Top-down: P5 -> fuse with P4
        self.lat_p5  = ConvBnRelu(c4, c3, k=1, s=1, p=0)
        self.fuse_p4 = CSPBlock(c3 + c3, c3, n=nd)

        # Top-down: P4 (fused) -> fuse with P3
        self.lat_p4  = ConvBnRelu(c3, c2, k=1, s=1, p=0)
        self.fuse_p3 = CSPBlock(c2 + c2, c2, n=nd)

        # Bottom-up: refine P4 and P5 after top-down pass
        self.down_p3 = ConvBnRelu(c2, c2, k=3, s=2, p=1)
        self.ref_p4  = CSPBlock(c2 + c3, c3, n=nd)

        self.down_p4 = ConvBnRelu(c3, c3, k=3, s=2, p=1)
        self.ref_p5  = CSPBlock(c3 + c3, c4, n=nd)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Top-down
        p5_lat = self.lat_p5(p5)
        p4_td  = self.fuse_p4(torch.cat([F.interpolate(p5_lat, scale_factor=2, mode='nearest'), p4], 1))
        p3_td  = self.fuse_p3(torch.cat([F.interpolate(self.lat_p4(p4_td), scale_factor=2, mode='nearest'), p3], 1))

        # Bottom-up (refines features after top-down enrichment)
        p4_out = self.ref_p4(torch.cat([self.down_p3(p3_td), p4_td], 1))
        p5_out = self.ref_p5(torch.cat([self.down_p4(p4_out), p5_lat], 1))

        return p3_td, p4_out, p5_out


# ─────────────────────────────────────────────────────────────────────────────
# Detection Head
# ──────��─────────────────────���────────────────────────────���───────────────────

class DetectionHead(nn.Module):
    """
    Anchor-free detection head — predicts per grid cell:
      - (x, y)      — center offset within cell (sigmoid -> 0..1)
      - (w, h)      — box size in grid units
      - objectness  — is there an object here? (sigmoid -> 0..1)
      - classes     — one score per class (sigmoid, multi-label friendly)

    Total output per scale: grid_h x grid_w x (5 + NUM_CLASSES)
    """
    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.0):
        super().__init__()
        mid = in_channels // 2
        layers = [
            ConvBnRelu(in_channels, mid, k=3, s=1, p=1),
            ConvBnRelu(mid,        mid, k=3, s=1, p=1),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.pre = nn.Sequential(*layers)
        # 5 = (cx, cy, w, h, obj)
        self.pred = nn.Conv2d(mid, 5 + num_classes, kernel_size=1)
        nn.init.normal_(self.pred.weight, std=0.01)
        nn.init.constant_(self.pred.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.pre(x))


# ──────────────────────────────────────────��──────────────────────────────────
# Full Model
# ────────��────────────────��─────────────────────────────────��─────────────────

class FashionNet(nn.Module):
    """
    FashionNet — complete from-scratch detector.

    Usage:
        model = FashionNet()                          # scale "s" (default, ~11.7M)
        model = FashionNet(scale="m")                 # scale "m" (~25M)
        model = FashionNet(scale="l")                 # scale "l" (~43M)
        preds = model(images)   # list of 3 tensors, one per scale

    The 3 output tensors have shape:
        (B, 5+NC, 80, 80)
        (B, 5+NC, 40, 40)
        (B, 5+NC, 20, 20)
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.0,
                 scale: str = "s"):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale

        cfg = SCALE_CONFIGS[scale]
        _, c2, c3, c4 = cfg["stages"]   # P3=c2, P4=c3, P5=c4

        self.backbone = FashionBackbone(scale=scale)
        self.neck     = FashionNeck(scale=scale)
        self.head_p3  = DetectionHead(c2, num_classes, dropout=dropout)
        self.head_p4  = DetectionHead(c3, num_classes, dropout=dropout)
        self.head_p5  = DetectionHead(c4, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p3, p4, p5    = self.backbone(x)
        fp3, fp4, fp5 = self.neck(p3, p4, p5)
        return [
            self.head_p3(fp3),  # (B, 5+NC, 80, 80)
            self.head_p4(fp4),  # (B, 5+NC, 40, 40)
            self.head_p5(fp5),  # (B, 5+NC, 20, 20)
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────��──────────────────────────────────��───────────
# Tiny variant — for fast CPU testing only
# ───────���──────────────��─────────────────────────────────���────────────────────

class TinyFashionNet(nn.Module):
    """
    Stripped-down version of FashionNet for fast CPU prototyping.
    Uses ~10x fewer channels than the full model (~400K params vs 11M).
    Results will be poor — use only to verify the pipeline runs end-to-end.

    To get real results, train FashionNet (full) on a GPU.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Minimal backbone — 4 downsampling convs, no residual blocks
        self.b1 = ConvBnRelu(3,   16, k=3, s=2, p=1)   # /2
        self.b2 = ConvBnRelu(16,  32, k=3, s=2, p=1)   # /4  -> P3
        self.b3 = ConvBnRelu(32,  64, k=3, s=2, p=1)   # /8  -> P4
        self.b4 = ConvBnRelu(64, 128, k=3, s=2, p=1)   # /16 -> P5

        # Minimal neck — single lateral connection
        self.lat = ConvBnRelu(128, 64, k=1, s=1, p=0)
        self.fuse= ConvBnRelu(64+64, 64, k=1, s=1, p=0)

        # Single-scale head (P4 only — good enough for pipeline testing)
        self.head = nn.Conv2d(64, 5 + num_classes, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p2 = self.b2(self.b1(x))          # 160x160, 32ch
        p3 = self.b3(p2)                   # 80x80,   64ch
        p4 = self.b4(p3)                   # 40x40,  128ch

        # Simple FPN: upsample P4 -> fuse with P3
        lat = self.lat(p4)                 # 40x40, 64ch
        up  = F.interpolate(lat, scale_factor=2, mode='nearest')  # 80x80
        fused = self.fuse(torch.cat([up, p3], dim=1))             # 80x80, 64ch

        # Return 3 scales (head runs on fused only; other two are copies for
        # loss compatibility — they produce near-zero predictions by design)
        pred = self.head(fused)
        return [pred,
                self.head(F.avg_pool2d(fused, 2)),   # 40x40
                self.head(F.avg_pool2d(fused, 4))]   # 20x20

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    dummy = torch.zeros(1, 3, 640, 640)

    for scale in ("s", "m", "l"):
        model = FashionNet(scale=scale)
        outs  = model(dummy)
        print(f"FashionNet-{scale}: {model.count_parameters():,} params")
        for i, o in enumerate(outs):
            print(f"  Scale {i+1} output shape: {tuple(o.shape)}")
        print()

    print("--- TinyFashionNet ---")
    tiny   = TinyFashionNet()
    outs_t = tiny(dummy)
    print(f"TinyFashionNet parameter count: {tiny.count_parameters():,}")
    for i, o in enumerate(outs_t):
        print(f"  Scale {i+1} output shape: {tuple(o.shape)}")
