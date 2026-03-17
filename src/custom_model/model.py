"""
src/custom_model/model.py
─────────────────────────
FashionNet — custom CNN detector built from scratch in pure PyTorch.
No Ultralytics, no pretrained weights, no borrowed architectures.

Architecture overview:
  Input (3 × 640 × 640)
      │
  [Backbone]  — custom CNN with residual blocks, 4 downsampling stages
      │  P3 (128ch, 80×80)
      │  P4 (256ch, 40×40)
      │  P5 (512ch, 20×20)
  [Neck]      — simple Feature Pyramid: upsample + concat + fuse
      │  fused_p3, fused_p4, fused_p5
  [Head]      — per-scale prediction: objectness + class + bbox
      │
  Output: list of raw prediction tensors (decoded in loss / postprocess)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


NUM_CLASSES = 13   # DeepFashion2 categories


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

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
    Residual block: two 3×3 convs + skip connection.
    Skip uses 1×1 conv projection when channel count changes.
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


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────

class FashionBackbone(nn.Module):
    """
    4-stage CNN backbone outputting multi-scale feature maps.

    Input:  (B, 3, 640, 640)
    Output: P3 (B, 128, 80, 80)   ← small objects / accessories
            P4 (B, 256, 40, 40)   ← medium clothing items
            P5 (B, 512, 20, 20)   ← large garments, full outfits

    ~2.4M parameters total.
    """
    def __init__(self):
        super().__init__()

        # Stem: 640 → 320
        self.stem = nn.Sequential(
            ConvBnRelu(3,  32, k=3, s=2, p=1),
            ConvBnRelu(32, 64, k=3, s=1, p=1),
        )

        # Stage 1: 320 → 160  (64 → 64)
        self.stage1 = nn.Sequential(
            ConvBnRelu(64, 64, k=3, s=2, p=1),
            CSPBlock(64, 64, n=1),
        )

        # Stage 2: 160 → 80   (64 → 128)   → P3
        self.stage2 = nn.Sequential(
            ConvBnRelu(64, 128, k=3, s=2, p=1),
            CSPBlock(128, 128, n=2),
        )

        # Stage 3: 80 → 40    (128 → 256)  → P4
        self.stage3 = nn.Sequential(
            ConvBnRelu(128, 256, k=3, s=2, p=1),
            CSPBlock(256, 256, n=3),
        )

        # Stage 4: 40 → 20    (256 → 512)  → P5
        self.stage4 = nn.Sequential(
            ConvBnRelu(256, 512, k=3, s=2, p=1),
            CSPBlock(512, 512, n=2),
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
        p3 = self.stage2(x)   # 80×80
        p4 = self.stage3(p3)  # 40×40
        p5 = self.stage4(p4)  # 20×20
        return p3, p4, p5


# ─────────────────────────────────────────────────────────────────────────────
# Neck (Feature Pyramid Network)
# ─────────────────────────────────────────────────────────────────────────────

class FashionNeck(nn.Module):
    """
    Top-down FPN neck: fuses deep semantic features (P5) with
    shallower spatial features (P4, P3) via upsampling + concatenation.

    This gives the detection head rich features at every scale without
    needing separate deep networks per scale.
    """
    def __init__(self):
        super().__init__()

        # P5 → fuse with P4
        self.lat_p5  = ConvBnRelu(512, 256, k=1, s=1, p=0)
        self.fuse_p4 = CSPBlock(256 + 256, 256, n=1)

        # P4 (fused) → fuse with P3
        self.lat_p4  = ConvBnRelu(256, 128, k=1, s=1, p=0)
        self.fuse_p3 = CSPBlock(128 + 128, 128, n=1)

        # Bottom-up path: refine P4 and P5 after top-down pass
        self.down_p3 = ConvBnRelu(128, 128, k=3, s=2, p=1)
        self.ref_p4  = CSPBlock(128 + 256, 256, n=1)

        self.down_p4 = ConvBnRelu(256, 256, k=3, s=2, p=1)
        self.ref_p5  = CSPBlock(256 + 256, 512, n=1)

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Top-down
        p5_lat  = self.lat_p5(p5)                                        # 20×20, 256ch
        p4_td   = self.fuse_p4(torch.cat([F.interpolate(p5_lat, scale_factor=2, mode='nearest'), p4], 1))  # 40×40, 256ch
        p3_td   = self.fuse_p3(torch.cat([F.interpolate(self.lat_p4(p4_td), scale_factor=2, mode='nearest'), p3], 1))  # 80×80, 128ch

        # Bottom-up (refines features after top-down enrichment)
        p4_out  = self.ref_p4(torch.cat([self.down_p3(p3_td), p4_td], 1))   # 40×40, 256ch
        p5_out  = self.ref_p5(torch.cat([self.down_p4(p4_out), p5_lat], 1)) # 20×20, 512ch

        return p3_td, p4_out, p5_out


# ─────────────────────────────────────────────────────────────────────────────
# Detection Head
# ─────────────────────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Anchor-free detection head — predicts per grid cell:
      • (x, y)      — center offset within cell (sigmoid → 0..1)
      • (w, h)      — box size as log-space offsets from anchor priors
      • objectness  — is there an object here? (sigmoid → 0..1)
      • classes     — one score per class (sigmoid, multi-label friendly)

    Total output per scale: grid_h × grid_w × (5 + NUM_CLASSES)
    """
    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        mid = in_channels // 2
        self.pre = nn.Sequential(
            ConvBnRelu(in_channels, mid, k=3, s=1, p=1),
            ConvBnRelu(mid,        mid, k=3, s=1, p=1),
        )
        # 5 = (cx, cy, w, h, obj)
        self.pred = nn.Conv2d(mid, 5 + num_classes, kernel_size=1)
        nn.init.normal_(self.pred.weight, std=0.01)
        nn.init.constant_(self.pred.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.pre(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────

class FashionNet(nn.Module):
    """
    FashionNet — complete from-scratch detector.

    Usage:
        model = FashionNet()
        preds = model(images)   # list of 3 tensors, one per scale

    The 3 output tensors have shape:
        (B, 5+NC, 80, 80)   — large objects detected at this scale are smaller
        (B, 5+NC, 40, 40)
        (B, 5+NC, 20, 20)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = FashionBackbone()
        self.neck        = FashionNeck()
        self.head_p3     = DetectionHead(128,  num_classes)
        self.head_p4     = DetectionHead(256,  num_classes)
        self.head_p5     = DetectionHead(512,  num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p3, p4, p5             = self.backbone(x)
        fp3, fp4, fp5          = self.neck(p3, p4, p5)
        return [
            self.head_p3(fp3),  # (B, 5+NC, 80, 80)
            self.head_p4(fp4),  # (B, 5+NC, 40, 40)
            self.head_p5(fp5),  # (B, 5+NC, 20, 20)
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




# ─────────────────────────────────────────────────────────────────────────────
# Tiny variant — for fast CPU testing only
# ─────────────────────────────────────────────────────────────────────────────

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
        self.b2 = ConvBnRelu(16,  32, k=3, s=2, p=1)   # /4  → P3
        self.b3 = ConvBnRelu(32,  64, k=3, s=2, p=1)   # /8  → P4
        self.b4 = ConvBnRelu(64, 128, k=3, s=2, p=1)   # /16 → P5

        # Minimal neck — single lateral connection
        self.lat = ConvBnRelu(128, 64, k=1, s=1, p=0)
        self.fuse= ConvBnRelu(64+64, 64, k=1, s=1, p=0)

        # Single-scale head (P4 only — good enough for pipeline testing)
        self.head = nn.Conv2d(64, 5 + num_classes, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p2 = self.b2(self.b1(x))          # 160×160, 32ch
        p3 = self.b3(p2)                   # 80×80,   64ch
        p4 = self.b4(p3)                   # 40×40,  128ch

        # Simple FPN: upsample P4 → fuse with P3
        lat = self.lat(p4)                 # 40×40, 64ch
        up  = F.interpolate(lat, scale_factor=2, mode='nearest')  # 80×80
        fused = self.fuse(torch.cat([up, p3], dim=1))             # 80×80, 64ch

        # Return 3 scales (head runs on fused only; other two are copies for
        # loss compatibility — they produce near-zero predictions by design)
        pred = self.head(fused)
        return [pred,
                self.head(F.avg_pool2d(fused, 2)),   # 40×40
                self.head(F.avg_pool2d(fused, 4))]   # 20×20

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("--- FashionNet ---")
    model  = FashionNet()
    dummy  = torch.zeros(1, 3, 640, 640)
    outs   = model(dummy)
    params = model.count_parameters()
    print(f"FashionNet parameter count: {params:,}")
    for i, o in enumerate(outs):
        print(f"  Scale {i+1} output shape: {tuple(o.shape)}")

    print("\n--- TinyFashionNet ---")
    tiny   = TinyFashionNet()
    outs_t = tiny(dummy)
    print(f"TinyFashionNet parameter count: {tiny.count_parameters():,}")
    for i, o in enumerate(outs_t):
        print(f"  Scale {i+1} output shape: {tuple(o.shape)}")