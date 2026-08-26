"""Simple size-agnostic CNN baseline (v1 experiment).

Design notes (see chat/IDEAS.md discussion on variable input sizes):
- All-convolutional + Global Average Pooling => accepts ANY input size >= 32px.
  No flatten/FC on spatial dims, so nothing forces a fixed resolution.
- Trained on (possibly cropped) fixed-size batches for efficiency; inference can
  run full-resolution or crop-vote (predict groups images by size into batches).
- ~470K params. Well under any budget; trains on CPU/MPS in minutes on CIFAKE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model import BaseModel

IMAGENET_FREE_MEAN = 0.5  # simple [-1,1] normalization; no pretrained stats needed


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w),            # 32
            _block(w, w),
            nn.MaxPool2d(2),
            _block(w, 2 * w),        # 64
            _block(2 * w, 2 * w),
            nn.MaxPool2d(2),
            _block(2 * w, 4 * w),    # 128
            _block(4 * w, 4 * w),
            nn.MaxPool2d(2),
            _block(4 * w, 8 * w),    # 256
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # <- size-agnostic: GAP, not flatten
        self.head = nn.Linear(8 * w, 1)

    def forward(self, x):  # x: (N,3,H,W), any H,W >= 32
        f = self.features(x)
        f = self.pool(f).flatten(1)
        return self.head(f).squeeze(1)  # logits


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_tensor(images) -> torch.Tensor:
    """list of same-size PIL RGB -> (N,3,H,W) float in [-1,1]."""
    arrs = [np.asarray(im, dtype=np.float32) / 255.0 for im in images]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    return (x - IMAGENET_FREE_MEAN) / 0.5


class CNNModel(BaseModel):
    """Harness adapter. Groups mixed-size inputs into same-size sub-batches."""

    name = "cnn"

    def __init__(self, weights_path: str = "outputs/cnn_baseline.pt"):
        self.device = pick_device()
        ckpt = torch.load(weights_path, map_location=self.device)
        self.net = SimpleCNN(width=ckpt.get("width", 32))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    # memory guard: cap total pixels per forward pass (8M px ≈ eight 1024x1024
    # images) so full-resolution eval doesn't blow up GPU/MPS memory.
    PIXEL_BUDGET = 8_000_000

    @torch.no_grad()
    def predict(self, images):
        scores = np.zeros(len(images), dtype=np.float32)
        by_size: dict[tuple, list[int]] = {}
        for i, im in enumerate(images):
            by_size.setdefault(im.size, []).append(i)
        for (w, h), idxs in by_size.items():
            per_img = max(1, w * h)
            chunk = max(1, self.PIXEL_BUDGET // per_img)
            for j in range(0, len(idxs), chunk):
                part = idxs[j:j + chunk]
                x = to_tensor([images[i] for i in part]).to(self.device)
                logits = self.net(x)
                scores[part] = torch.sigmoid(logits).float().cpu().numpy()
        return scores
