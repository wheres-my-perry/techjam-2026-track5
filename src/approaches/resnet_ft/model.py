"""Fine-tuned ResNet-50 for AIGC detection.

Middle ground between the scratch CNN and frozen CLIP: pretrained edge/texture
filters (artifact-friendly, unlike CLIP's caption-aligned semantics) + full
fine-tuning so they specialize to generator fingerprints. ~23.5M params.
ResNet is fully convolutional up to its adaptive average pool, so any input
size >= ~64px works (same size-agnostic property as our scratch CNN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model import BaseModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_net(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def to_tensor(images) -> torch.Tensor:
    arrs = [np.asarray(im, dtype=np.float32) / 255.0 for im in images]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class ResNetFTModel(BaseModel):
    name = "resnet_ft"
    PIXEL_BUDGET = 4_000_000  # deeper net than SimpleCNN -> smaller budget

    def __init__(self, weights_path: str = "outputs/resnet_ft/baseline.pt"):
        self.device = pick_device()
        ckpt = torch.load(weights_path, map_location="cpu")
        self.net = build_net(pretrained=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    @torch.no_grad()
    def predict(self, images):
        scores = np.zeros(len(images), dtype=np.float32)
        by_size: dict[tuple, list[int]] = {}
        for i, im in enumerate(images):
            by_size.setdefault(im.size, []).append(i)
        for (w, h), idxs in by_size.items():
            chunk = max(1, self.PIXEL_BUDGET // max(1, w * h))
            for j in range(0, len(idxs), chunk):
                part = idxs[j:j + chunk]
                x = to_tensor([images[i] for i in part]).to(self.device)
                logits = self.net(x).squeeze(1)
                scores[part] = torch.sigmoid(logits).float().cpu().numpy()
        return scores
