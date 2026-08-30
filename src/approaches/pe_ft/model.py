"""Fine-tuned Perception Encoder (facebook/PE-Core-L14-336) for AIGC detection.

Thinh's call (2026-08-29): swap the ResNet-50 trunk for PE-Core-L14-336, a
ViT-L/14 CLIP-style encoder (316M params, 1024-d; well under the 2B limit).
Loaded via timm `vit_pe_core_large_patch14_336.fb` with dynamic_img_size so
it runs on our 112-168px native-resolution crops with interpolated position
embeddings -- NO upscaling to 336 (that would reintroduce the resampling
signature canonicalization removes).

Crop constraint: sides must be multiples of the 14px patch. The model
declares CROP_MIN/MAX/STEP and the shared crop code (src.crops) and the
vote+ wrapper honour them, so train and inference crop identically.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model import BaseModel

TIMM_NAME = "vit_pe_core_large_patch14_336.fb"
PE_MEAN = (0.5, 0.5, 0.5)
PE_STD = (0.5, 0.5, 0.5)
EMB = 1024


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PENet(nn.Module):
    """PE trunk (pooled 1024-d) + linear head -> 1 logit."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm
        self.trunk = timm.create_model(TIMM_NAME, pretrained=pretrained,
                                       num_classes=0, dynamic_img_size=True)
        self.head = nn.Linear(EMB, 1)

    def forward(self, x):
        return self.head(self.trunk(x))

    def forward_feat(self, x):
        """(embedding, logit): the pooled 1024-d trunk output and the head's logit.
        Used by the augmentation-consistency loss (Thinh, 2026-08-30)."""
        e = self.trunk(x)
        return e, self.head(e)


def build_net(pretrained: bool = True) -> nn.Module:
    return PENet(pretrained=pretrained)


def to_tensor(images) -> torch.Tensor:
    arrs = [np.asarray(im, dtype=np.float32) / 255.0 for im in images]
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
    mean = torch.tensor(PE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(PE_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class PEFTModel(BaseModel):
    name = "pe_ft"
    CROP_MIN, CROP_MAX, CROP_STEP = 112, 168, 14
    PIXEL_BUDGET = 2_000_000  # ViT-L: ~70 crops of 168px per forward

    def __init__(self, weights_path: str = "outputs/pe_ft/baseline.pt"):
        self.device = pick_device()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.net = build_net(pretrained=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    @torch.no_grad()
    def predict(self, images):
        from src.crops import snap
        scores = np.zeros(len(images), dtype=np.float32)
        by_size: dict[tuple, list[int]] = {}
        for i, im in enumerate(images):
            by_size.setdefault(im.size, []).append(i)
        for (w, h), idxs in by_size.items():
            chunk = max(1, self.PIXEL_BUDGET // max(1, w * h))
            for j in range(0, len(idxs), chunk):
                part = idxs[j:j + chunk]
                ims = [images[i] for i in part]
                # a bare (un-voted) call may hand us a non-multiple-of-14 side:
                # centre-crop down to the nearest multiple, never resize
                cw, ch = snap(w, self.CROP_STEP), snap(h, self.CROP_STEP)
                if (cw, ch) != (w, h):
                    x0, y0 = (w - cw) // 2, (h - ch) // 2
                    ims = [im.crop((x0, y0, x0 + cw, y0 + ch)) for im in ims]
                x = to_tensor(ims).to(self.device)
                with torch.autocast(self.device, dtype=torch.bfloat16,
                                    enabled=self.device == "cuda"):
                    logits = self.net(x).squeeze(1)
                scores[part] = torch.sigmoid(logits.float()).cpu().numpy()
        return scores
