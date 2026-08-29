"""pe_seg: PE-Core-L14-336 with a PER-PATCH head (approach 09, 2026-08-29).

Problem it solves (Thinh): images where everything is real except one region
that was generated/altered. Median altered area in SID_Set is 8.8% of the
image, so a single pooled image score is dominated by the real 91%. Here each
14x14 patch token predicts "altered", supervised by the pixel mask, and the
image score is the mean of the top-k patch logits -- the pooling is learned
INSIDE the transformer with full attention context (not crop voting).

Inference: shrink long side to LONG (448), crop to a multiple of 14, one
forward pass at dynamic size -> patch logit map -> image score + heat-map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.model import BaseModel
from src.approaches.pe_ft.model import TIMM_NAME, EMB, pick_device, to_tensor

PATCH = 14
LONG = 448          # long side at inference/training (32 x ~21 tokens)
TRAIN_CROP = 294    # 21x21 tokens; the largest square that fits every SID image after --long 448 (1024x683 -> 448x299). One crop size for BOTH classes
TOPK_FRAC = 0.05    # image logit = mean of top 5% patch logits (min 4)


class PESegNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm
        self.trunk = timm.create_model(TIMM_NAME, pretrained=pretrained, num_classes=0,
                                       dynamic_img_size=True)
        self.head = nn.Sequential(nn.LayerNorm(EMB), nn.Linear(EMB, 1))

    def patch_logits(self, x):
        """(B,3,H,W) -> (B, H/14, W/14) patch logits."""
        f = self.trunk.forward_features(x)                       # (B, 1+N, D)
        f = f[:, self.trunk.num_prefix_tokens:, :]
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        return self.head(f).squeeze(-1).view(-1, h, w)

    @staticmethod
    def pool(pl):
        """(B,h,w) patch logits -> (B,) image logit: mean of top-k."""
        flat = pl.flatten(1)
        k = max(4, int(round(flat.shape[1] * TOPK_FRAC)))
        return flat.topk(min(k, flat.shape[1]), dim=1).values.mean(1)

    def forward(self, x):
        pl = self.patch_logits(x)
        return self.pool(pl), pl


def prep(im: Image.Image, long: int = LONG) -> Image.Image:
    """Shrink so the long side == long (never upscale), then crop down to a
    multiple of 14 (centre). Same in training and inference."""
    w, h = im.size
    if max(w, h) > long:
        sc = long / max(w, h)
        im = im.resize((max(PATCH, round(w * sc)), max(PATCH, round(h * sc))), Image.LANCZOS)
        w, h = im.size
    cw, ch = max(PATCH, (w // PATCH) * PATCH), max(PATCH, (h // PATCH) * PATCH)
    if (cw, ch) != (w, h):
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
    return im


class PESegModel(BaseModel):
    name = "pe_seg"

    def __init__(self, weights_path: str = "outputs/pe_seg/baseline.pt"):
        self.device = pick_device()
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.net = PESegNet(pretrained=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()

    @torch.no_grad()
    def maps(self, images):
        """Per-image (image_prob, patch_prob_map[h,w]) at LONG scale."""
        out = []
        for im in images:
            x = to_tensor([prep(im.convert("RGB"))]).to(self.device)
            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                logit, pl = self.net(x)
            out.append((float(torch.sigmoid(logit.float())[0]), torch.sigmoid(pl.float())[0].cpu().numpy()))
        return out

    def predict(self, images):
        return np.array([p for p, _ in self.maps(images)], dtype=np.float32)
