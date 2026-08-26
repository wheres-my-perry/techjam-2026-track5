"""CLIP backbone (frozen) + linear head.

Rationale: CLIP embeddings were never optimized for any single generator, so a
linear probe on them generalizes to unseen generators far better than end-to-end
CNNs (Ojha et al., "Towards Universal Fake Image Detectors", CVPR 2023).

Preprocessing note: CLIP has a fixed canonical input (resize shorter side ->
center crop 224). That resize is inherent to the backbone; robustness comes from
augmenting at the pixel level BEFORE preprocessing (train) and from the grid
being applied before preprocessing (eval). A crop-voting variant can come later.
"""

from __future__ import annotations

import numpy as np
import torch

from src.model import BaseModel

DEFAULT_BACKBONE = "ViT-L-14"
DEFAULT_PRETRAINED = "openai"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_backbone(backbone: str, pretrained: str, device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        backbone, pretrained=pretrained)
    model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def embed_images(model, preprocess, images, device, batch=64) -> np.ndarray:
    """PIL images -> L2-normalized CLIP embeddings (N, D)."""
    outs = []
    for i in range(0, len(images), batch):
        x = torch.stack([preprocess(im) for im in images[i:i + batch]]).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        outs.append(f.float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1), dtype=np.float32)


class CLIPLinearModel(BaseModel):
    name = "clip_linear"

    def __init__(self, weights_path: str = "outputs/clip_linear/baseline.pt"):
        self.device = pick_device()
        ckpt = torch.load(weights_path, map_location="cpu")
        self.backbone_name = ckpt["backbone"]
        self.pretrained = ckpt["pretrained"]
        self.model, self.preprocess = load_backbone(
            self.backbone_name, self.pretrained, self.device)
        self.w = torch.tensor(ckpt["w"])          # (D,)
        self.b = float(ckpt["b"])

    def predict(self, images):
        emb = embed_images(self.model, self.preprocess, images, self.device)
        logits = emb @ self.w.numpy() + self.b
        return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
