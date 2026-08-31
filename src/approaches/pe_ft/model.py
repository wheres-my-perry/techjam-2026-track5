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


HEAD_HIDDEN = 64   # MLP head width. Thinh's friend measured 1024 -> 64 -> 1 as optimal (2026-08-31).


HEAD2_WIDE, HEAD2_NARROW = 256, 32  # "mlp2": 1024 -> 256 -> 32 -> 1 (Thinh, 2026-08-31)


def make_head(kind: str = "linear"):
    """Classifier on top of the pooled 1024-d trunk embedding.

    "linear" is the original single projection; "mlp" is 1024 -> 64 -> 1, which Thinh's friend
    found optimal experimentally. Checkpoints record which one they used through their state_dict
    key names, so both load without a flag (see PENet.load_state_dict below).
    """
    if kind == "mlp":
        return nn.Sequential(nn.Linear(EMB, HEAD_HIDDEN), nn.GELU(), nn.Linear(HEAD_HIDDEN, 1))
    if kind == "mlp2":
        # Deeper head whose FIRST activation is a second, head-owned embedding. Thinh's idea
        # (2026-08-31): the augmentation-consistency constraint failed when it was applied to the
        # trunk's 1024-d embedding because it altered the pretrained model we are fine-tuning.
        # Applying it here instead closes the clean/augmented gap in a representation we own,
        # leaving the trunk to be trained by BCE alone. See train.py --consist-at head.
        return nn.Sequential(nn.Linear(EMB, HEAD2_WIDE), nn.GELU(),
                             nn.Linear(HEAD2_WIDE, HEAD2_NARROW), nn.GELU(),
                             nn.Linear(HEAD2_NARROW, 1))
    return nn.Linear(EMB, 1)


class PENet(nn.Module):
    """PE trunk (pooled 1024-d) + head -> 1 logit."""

    def __init__(self, pretrained: bool = True, head: str = "linear"):
        super().__init__()
        import timm
        self.trunk = timm.create_model(TIMM_NAME, pretrained=pretrained,
                                       num_classes=0, dynamic_img_size=True)
        self.head = make_head(head)

    def load_state_dict(self, sd, strict=True):
        # A linear-head checkpoint has head.weight; an MLP one has head.0.weight. Rebuild the head
        # to match the checkpoint so old and new weights both load with no caller change.
        # Which head shape does this checkpoint carry? head.weight -> linear; head.0.weight with
        # 64 rows -> mlp; with 256 rows -> mlp2. Read it off the tensor, never from a flag.
        def _kind_of(w):
            if w is None:
                return "linear"
            return "mlp2" if w.shape[0] == HEAD2_WIDE else "mlp"
        want = _kind_of(sd.get("head.0.weight"))
        have = _kind_of(self.head[0].weight if isinstance(self.head, nn.Sequential) else None)
        if want != have:
            self.head = make_head(want).to(next(self.parameters()).device)
        return super().load_state_dict(sd, strict=strict)

    def forward(self, x):
        return self.head(self.trunk(x))

    def forward_feat(self, x):
        """(embedding, logit): the pooled 1024-d trunk output and the head's logit.
        Used by the augmentation-consistency loss (Thinh, 2026-08-30)."""
        e = self.trunk(x)
        return e, self.head(e)

    def forward_feat_head(self, x):
        """(head embedding, logit) for --consist-at head (Thinh, 2026-08-31).

        The head embedding is computed from a DETACHED trunk output, so the agreement loss can
        only update the head's first layer. Without the detach its gradient would flow straight
        back into the trunk and alter the pretrained model -- which is the failure this idea
        exists to avoid, so the detach is the whole point, not an optimisation.
        """
        e = self.trunk(x)
        logit = self.head(e)
        h = self.head[1](self.head[0](e.detach()))   # Linear -> GELU, head-owned embedding
        return h, logit


def build_net(pretrained: bool = True, head: str = "linear") -> nn.Module:
    return PENet(pretrained=pretrained, head=head)


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
