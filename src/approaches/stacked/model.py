"""Stacked ensemble (approach 07): a tiny classifier over member model scores.

Rationale (observation #11): our members fail in COMPLEMENTARY ways — resnet
dies to blur while clip_linear barely notices transforms; real_manifold is
inverted on GAN/VQ (itself usable signal, observation #3). A stacker over the
score vector inherits each member's strengths without retraining any of them.

Members are stored by full model-spec string (e.g. "patch_relation:path") and
loaded through src.model.load_model, so anything the registry can build —
vote+/noise+ wrappers included — can be a member.
"""

from __future__ import annotations

import pickle

import numpy as np

from src.model import BaseModel, load_model


class StackedModel(BaseModel):
    name = "stacked"

    def __init__(self, weights_path: str = "outputs/stacked/baseline.npz"):
        z = np.load(weights_path, allow_pickle=False)
        self.members = [str(m) for m in z["members"]]
        self.kind = str(z["kind"])
        if self.kind == "lr":
            self.w = z["w"]
            self.b = float(z["b"])
        else:  # gradient-boosted stacker lives in a sidecar pickle
            with open(str(weights_path) + ".pkl", "rb") as fh:
                self.clf = pickle.load(fh)
        self._models = [load_model(m) for m in self.members]

    def predict(self, images):
        S = np.stack([m.predict(images) for m in self._models], axis=1)
        if self.kind == "lr":
            logit = S @ self.w + self.b
            return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
        return self.clf.predict_proba(S)[:, 1].astype(np.float32)
