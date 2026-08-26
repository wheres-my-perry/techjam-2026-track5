# Approaches

One folder per approach; approaches never import each other. The shared contract is:

1. **Folder**: `src/approaches/<name>/` with `model.py` (+ `train.py` and anything else you need).
2. **Model class**: implements `predict(images: list[PIL.Image]) -> np.ndarray of P(AI) in [0,1]`
   (subclass `src.model.BaseModel`); constructor takes a weights path.
3. **Register**: add one line to `_APPROACHES` in `src/model.py`.
4. **Weights & results**: write to `outputs/<name>/...` (weights are gitignored, eval results are committed).
5. **Train CLI**: `python -m src.approaches.<name>.train --train <manifest> --val <manifest> ...`
   Use `src.transforms.random_train_transform` for robustness augmentation (`--augment`).
6. **Evaluate** (nothing approach-specific needed):
   `python -m src.evaluate --manifest <test.csv> --model <name>:outputs/<name>/xxx.pt --out outputs/<name>/eval_xxx`

Everything in `src/` root (data, transforms, metrics, evaluate, predict) is shared infrastructure —
fix bugs there for everyone, don't fork it into your approach folder.

Current approaches: `cnn` (simple GAP CNN baseline; CIFAKE results in PROGRESS.md 2026-08-26).
