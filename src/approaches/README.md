# Approaches

Each registered model family lives under `src/approaches/<name>/`. Approach folders may reuse
shared helpers from another family when the dependency is explicit—for example, `pe_seg` reuses
`pe_ft` model utilities—but shared evaluation and data behavior belongs in `src/`.

The integration contract is:

1. Create `src/approaches/<name>/model.py`, plus `train.py` when training is required.
2. Implement `predict(images: list[PIL.Image]) -> np.ndarray` containing P(AI) values in [0, 1],
   normally by subclassing `src.model.BaseModel`.
3. Register the module, class, and default weights path in `src.model._APPROACHES`.
4. Keep generated weights and score archives under `outputs/<name>/`; they are ignored by Git.
5. Expose training as `python -m src.approaches.<name>.train ...` where practical.
6. Evaluate through the shared harness:

   ```
   python -m src.evaluate --manifest <test.csv> \
       --model <name>:outputs/<name>/<weights> --threshold <fixed-cutoff> \
       --out outputs/<name>/<evaluation>
   ```

The current registry contains `cnn`, `clip_linear`, `resnet_ft`, `pe_ft`, `pe_seg`,
`real_manifold`, `spectral`, `patch_relation`, and `stacked`. Registry membership means the
implementation is available; it does not mean the approach is currently training or recommended.
The shipped family is `pe_ft`.
