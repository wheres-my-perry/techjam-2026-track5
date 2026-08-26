# Ideas

Candidate approaches worth building on. Not commitments — see DECISIONS.md for what we actually chose.

## Patch-level scoring for partially-fake (tampered) images — Thinh, 2026-08-26

**Observation.** A tampered image is mostly real with one fake sub-region. A classifier that pools features over the whole image averages the small fake region against a large real background — the local signal gets diluted and the image scores "real."

**Idea.** Score sub-areas of the image independently, then aggregate with an any-patch rule: *an image is fake if at least one sub-region looks fake* (multiple-instance-learning assumption). With a convolutional backbone this is nearly free — the spatial feature map before global pooling is already a grid of local descriptors, so a 1×1-conv head turns it into a grid of per-patch fake-logits in a single forward pass (no explicit sliding window needed).

**Why it pays off here.**
1. Catches locally-tampered images that global pooling misses (SID_Set `tampered` class becomes usable as a positive class instead of being excluded).
2. **Free supervision:** SID_Set ships pixel masks for tampered regions → patch labels come from downsampling the mask onto the feature grid. No annotation work.
3. **Crop robustness for free:** center-crop 80% keeps most patches; whichever fake patch survives still fires. Global features shift under cropping; patch votes largely don't.
4. **Explainability for the demo:** the patch-logit grid upsampled over the image = a "which region is fake" heatmap — directly feeds judging criteria (explainability is explicitly in scope).

**Risk / mitigation.** Hard max over patches amplifies single-patch noise → more false positives on real images. Use top-k mean (e.g. mean of top 5% patch scores) or a learned pooling instead of pure max; calibrate on val.

**Fit into plan.** v1 baseline stays global (simple, fast). Patch head is the v2 upgrade — same backbone, add the 1×1-conv patch head + top-k pooling, train with image labels (+ mask supervision where available).


## Variable input sizes: crop, don't resize — 2026-08-26

CNN fixed-input is an artifact of the final FC layer; with Global Average Pooling the net
accepts any resolution. Resizing is a low-pass filter that erases exactly the high-frequency
generator artifacts we detect, so: train on random fixed-size crops at native resolution;
infer on the full image (GAP) or on multiple crops with score aggregation — which is the same
mechanism as the patch-scoring idea above. Caveat: crops lose global semantic context
("six fingers"-type tells) → possible v2 hybrid: native-res patch branch + resized global branch.
`src/cnn.py` implements the GAP design.
