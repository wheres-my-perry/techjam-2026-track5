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


## Backbone / approach candidates beyond clip_linear — 2026-08-26

Ordered by alignment with the artifact-focus hypothesis (Thinh):

1. **resnet_ft / convnext_ft** — ImageNet-pretrained ResNet-50 or ConvNeXt-Tiny (~25-30M params),
   FULLY fine-tuned with contest-transform aug. Classic recipe of Wang et al. "CNN-generated images
   are surprisingly easy to spot" (CVPR'20). Pretrained edge/texture filters ≈ artifact-friendly,
   without CLIP's semantic bias. Strongest test of the artifact-focus claim. → next approach folder.
2. **dinov2_linear** — frozen DINOv2 + linear probe. Self-supervised, no text alignment → features
   keep more texture/local structure than CLIP. "CLIP but less semantic."
3. **siglip / eva-clip** — better-trained CLIP cousins; same critique as CLIP. Skip unless clip_linear
   earns its place on WildFake.
4. **Artifact front-end** — feed high-pass residual / SRM filters / DCT stats instead of RGB, so the
   model can ONLY see artifacts. Max artifact alignment; pairs with patch-scoring. Risk: JPEG/blur
   attack exactly this channel → lives or dies by augmentation.

## Why not just keep improving the scratch CNN? — 2026-08-26

Legitimate — cnn stays a live approach. But its CIFAKE 0.94 hides the known cliff: from-scratch CNN
fingerprint knowledge is generator-specific, and the literature shows such models can drop toward
chance on UNSEEN generators (contest judges use DALL·E-Advanced, which we never train on).
CIFAKE cannot measure this (single generator). Improvement path if pursued: error analysis on its
failures, width/resolution scaling on real data, artifact front-end input, patch head. Verdict by
harness on WildFake with a held-out generator — not by argument.


## Long-range inconsistency detection + score stacking — Thinh, 2026-08-27

**Observation.** Local anomalies (tiny AI-glitch details) are CNN territory. But some fakes are
exposed by RELATIONS between distant parts (two far-apart regions that are mutually illogical:
lighting/shadow disagreement, mismatched pairs, impossible geometry). Convolution's local windows
are structurally weak at this.

**Mechanisms, ranked by fit:**
1. Self-attention = the direct answer (every patch attends to every patch). CLIP's encoder is a ViT,
   so clip_linear partially covers this axis already; DINOv2 probe = same recipe, less caption-biased.
2. **Relation head over patch features (build-weekend candidate):** small 1-2 layer transformer over
   the backbone's patch-embedding grid, trained with image labels. Combined with the earlier
   patch-scoring head → one backbone, two observation types (local anomaly + cross-patch consistency).
   ~150 LOC given our infra; strong innovation-score story.
3. Explicit physics checks (lighting/shadows/reflections): too handcrafted for 72h. VLM-as-judge:
   too slow for eval, but great demo explainability garnish.

**Stacking (nothing wasted):** every approach outputs P(AI) per image → stacked ensemble = logistic
regression over the N model scores, trained on val, registered as approach `ensemble` in the registry
(drops into the harness unchanged). Pays when members fail differently — check per-generator failure
overlap in tomorrow's results. Params budget = sum of members; still far under 2B.


## Visual-inspection cue catalog — Thinh + Claude, label-blind review of WildFake samples, 2026-08-27

Blind-guess result: reals identifiable via imperfection (sensor grain, motion blur, compositional
randomness); 3/12 fakes fooled Claude at 200px (stylegan cat, both stargan dogs). Cues found, each
with mechanism:

1. **Detail-sharpness inconsistency** (stylegan): hyper-sharp focal features in airbrushed surroundings,
   violating depth-of-field physics; edge halos. → per-patch sharpness map + focus-field plausibility check.
2. **Regular-structure failure** (ddim bedrooms): irregular window mullions, merging bed rails, wobbling
   wall lines. Long-range regularity constraint → line/periodicity stats or transformer.
3. **Broken text** (ddpm, vqvae): looks-like-text-but-gibberish. → cheap OCR-confidence feature
   (afternoon build, feeds ensemble).
4. **Periphery/contact melt** (biggan): salient subject fine, supporting objects (hands, contact points)
   melted. → patch head with attention to off-subject/contact regions.
5. **Nameable-object failure** (ddpm): regions no label fits. → per-region classification entropy.
6. **Material-texture mismatch** (furry pillows). → semantic-relational; CLIP features carry implicitly.
7. **Noise-field inconsistency**: reals have uniform grain; fakes mix sterile and textured regions.
   [2026-08-29 test: as score-spread across crops (mean + a*std of the 19 vote views, resnet_ft on
   canon2_val) it carries NO signal — a=0 best, a=2 costs 0.04. The cue, if real, is not visible
   in classifier-score disagreement; would need explicit per-patch noise estimates.]
   → per-patch noise estimation + uniformity test (classic forensics; JPEG-fragile, ensemble-only).
8. **Compositional intentionality prior**: reals are often pointless; generations look composed.
   Weak global semantic cue; CLIP-space likely encodes it.

**Physics-check architecture (recap):** two-stage — per-region physical descriptors (lighting dir,
shadow angle, noise, sharpness) → cross-region consistency module. I.e. the relation head over
engineered physical features. Unifying architecture for ALL cues above: per-patch descriptors +
attention consistency + global semantics, fused; standalone cheap cues join via the score-stacking
ensemble.


## General approach families (anti-overfit taxonomy) — 2026-08-27

Principle (Thinh): specific glitches are generator-specific and expire; approaches must target
cross-generator INVARIANTS. Three invariants: (a) camera-pipeline vs decoder statistics,
(b) weak cross-region coherence in fakes, (c) off-manifold semantics.

1. **Patch + relation architecture** (b): learn which cross-patch relations betray fakes — never
   hand-code a specific inconsistency. Flagship candidate.
2. **Real-manifold anomaly detection** (a) — NEW family: model reals only (noise-residual stats,
   camera-pipeline regularities), score deviation. Generator-agnostic by construction; new
   generators can't fake having passed through a camera. ~1 day build (one-class over residual feats).
3. **Spectral analysis** (a): upsampling signatures persist across generator families; band-limited
   + augmented to survive blur/JPEG.
4. **Pretrained-feature probes** (b+c): CLIP/DINO — in flight.
5. **Test-time prediction-consistency** (meta): score stability across crops/perturbations as a
   feature; free, reuses existing models.
6. **Diffusion-reconstruction error / DIRE** (a): pretrained diffusion reconstructs fakes better than
   reals. (Legit cousin of Thinh's day-1 diffusion instinct.) Heavy inference, GAN-weak; bench only.
7. **Diversity-designed stacking**: ensemble members chosen one-per-invariant-family, not by solo score.

**Selection discipline:** judge every approach ONLY by held-out-generator AUROC (ddpm row + official
DALL·E benchmark), never in-domain. Stretch: leave-one-generator-out rotation.
