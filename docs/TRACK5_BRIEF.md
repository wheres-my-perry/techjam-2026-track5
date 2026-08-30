# Track 5 — Robust Detection of AI-Generated Images Under Real-World Transformations

Source: official TechJam 2026 Tracks & Problem Statements (Lark wiki, Early Bird, last updated Aug 26).
Workshop webinar: **Aug 28, 5:00–5:45pm SGT**.

**Verified against the full source text on 2026-08-30 (Thinh pasted it):** this condensation is
faithful. Two things the source settles: (1) "a subset of the following augmentations" limits WHICH
transforms, not how many per image — stacking is not excluded, and the background text ("compressed,
cropped, reposted") describes chains (Thinh). So the 15-condition single-transform grid is the
minimum, and stacked conditions (fixed chains + seeded random 2- and 3-stacks, `EXTRA_GRID`) are
reported alongside it. (2) There is no
hidden scored test set, metric or threshold: the COCO/DALL·E set is "for demonstration purposes
only" and the final score is the five judging criteria. The script must output `pred` (a
confidence); the fixed cut-off is our product decision, judged under "false positives / trade-offs".

## Background

Generative AI makes highly realistic synthetic images easy to create at scale → misinformation, impersonation, fraud, loss of trust. Detection is hardest **after** images are compressed, cropped, reposted, or lightly edited — robustness matters more than lab-only accuracy.

## Problem statement

Build a prototype that distinguishes **AI-generated images from authentic images**, maintaining accuracy under realistic post-processing. Present a clear technical approach, an evaluation strategy, and discussion of trade-offs (robustness, generalisation, false positives).

### Robustness transformations (a subset of these will be considered)

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG compression | quality = 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian blur | kernel σ = 0.5, 1.0, 2.0 | Out-of-focus |
| Resize | scale 0.5× / 0.25× then upscale | Thumbnail generation |
| Gaussian noise | σ = 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color jitter | brightness/contrast/sat ±20% | Filter apps, auto-enhance |
| Center crop | crop 80% | Profile-picture cropping |

## Constraints & scope

- **In scope:** image-level AIGC detection, robustness to common transforms, feature engineering, model design, evaluation design, error analysis, explainability ideas.
- **Out of scope:** full production deployment, platform-wide moderation systems, video/audio modalities.
- **Limits:** hackathon-scale prototype, limited compute. **Models must be < 2B parameters.**
- **Allowed:** public/licensed datasets, self-created transformed test cases, stated deployment assumptions.

## Data

- Datasets suggested: [SID_Set (HF)](https://huggingface.co/datasets/saberzl/SID_Set), [CIFAKE (Kaggle)](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), [WildFake (ModelScope)](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
- **Validation set (demo benchmark only, DO NOT TRAIN ON IT):** WildFake subset — Non-AIGC: COCO val2017 (4,998) · AIGC: DALL·E Advanced (8,843). Reference benchmark only; does not contribute to final score.

## Deliverables

1. **Written description (Devpost):** how solution addresses problem, dev tools, models/APIs, libraries/frameworks, datasets/assets used.
2. **Public repo:** well-structured commented code; **a script: image directory in → JSON out with `image_path` and `pred` (confidence it's AI-generated) per image**; README with overview, setup, steps to reproduce, limitations & future work, team-member contributions.
3. **Demo video:** end-to-end (inference results/dashboard/predictions), public on YouTube, linked in Devpost, no unlicensed third-party content.
4. **Robustness evaluation summary:** compact table/visual comparing clean vs transformed performance.
5. **Error analysis note:** representative false positives/negatives + trade-offs.

## Judging criteria (track-specific weights)

| Criterion | Weight |
|---|---|
| Technical execution (code, architecture, reliable demo, primary metric & robustness) | **35%** |
| Innovation & problem insight | 20% |
| Impact & relevance | 20% |
| Feasibility & practicality | 15% |
| Presentation & communication (final event only) | 10% |
