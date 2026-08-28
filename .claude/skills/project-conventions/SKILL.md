---
name: project-conventions
description: Working conventions for this repo, set by the team (Thinh). Apply in EVERY session touching this project - research, coding, experiments, documentation, and when reporting results to teammates.
---

# Project Conventions (Thinh's standing instructions)

## How to work
- **One thing at a time.** No multitasking across approaches/topics; finish or park explicitly before moving on.
- **Think from constraints first**, not approach-first. Correct the user when they're wrong — with reasons, kindly.
- **Never infer identity** from account/profile names. Ask. (User: Thinh; GitHub natsupercell.)
- Explain on request **briefly and easily, no jargon walls**; use analogies (the "generator schools" analogy is house style for generator families).
- Shell blocks must be zsh-paste-safe: **no # comments inside command blocks**; always include `cd` context; keep commands simple.

## Documentation discipline (nothing goes to waste)
- **Continuously document insights the moment they occur** — including negative results and dead ends, with the evidence and the verdict.
- One file per approach in `docs/approaches/NN-name.md`: mechanism, verdicts, results, kill-tests, status. ALL insights about that approach live there.
- `ARCHITECTURE.md` = how it works (update in the SAME commit as interface changes). `CHANGELOG.md` = dated shipped changes. `docs/PROGRESS.md` = status now / next. `docs/DECISIONS.md` = why. `docs/IDEAS.md` = candidates.
- Docs are **factual first, minimal second**. Credit ideas to who proposed them.

## Experiment discipline
- **Anti-overfit rule:** judge every approach ONLY by held-out-generator AUROC + official benchmark (DALL·E/COCO), never in-domain numbers. Generators are "schools"; don't build detectors married to one school.
- **Evidence-gated building:** cheapest testable slice first; a kill-test before any big investment; stop investing the moment evidence says stop.
- Negative results are deliverables — file them with the same care as wins.
- Never train/tune on the official validation slices (dalle3, coco-val2017). Label 1 = AI-generated, everywhere.

## Infra facts that bite
- GPU server work goes through **Slurm** (`sbatch` / `squeue` / `scancel`) — Thinh's decision
  2026-08-27, overriding an earlier no-Slurm request. tmux only for CPU-light interactive work.
- Under Slurm, NEVER set CUDA_VISIBLE_DEVICES (cgroup renumbering -> silent CPU fallback).
- The shared GPU server kills jobs unpredictably: all long steps must be resumable (epoch state files, sharded caches) and wrapped in retry loops.
- Heavy data/training runs on the GPU server; the sandbox and bridge VM have no ML-site network.

## Prediction discipline (added 2026-08-27)
- Maintain `docs/GENERATOR_MATRIX.md`: for every (approach x generator family), register a predicted
  outcome WITH reasoning BEFORE measuring; replace with measured verdicts. Never post-hoc rationalize.
- When analyzing any approach, always reason per generator FAMILY (diffusion/token/GAN/edit), not
  per individual model — models churn, families persist.

## Benchmark integrity (standing rule, Thinh 2026-08-28 — after the official_val size confound)
- The benchmark is validated INDEPENDENTLY of any model. A flawed model must never look good,
  and a flawed benchmark must never be able to hand out a high score.
- Every new or changed manifest MUST pass `python -m scripts.shortcut_audit --manifest <csv>`
  (metadata-only AUROC ~0.5 = clean; >0.65 = FAIL, results unreportable) plus
  `python -m scripts.size_audit` eyeball check, BEFORE any model result from it is reported.
- Re-run the audits periodically ("check it over and over"), not just at creation.
- Any too-good-to-be-true result (>= 0.99 anywhere, or perfect rows) triggers a shortcut hunt
  FIRST, celebration never. The 2026-08-28 lesson: official_val reals were 200x200 thumbnails
  vs 1024+ fakes; metadata alone scored ~1.0; three "miracles" were this one artifact.

## Content matching (standing rule, Thinh 2026-08-29 — after the church/bedroom skew)
- Size was the first dumb variable; CONTENT is the second. If a subject appears on only one
  side of the label (real bedrooms 0, fake bedrooms 21K), the model learns "bedroom = fake"
  and the score is subject recognition, not detection. Every content bucket that appears in
  the fakes must have reals in the SAME split, and vice versa — matched per split, not per corpus.
- Know what each generator was trained on before adding it; that IS its content. DDPM =
  LSUN church + bedroom, so its reals must be church + bedroom at the same native size.
- Gate: `python -m scripts.canary_audit --manifest <csv> --strict` on every new/changed
  manifest, alongside shortcut_audit. Canaries are models too weak to detect AI (mean colour,
  histogram, 8x8 thumbnail, sigma-8 blur); they must score ~0.5. Same bands as shortcut_audit.
  `python -m scripts.content_audit` gives the structural real-vs-fake table per subject.
- No source may dominate one side: cap any single real source so it is not >~20% of a split's
  reals (LSUN church was 44% of test reals; "church = real").
- Labels come from the dataset's own per-image label field, never folder names (ArtiFact:
  `Fake/afhq` is real photos; `pro_gan` holds both classes; `sfhq` is fake).
- Tampered/inpainted images (lama, mat, generative_inpainting, palette, glide-in) never enter
  train — a random crop of a locally-edited photo is usually an unedited crop with a "fake"
  label. They live in test as the tampered stress-test.
- Official-slice hygiene: any newly added source may CONTAIN an official benchmark slice under a
  different name (ArtiFact ships COCO val2017; caught 2026-08-29 with 487 rows already in train).
  Grep each new source for the benchmark's slices and exclude by name — do not wait for pixel proof.
- Hunt every >=0.99 row with the audits restricted to that subject/generator vs reals. Found
  2026-08-29: face_synthetics / star_gan / sfhq are separable from real faces by mean colour and
  file size alone — their perfect rows are excluded from any claim.
- The official benchmark FAILS the canary (colour 0.755, histogram 0.764: DALL·E palette vs
  COCO photos). We cannot change it; every official number carries that caveat, and a model
  that leans on colour will look better there than it deserves.

## Size-canonicalization tiers (Thinh 2026-08-28: transforms cannot fix disjoint size classes)
- Measure class size distributions first (size_audit). Near-overlap (ratio <~1.5x): seeded
  random-band resize (scripts/canonicalize.py) is adequate; verify empirically after training.
- Large gap (e.g. 200 vs 1024+): NO transform is trusted — either the output size or the resize
  factor must correlate with the label. Fix the DATA: re-source the mismatched class at native
  resolution, subset to overlap, or report the limitation. Never launder with math.
- Canonicalizer never upscales (target clamped to source size). Always audit after.
