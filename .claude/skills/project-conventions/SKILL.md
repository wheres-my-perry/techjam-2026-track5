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
- **NO SLURM on the GPU server** (server owner's rule, 2026-08-27). Run in tmux directly; pick a GPU
  with `nvidia-smi` then `export CUDA_VISIBLE_DEVICES=<free id>` (correct outside Slurm only).
- Under Slurm, NEVER set CUDA_VISIBLE_DEVICES (cgroup renumbering -> silent CPU fallback).
- The shared GPU server kills jobs unpredictably: all long steps must be resumable (epoch state files, sharded caches) and wrapped in retry loops.
- Heavy data/training runs on the GPU server; the sandbox and bridge VM have no ML-site network.

## Prediction discipline (added 2026-08-27)
- Maintain `docs/GENERATOR_MATRIX.md`: for every (approach x generator family), register a predicted
  outcome WITH reasoning BEFORE measuring; replace with measured verdicts. Never post-hoc rationalize.
- When analyzing any approach, always reason per generator FAMILY (diffusion/token/GAN/edit), not
  per individual model — models churn, families persist.
