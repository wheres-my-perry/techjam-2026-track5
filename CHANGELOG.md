# Changelog

Dated, shipped changes only. (Current status & next steps: [docs/PROGRESS.md](docs/PROGRESS.md);
design candidates: [docs/IDEAS.md](docs/IDEAS.md); decisions: [docs/DECISIONS.md](docs/DECISIONS.md).)

## 2026-08-28 (night shift 2b — attention)
- New approach: src/approaches/patch_relation/ (01 stage 2) — frozen resnet_ft trunk + 2-layer
  transformer relation head over 3x3 native-res patch grid; registered as `patch_relation`.
  Sharded resumable embedding cache; self-contained checkpoint. Job: run_attn.{sh,sbatch} (GPU 2).

## 2026-08-28 (night shift 2)
- New approach: src/approaches/spectral/ (03) — FFT artifact features + logistic head,
  CPU-only, registered as model name `spectral`. Smoke-tested end-to-end on synthetic
  checkerboard/upsample fakes (harness integration verified).
- resnet_ft train gains --blur-boost (extra blur/downscale augmentation targeting the
  measured blur/resize weakness); new checkpoint target outputs/resnet_ft/wf_blur.pt.
- Night jobs: run_night2.{sh,sbatch} (GPU: blur retrain + vote+clip_linear/vote+cnn evals),
  run_spec.{sh,sbatch} (CPU: spectral train+eval, vote+real_manifold).

## 2026-08-28
- Night eval batch (Slurm job 10) complete: 7 evals at --limit 1200 across wf_test + official.
  Verdict in docs/PROGRESS.md; per-family actuals in docs/GENERATOR_MATRIX.md.
- Crop-voting wrapper (vote+<model>) validated: official benchmark resnet_ft 0.207 (inverted) ->
  vote+resnet_ft 0.938 clean / 0.913 mean transformed. wf_test clean 0.954.
- Workflow: back to Slurm (sbatch/squeue/scancel) per Thinh; SERVER.md, CHEATSHEET.md, and
  project-conventions skill reverted. Server has no rsync — results pulled via tar-over-ssh
  (command in CHEATSHEET.md).

## 2026-08-27

- Kill-resilient training: per-epoch resume state for cnn/resnet_ft, sharded resumable CLIP
  embedding cache (atomic shard writes), retry-loop sbatch wrappers. Smoke-tested (shard resume,
  cache keying, retry flow).
- Third approach `resnet_ft`: ImageNet-pretrained ResNet-50, fully fine-tuned, registered.
- WildFake pipeline completed on GPU server: 250K images indexed, manifests built
  (5 train generator families, ddpm held out, official benchmark 8,843+4,998 verified exact).
- `get_wildfake.py` rewritten CSV-driven (label_csv_files authoritative); selective zip extraction
  (`--extract-filter`), `--delete-zips`, `--official-val`; ModelScope CLI/API compat fixes.
- ARCHITECTURE.md + this changelog added (team-lead request: agent-maintained docs).
- First real-data result: scratch cnn (w64, aug, crop224, 15 ep) val AUROC **0.811** on 5-generator
  WildFake — vs 0.98 on single-generator CIFAKE. Capacity/diversity gap confirmed.

## 2026-08-26

- Repo bootstrapped (org `wheres-my-perry`, public), docs set (TRACK5_BRIEF/PROGRESS/DECISIONS/IDEAS),
  5 Claude agent skills in `.claude/skills/`.
- Evaluation harness: 15-condition contest transform grid, AUROC / balanced-acc@frozen-threshold /
  FPR@95%TPR, per-generator breakdown, error dumps, contest `predict` CLI, tests + synthetic smoke test.
- Approach structure `src/approaches/<name>/` (decision: folders over branches); `cnn` and
  `clip_linear` implemented.
- CIFAKE experiments: aug-vs-noaug robustness result (mean transformed AUROC 0.88→0.94, worst
  0.70→0.885 at clean cost −0.012) — core augmentation hypothesis validated. cnn w64: 0.961 mean
  transformed. clip_linear B/32 sanity run.
- Colab pipeline notebook (superseded by GPU server, kept as fallback).
