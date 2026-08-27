# Changelog

Dated, shipped changes only. (Current status & next steps: [docs/PROGRESS.md](docs/PROGRESS.md);
design candidates: [docs/IDEAS.md](docs/IDEAS.md); decisions: [docs/DECISIONS.md](docs/DECISIONS.md).)

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
