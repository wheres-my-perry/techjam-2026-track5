# GPU Server Handbook (mio03)

Self-contained orientation for any teammate or AI agent. Pair with ARCHITECTURE.md (system design)
and docs/PROGRESS.md (current status).

## Access
- `ssh -p 2205 chim@157.66.47.161` — password: ask Thinh (do not commit it).
- Shared machine: 2x RTX 5090 (32GB each), 64 CPU cores, 1.7TB disk (~300GB free), other users'
  jobs run alongside ours. Slurm partitions: `cpu` (default), `gpu`.

## Layout (everything lives in ~/techjam-2026-track5, a clone of this repo)
- `.venv/` — python env. ALWAYS `source .venv/bin/activate` first (fresh shells lack `python`).
  Installed: torch (CUDA), open_clip_torch, modelscope, datasets, numpy, Pillow, scikit-learn, pytest.
- `data/wildfake/raw/` — ~250K images: reals (coco-val2017 slice, afhq, celebahq, church, ffhq,
  imagenet), fakes (DDIM, DDPM, BigGAN, StyleGAN, StarGAN, VQVAE zips extracted, dalle3 subtree
  from DALLE.zip), plus `label_csv_files/` (authoritative labels).
- `data/manifests/` — wildfake_{train,val,test}.csv (ddpm held out of train/val — test only) and
  official_val.csv (8843 DALL-E-Advanced + 4998 COCO-val2017; NEVER train/tune on it).
- `outputs/<approach>/` — trained weights (*.pt / *.npz), resume state (*.pt.state), eval results
  (eval_*/results.json + robustness_table.md), clip embedding cache (outputs/clip_linear/cache/).
- Logs: `slurm_*.log` (Slurm jobs), `real_manifold.log`, `manifold_diag.log` (tmux runs).
- tmux: session `manifold` may exist (`tmux attach -t manifold`, detach Ctrl-b d).

## Running things (conventions — violating these has burned us)
- GPU work goes through Slurm: write an sbatch file (`--partition=gpu --gres=gpu:1`), `sbatch it`,
  monitor `squeue -u chim` + `tail -f slurm_<id>.log`. Empty squeue = finished (check log tail).
- **NEVER set CUDA_VISIBLE_DEVICES** (Slurm cgroup renumbering -> silent CPU fallback).
- Jobs get killed unpredictably on this box: long steps must resume (training saves *.state per
  epoch; clip extraction caches 2000-sample shards) and sbatch steps are wrapped in retry loops
  (see run_all.sbatch / run_official.sbatch as templates).
- torch.load needs `weights_only=False` for our checkpoints (new-torch default trap).
- Eval `--limit N` takes a seeded random subsample (never head-truncation).
- CPU-only work (real_manifold, scripts) can run directly in tmux without Slurm.

## Reproducing anything
All commands are in ARCHITECTURE.md + CHANGELOG.md + docs/approaches/*.md; training entry points:
`python -m src.approaches.<name>.train --help`. Evaluation:
`python -m src.evaluate --manifest data/manifests/<x>.csv --model <name>:outputs/<name>/<w> --out outputs/<name>/eval_x --limit 1200`.
