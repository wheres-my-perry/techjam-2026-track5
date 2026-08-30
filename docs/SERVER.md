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

## Running things (conventions — 2026-08-27: back to Slurm, per Thinh)
- Submit GPU work as Slurm jobs: `sbatch <script>.sbatch` -> returns instantly -> monitor with
  `squeue -u chim` and `tail -f slurm_<name>_<jobid>.log`. Cancel with `scancel <jobid>`.
- Job scripts: `#SBATCH --partition=gpu --gres=gpu:1`; NEVER set CUDA_VISIBLE_DEVICES inside a
  Slurm job (cgroup renumbering -> silent CPU fallback). Slurm assigns the GPU.
- Outside Slurm (rare: tmux/interactive), the opposite: check `nvidia-smi`, then
  `export CUDA_VISIBLE_DEVICES=<free id>` before running.
- Long steps still must resume (training *.state per epoch; clip shard cache) and be wrapped in
  retry loops — see run_night.sbatch for the template.
- torch.load needs `weights_only=False` for our checkpoints (new-torch default trap).
- Eval `--limit N` takes a seeded random subsample (never head-truncation).
- Updating code: `git fetch origin && git reset --hard origin/main` (NOT `git pull` — the server
  is a read-only mirror; results committed from the Mac collide with its untracked copies).
- CPU-only work (real_manifold, scripts) can run in tmux directly, no GPU pinning needed.

## Reproducing anything
All commands are in ARCHITECTURE.md + CHANGELOG.md + docs/approaches/*.md; training entry points:
`python -m src.approaches.<name>.train --help`. Evaluation:
`python -m src.evaluate --manifest data/manifests/<x>.csv --model <name>:outputs/<name>/<w> --out outputs/<name>/eval_x --limit 1200`.

## GPU numbering under Slurm (found 2026-08-30)
Slurm's GRES index is the REVERSE of `nvidia-smi`'s: Slurm `IDX:0` = nvidia-smi GPU 1
(PCI 81:00.0, GPU-3b677bf6…), Slurm `IDX:1` = nvidia-smi GPU 0 (PCI 41:00.0, GPU-9ed22abb…).
A `--gres=gpu:1` job therefore lands on nvidia-smi GPU 1 first when both are free. Slurm also
cannot see jobs started outside Slurm, so a bare process on a card does not stop Slurm from
handing that card out. Rules: every GPU job goes through sbatch (Thinh, 2026-08-30); if a card
must be avoided, reserve `--gres=gpu:2` and assert the UUID of CUDA device 0 at job start
(see run_r27.sbatch) — never set CUDA_VISIBLE_DEVICES inside the job.
