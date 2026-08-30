#!/bin/bash
#SBATCH --job-name=tiles
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_tiles_%j.log
cd ~/techjam-2026-track5; export PYTHONPATH=.
S=/tmp/claude-1006/-home-chim-techjam-2026-track5/217c8dee-cd23-4c85-b87a-c20ed3db7c0a/scratchpad
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
CK=outputs/pe_ft/canon4.pt
for spec in "grid:vote(L=320)" "t1:vote(t=1,L=320)" "t2:vote(t=2,L=320)" "t3:vote(t=3,L=320)"; do
  tag=${spec%%:*}; sp=${spec#*:}
  echo "== dump $tag  $sp  $(date)"
  .venv/bin/python -m scripts.crop_dump --root $S/randtest --model "$sp+pe_ft:$CK" --save outputs/random_gen/dump_$tag.npz 2>&1 | grep -vE "Warning|warn" | tail -3
  .venv/bin/python -m scripts.crop_agg outputs/random_gen/dump_$tag.npz
done
echo TILES_DONE
