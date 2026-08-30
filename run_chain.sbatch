#!/bin/bash
#SBATCH --job-name=chain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_chain_%j.log
cd ~/techjam-2026-track5; export PYTHONPATH=.
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
NAME=${1:-canon4}; THR=${2:-0.15}
SPEC="vote(L=320)+pe_ft:outputs/pe_ft/$NAME.pt"
COND=clean,chain_repost,jpeg_twice
echo "== CHAIN stress, $NAME @ $THR, judges' benchmark  $(date)"
.venv/bin/python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" --threshold $THR --limit 1200 \
  --conditions $COND --out outputs/pe_ft/eval_${NAME}_official_chain 2>&1 | grep -vE "Warning|warn" | grep -E "^\||^At the|Saved|Traceback|Error"
echo "== CHAIN stress, $NAME @ $THR, unseen-64 probe  $(date)"
.venv/bin/python -m src.evaluate --manifest data/manifests/unseen64_tf.csv --model "$SPEC" --threshold $THR \
  --conditions $COND --out outputs/pe_ft/eval_${NAME}_unseen_chain 2>&1 | grep -vE "Warning|warn" | grep -E "^\||^At the|Saved|Traceback|Error"
echo CHAIN_DONE
