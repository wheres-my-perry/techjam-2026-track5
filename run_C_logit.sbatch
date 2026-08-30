#!/bin/bash
#SBATCH --job-name=C_logit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_C_%j.log
cd ~/techjam-2026-track5; export PYTHONPATH=.
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
SPEC="vote(f=1,L=320)+pe_ft:outputs/pe_ft/canon4.pt"
THR=0.035
echo "== OPTION C: log-odds mean. OFFICIAL at provisional threshold $THR (scores.npz allows any sweep)  $(date)"
.venv/bin/python -m src.evaluate --manifest data/manifests/canon_official.csv --model "$SPEC" \
  --threshold $THR --limit 1200 --out outputs/pe_ft/eval_canon4_official_logit 2>&1 | grep -vE "Warning|warn" | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo "== OPTION C: CANON4_TEST (8000 seeded)  $(date)"
.venv/bin/python -m src.evaluate --manifest data/manifests/canon4_test.csv --model "$SPEC" \
  --threshold $THR --limit 8000 --out outputs/pe_ft/eval_canon4_test_logit 2>&1 | grep -vE "Warning|warn" | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo C_DONE
