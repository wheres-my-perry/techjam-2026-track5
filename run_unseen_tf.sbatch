#!/bin/bash
#SBATCH --job-name=unseen_tf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_unseentf_%j.log
cd ~/techjam-2026-track5; export PYTHONPATH=.
GPU0=GPU-9ed22abb-341d-43a5-c1d5-ca27c8f5d238
got=$(.venv/bin/python -c "import torch; print('GPU-'+str(torch.cuda.get_device_properties(0).uuid))")
echo "CUDA device 0 = $got (physical GPU 0 = $GPU0)"
if [ "$got" != "$GPU0" ]; then echo "ABORT: device 0 is not physical GPU 0"; exit 1; fi
CK=outputs/pe_ft/canon4.pt
echo "== UNSEEN-64 under the 15-condition grid, canon4, plain mean, fixed 0.15  $(date)"
.venv/bin/python -m src.evaluate --manifest data/manifests/unseen64_tf.csv --model "vote(L=320)+pe_ft:$CK" \
  --threshold 0.15 --out outputs/pe_ft/eval_canon4_unseen_tf 2>&1 | grep -vE "Warning|warn" | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo "== UNSEEN-64 under the grid, canon4, log-odds mean (option C), provisional 0.035  $(date)"
.venv/bin/python -m src.evaluate --manifest data/manifests/unseen64_tf.csv --model "vote(f=1,L=320)+pe_ft:$CK" \
  --threshold 0.035 --out outputs/pe_ft/eval_canon4_unseen_tf_logit 2>&1 | grep -vE "Warning|warn" | grep -E "^\||^At the|^\*\*|Saved|Traceback|Error"
echo UNSEEN_TF_DONE
