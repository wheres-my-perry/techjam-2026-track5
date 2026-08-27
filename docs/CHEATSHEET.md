# Server Cheatsheet

Quick commands, copy-paste ready. (Deeper context: docs/SERVER.md.)

## Every new SSH session, first
```
cd ~/techjam-2026-track5
source .venv/bin/activate
```

## GPU status
```
nvidia-smi
```
Read per GPU: `XMiB / 32607MiB` = memory used/total, `NN%` = how busy.
Free GPU = ~0MiB and 0%. Pick one: `export CUDA_VISIBLE_DEVICES=0` (or 1).

## Disk space
```
df -h ~ | tail -1
```

## tmux (long-running jobs)
```
tmux new -s NAME          start a session
Ctrl-b d                  detach (job keeps running)
tmux attach -t NAME       come back
tmux ls                   list sessions
Ctrl-b c / Ctrl-b p       new window / previous window (inside tmux)
```

## Submit a job (Slurm — submit -> shell back -> monitor)
```
sbatch run_night.sbatch
squeue -u chim            is it running? (R = running, PD = queued)
scancel JOBID             stop it
```
Returns to prompt instantly; survives logout.

## Watch a running job
```
tail -f slurm_night_JOBID.log   live view (Ctrl-C exits view, job unaffected)
tail -n 5 slurm_night_JOBID.log  quick peek
grep -c "Saved to" slurm_night_JOBID.log   how many of the evals finished
```

## Kill a stuck run
Slurm job: `scancel JOBID` (find JOBID with `squeue -u chim`).
tmux run: attach (`tmux attach -t NAME`), press Ctrl-C.

## Update code from GitHub
```
git pull
```

## Run an evaluation (template)
```
python -m src.evaluate --manifest data/manifests/wildfake_test.csv --model resnet_ft:outputs/resnet_ft/wf_aug.pt --out outputs/resnet_ft/eval_x --limit 1200
```
Model names: cnn, clip_linear, resnet_ft, real_manifold; prefix `vote+` for crop voting.

## Copy results to your Mac (run ON THE MAC)
```
cd ~/Documents/code/hackathon/techjam-2026-track5
ssh -p 2205 chim@157.66.47.161 "cd techjam-2026-track5 && tar czf - --exclude=cache --exclude=*.state --exclude=*.pt --exclude=*.npz outputs slurm_night_*.log" | tar xzf -
```
