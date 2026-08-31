# Claude Code briefing — TechJam 2026 Track 5 (AIGC detection)

You are joining mid-project. Team lead: Thinh (GitHub natsupercell). Deadline: Sep 1 12:00 SGT.

## Read these first, in order
1. docs/PROGRESS.md — current state + full decision log (newest first)
2. .claude/skills/project-conventions/SKILL.md — Thinh's standing rules (BINDING)
3. docs/DATA.md — datasets incl. the size-confound section; docs/DATA_CANDIDATES.md — approved expansion
4. ARCHITECTURE.md + docs/approaches/*.md — system design, per-approach verdicts
5. docs/CHEATSHEET.md — server commands; docs/SERVER.md — GPU server handbook

## Non-negotiable rules (full list in the skill)
- ONE topic at a time. Correct Thinh when he is wrong. Explain simply when asked.
- Command blocks must be zsh-paste-safe: no # comments inside blocks, include cd.
- Benchmark integrity: every new/changed manifest passes ALL the gates BEFORE any result is
  reported. Run them with ONE command so none can be forgotten:
      python -m scripts.audit_all --prefix data/manifests/<name>          (train/val/test)
      python -m scripts.audit_all --manifest <eval>.csv --eval-set        (an evaluation set)
      python -m scripts.corpus_audit --prefix data/manifests/<name> --write-drop <drop.txt>
  The full list, and what each one is FOR (updated 2026-08-31 — the old rule named only the
  first three, which is exactly why content_audit got skipped and the canon2 'bedroom = fake'
  bug came back in canon6 at 92.7:1):
    * label_provenance_audit.py --strict  labels re-derived from source, independent of the
      builder (added 2026-08-30 after 24% of "fakes" turned out to be real photos)
    * bucket_audit.py --strict            real:fake balanced in every NATIVE size bucket
    * shortcut_audit.py                   metadata-only AUROC ~0.5
    * size_audit.py                       per-class image dimensions
    * canary_audit.py                     deliberately dumb pixel models must score ~0.5
    * content_audit.py                    every SUBJECT appears on both sides of the label.
      NON-NEGOTIABLE: one-sided content may be kept for TESTING but NEVER used for TRAINING —
      it is both a shortcut ("bedroom = fake") and a competence limit (a model trained on
      bedrooms detects bedrooms and nothing else; this is why canon2 scored 0/10 on wild photos)
    * corpus_audit.py                     blank/corrupt images, byte duplicates within and across
      splits, and val/test rows that are perceptual copies of a training image
  KNOW WHAT EACH GATE CANNOT SEE. shortcut_audit and size_audit read the CANONICAL files, which
  are all one size after canonicalization, so they are structurally BLIND to native size: a set
  can pass at 0.62 while "big = real" is perfectly learnable. audit_all checks native size from
  the manifest's `long` column for exactly this reason. Evaluation sets with one-class size
  buckets are reported SIZE-MATCHED only (scripts/size_matched.py).
  Too-good results (>=0.99) trigger a shortcut hunt, never celebration.
- Git topology (CHANGED 2026-08-28, Thinh's call — server is now primary): the SERVER clone
  (~/techjam-2026-track5 on chim@157.66.47.161:2205) is the WORKING clone; commit and push from
  there. The Mac clone (~/Documents/code/hackathon/techjam-2026-track5) is now a read-only
  mirror — refresh it with `git fetch origin && git reset --hard origin/main`, do not commit
  there. Code and results flow server -> GitHub -> Mac. Do not change this without asking Thinh.
- Work directly on the server: data, GPUs, and Slurm logs are all local there, so read logs and
  run jobs in place rather than over ssh. Run interactive sessions inside tmux (`tmux new -s
  claude`) — a dropped SSH connection must not kill work in progress.
- Another agent (Claude in Cowork) may still hold uncommitted work in the Mac clone. Before big
  edits anywhere, `git status`; never revert files you didn't change.
- Slurm for GPU work: sbatch with explicit --cpus-per-task AND --mem (defaults are broken here).
  NEVER set CUDA_VISIBLE_DEVICES inside a Slurm job. Node mio03: 64 CPU, 125G RAM, 2x RTX 5090.
- torch.load needs weights_only=False for our checkpoints. Eval --limit is a seeded subsample.
- Never train/tune on official benchmark slices (dalle, coco). Secrets never in git; server
  password: ask Thinh.
- Document as you go: CHANGELOG.md + docs/PROGRESS.md for every change; per-approach insights
  (including negative results) in docs/approaches/; predictions BEFORE measurements in
  docs/GENERATOR_MATRIX.md.

## State snapshot (2026-08-28 night — verify against docs/PROGRESS.md, it wins)
Both original benchmarks had a size->label confound; all pre-confound official numbers are void.
Honest cells: GANs ~0.91-0.93, vqvae 0.66-0.70, official_v2 (std-wrapped) ~0.89. Jobs likely in
flight or done: run_data.sbatch (ArtiFact+LSUN download -> canon2 corpus, cpu) chained to
run_canon2.sbatch (clean resnet retrain + evals, gpu). Read their slurm_*.log tails before
starting anything on the GPU.
