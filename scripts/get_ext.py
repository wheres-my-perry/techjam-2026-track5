"""Re-fetch the canon4/canon5 large-image expansion sources (2026-08-31).

The originals were pulled ad-hoc on the server that died; this records the exact
repo ids and shard slices so the corpus is reproducible. Partial shards only --
canon5 capped every one of these sources at 9-15K images anyway, so pulling the
full repos (ELSA alone is 2.6 TB) would be pure waste.
"""
import os, sys
from huggingface_hub import snapshot_download

ROOT = "/workspace/techjam-2026-track5"
# local dir -> (repo id, [shard patterns])   names match scripts/build_ext_manifest.py SOURCES
JOBS = {
    "data/ext/ELSA_D3":                ("elsaEU/ELSA_D3",                 [f"data/train-0000{i}-*.parquet" for i in range(8)]),
    "data/sid_set":                    ("saberzl/SID_Set",                [f"data/train-0000{i}-of-00249.parquet" for i in range(8)]),
    "data/ext/midjourney-v6-recap":    ("Photoroom/midjourney-v6-recap",  [f"train_00{i}.parquet" for i in range(4)]),
    "data/ext/open-images-v7-subset":  ("bitmind/open-images-v7-subset",  [f"data/train-0000{i}-*.parquet" for i in range(6)]),
    "data/ext/AFHQv2":                 ("huggan/AFHQv2",                  None),
    "data/ext/celeba-hq":              ("mattymchen/celeba-hq",           None),
}

def main():
    only = sys.argv[1:] or list(JOBS)
    for local, (repo, pats) in JOBS.items():
        if not any(o in local for o in only):
            continue
        d = os.path.join(ROOT, local)
        os.makedirs(d, exist_ok=True)
        print(f"=== {repo} -> {local} pats={pats}", flush=True)
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=d,
                              allow_patterns=pats, max_workers=8)
            print(f"=== DONE {repo}", flush=True)
        except Exception as e:
            print(f"=== FAIL {repo}: {type(e).__name__}: {e}", flush=True)

if __name__ == "__main__":
    main()
