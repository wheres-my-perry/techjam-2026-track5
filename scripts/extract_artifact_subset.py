"""Two-phase ArtiFact extraction: metadata first, then only the sampled images.

ArtiFact.zip holds 2,504,198 members; canon5 only ever used ~195K of them.
Extracting the whole tree costs ~an hour of wall clock and 2.5M inodes for
nothing, so:
  phase 1 (--meta)  extract every metadata.csv (the label source, ~30 files)
  phase 2 (--images <manifest>)  extract only the paths a manifest selected

Labels still come from the `target` column via build_artifact_manifest.py --
this script never infers a label from a folder name.
"""
import argparse, csv, os, zipfile

ZIP = "data/artifact/ArtiFact.zip"
DEST = "data/artifact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--images", help="manifest csv whose `path` column to extract")
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--dest", default=DEST)
    a = ap.parse_args()

    z = zipfile.ZipFile(a.zip)
    if a.meta:
        members = [m for m in z.namelist() if m.endswith("metadata.csv")]
        print(f"extracting {len(members)} metadata.csv files", flush=True)
        z.extractall(a.dest, members=members)
        print("META_DONE", flush=True)
        return

    if not a.images:
        raise SystemExit("need --meta or --images")
    want = []
    with open(a.images, newline="") as fh:
        for r in csv.DictReader(fh):
            p = r["path"]
            # manifest paths are <dest>/ArtiFact/<Parent>/<src>/<image_path>
            want.append(os.path.relpath(p, a.dest).replace(os.sep, "/"))
    have = set(z.namelist())
    members = [m for m in want if m in have]
    missing = len(want) - len(members)
    print(f"{len(members)} members to extract ({missing} manifest rows not in the zip)", flush=True)
    if missing:
        print("WARNING: manifest rows missing from the zip -- do not train until explained", flush=True)
    todo = [m for m in members if not os.path.exists(os.path.join(a.dest, m))]
    print(f"{len(todo)} not yet on disk", flush=True)
    for i in range(0, len(todo), 20000):
        z.extractall(a.dest, members=todo[i:i + 20000])
        print(f"  {min(i + 20000, len(todo))}/{len(todo)}", flush=True)
    print("IMAGES_DONE", flush=True)


if __name__ == "__main__":
    main()
