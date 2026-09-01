"""Upload trained checkpoints to the GitHub release as assets.

curl could not do this: --data-binary @file buffers 1.18 GB into RAM, and -T appends the local
filename to the URL, which corrupts the ?name= query and returns 400. requests streams a file
object with a real Content-Length. (2026-09-01)
"""
import hashlib, json, netrc, os, sys, requests

REPO = "wheres-my-perry/techjam-2026-track5"
TOKEN = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("github.com")[2]
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/canon6-v1", headers=H)
r.raise_for_status()
rel = r.json()
rid = rel["id"]
have = {a["name"]: a["id"] for a in rel["assets"]}
print(f"release {rid}, {len(have)} assets already there", flush=True)

for m in ["canon6_C", "canon6_AlowLR", "canon6_mlp", "canon6pe_mlp", "canon6_A", "canon6_B", "canon6_B6"]:
    f = f"outputs/pe_ft/{m}.pt"
    if not os.path.exists(f):
        print(f"  MISSING {m}", flush=True); continue
    name = f"{m}.pt"
    if name in have:
        print(f"  {name} already uploaded, skipping", flush=True); continue
    size = os.path.getsize(f)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    print(f"  uploading {name}  {size/1e6:.0f} MB  sha256 {h.hexdigest()[:16]}...", flush=True)
    with open(f, "rb") as fh:
        resp = requests.post(
            f"https://uploads.github.com/repos/{REPO}/releases/{rid}/assets",
            params={"name": name}, data=fh,
            headers={**H, "Content-Type": "application/octet-stream",
                     "Content-Length": str(size)}, timeout=1800)
    print(f"    http {resp.status_code}  {resp.text[:160] if resp.status_code != 201 else 'OK'}", flush=True)

r = requests.get(f"https://api.github.com/repos/{REPO}/releases/{rid}", headers=H).json()
print("=== assets on the release now ===")
for a in r["assets"]:
    print(f"  {a['name']:22s} {a['size']/1e6:7.0f} MB")
