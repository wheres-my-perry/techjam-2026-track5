#!/bin/bash
# pe_seg: per-patch localisation head on SID_Set (real / synthetic / tampered+mask).
#   bash run_seg.sh <out_name> [epochs] [limit_train]
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
NAME=${1:-sid}; EP=${2:-2}; LIM=${3:-0}
CK=outputs/pe_seg/$NAME.pt
rm -f $CK
python -m src.approaches.pe_seg.train --train data/manifests/seg_train.csv --val data/manifests/seg_val.csv \
  --epochs $EP --batch 16 --workers 12 --limit-train $LIM --out $CK || exit 1
echo "== HELD-OUT TEST (SID images never trained on)"
python - "$CK" <<'PY'
import sys, torch
from torch.utils.data import DataLoader
from src.approaches.pe_seg.train import SegDataset, collate, evaluate
from src.approaches.pe_seg.model import PESegNet
from src.approaches.pe_ft.model import pick_device
dev = pick_device(); net = PESegNet(pretrained=False); net.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]); net.to(dev)
dl = DataLoader(SegDataset("data/manifests/seg_test.csv", False), batch_size=16, num_workers=12, collate_fn=collate)
print("TEST", {k: round(v, 4) for k, v in evaluate(net, dl, dev).items()})
PY
echo "== WILD"
python -m scripts.wild_eval --model "pe_seg:$CK"
echo "SEG DONE"
