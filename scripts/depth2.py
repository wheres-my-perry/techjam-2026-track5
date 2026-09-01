"""Two readings of the same stacked-augmentation data, from the SAME saved scores.

A) FIXED cut-off  -- chosen once on clean reals, held constant. The product number: what happens if
   you ship one threshold and the world corrupts your inputs.
B) RECALIBRATED   -- the cut-off re-derived at each depth for 1% false alarms. Removes calibration
   drift and shows the model's intrinsic separating power at that corruption level.

A model that looks good under B and bad under A does not need retraining -- it needs its threshold
set per condition, which is not something we can do in production. (Thinh, 2026-09-01)
"""
import numpy as np, os
from sklearn.metrics import roc_auc_score

DEPTHS = [("0 clean", "clean"), ("1", "stack1_rand"), ("2", "stack2_rand"), ("3", "stack3_rand"),
          ("4", "stack4_rand"), ("5", "stack5_rand"), ("6 all", "stack6_rand")]
NAMES = ["canon6_mlp", "canon6_A", "canon6_AlowLR", "canon6_B", "canon6_B6", "canon6_C", "canon6pe_mlp"]
SHORT = {"canon6_mlp": "MLP base", "canon6_A": "A", "canon6_AlowLR": "A+lowLR", "canon6_B": "B",
         "canon6_B6": "B6", "canon6_C": "C", "canon6pe_mlp": "MLP+edits"}

data = {}
for n in NAMES:
    f = f"outputs/pe_ft/depth_{n}/scores.npz"
    if os.path.exists(f):
        data[n] = np.load(f, allow_pickle=True)
models = [n for n in NAMES if n in data]

for title, recal in (("A) FIXED cut-off, set once on clean reals (the product number)", False),
                     ("B) RECALIBRATED at each depth to 1% false alarms (intrinsic power)", True)):
    print(f"\n{title}")
    hdr = f"  {'augs':<8s}" + "".join(f"{SHORT[m]:>12s}" for m in models)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for label, cond in DEPTHS:
        row = f"  {label:<8s}"
        for m in models:
            o = data[m]; y = o["labels"]
            if f"score_{cond}" not in o.files:
                row += f"{'--':>12s}"; continue
            s = o[f"score_{cond}"]
            thr = (np.quantile(s[y == 0], 0.99) if recal
                   else np.quantile(o["score_clean"][y == 0], 0.99))
            row += f"{(s[y == 1] >= thr).mean() * 100:11.1f}%"
        print(row)

print("\nBalanced accuracy at depth 6 (recall and specificity averaged, FIXED cut-off)")
for m in models:
    o = data[m]; y = o["labels"]; s = o["score_stack6_rand"]
    thr = np.quantile(o["score_clean"][y == 0], 0.99)
    rec = (s[y == 1] >= thr).mean(); fa = (s[y == 0] >= thr).mean()
    print(f"  {SHORT[m]:<12s} recall {rec*100:5.1f}%  false alarms {fa*100:5.1f}%  "
          f"balanced {(rec + (1 - fa))/2*100:5.1f}   AUROC {roc_auc_score(y, s):.4f}")
