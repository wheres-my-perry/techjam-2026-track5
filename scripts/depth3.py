"""Pick ONE shipping cut-off the way production should: on the MIXED distribution the model will
actually meet (clean + all stack depths pooled), not on clean images alone. Then read every depth
at that single fixed threshold.

This is the fix for the drift seen in reading (A): the model was fine, the threshold was chosen in
the wrong place. Same rule we already wrote into the README limitations and then violated.
"""
import numpy as np, os

DEPTHS = [("0 clean", "clean"), ("1", "stack1_rand"), ("2", "stack2_rand"), ("3", "stack3_rand"),
          ("4", "stack4_rand"), ("5", "stack5_rand"), ("6 all", "stack6_rand")]
NAMES = ["canon6_mlp", "canon6_A", "canon6_AlowLR", "canon6_B", "canon6_B6", "canon6_C", "canon6pe_mlp"]
SHORT = {"canon6_mlp": "MLP base", "canon6_A": "A", "canon6_AlowLR": "A+lowLR", "canon6_B": "B",
         "canon6_B6": "B6", "canon6_C": "C", "canon6pe_mlp": "MLP+edits"}

data = {n: np.load(f"outputs/pe_ft/depth_{n}/scores.npz", allow_pickle=True)
        for n in NAMES if os.path.exists(f"outputs/pe_ft/depth_{n}/scores.npz")}
models = [n for n in NAMES if n in data]

thr = {}
for m in models:
    o = data[m]; y = o["labels"]
    pooled = np.concatenate([o[f"score_{c}"] for _, c in DEPTHS if f"score_{c}" in o.files])
    ypool = np.tile(y, sum(1 for _, c in DEPTHS if f"score_{c}" in o.files))
    thr[m] = float(np.quantile(pooled[ypool == 0], 0.99))

print("ONE cut-off per model, chosen on the POOLED clean+stacked distribution (1% false alarms)")
print("  " + "".join(f"{SHORT[m]:>12s}" for m in models))
print("  " + "".join(f"{thr[m]:12.4f}" for m in models))

for title, fn in (("RECALL", lambda s, y, t: (s[y == 1] >= t).mean()),
                  ("FALSE ALARMS", lambda s, y, t: (s[y == 0] >= t).mean())):
    print(f"\n{title}")
    print(f"  {'augs':<8s}" + "".join(f"{SHORT[m]:>12s}" for m in models))
    for label, cond in DEPTHS:
        row = f"  {label:<8s}"
        for m in models:
            o = data[m]; y = o["labels"]
            row += f"{fn(o[f'score_{cond}'], y, thr[m]) * 100:11.1f}%" if f"score_{cond}" in o.files else f"{'--':>12s}"
        print(row)

print("\nBalanced accuracy at depth 6 with the pooled cut-off")
for m in models:
    o = data[m]; y = o["labels"]; s = o["score_stack6_rand"]
    rec = (s[y == 1] >= thr[m]).mean(); fa = (s[y == 0] >= thr[m]).mean()
    print(f"  {SHORT[m]:<12s} recall {rec*100:5.1f}%  false alarms {fa*100:5.1f}%  balanced {(rec+(1-fa))/2*100:5.1f}")
