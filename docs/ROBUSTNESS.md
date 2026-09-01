# Robustness under stacked augmentation

```
ROBUSTNESS UNDER STACKED AUGMENTATION
Row k = k DISTINCT transform families composed on the same image (6 = all of them).
n = 400 judges' images per cell.

  cut-off per model (1% false alarms on that model's CLEAN reals):
    MLP (shipped)      0.2927
    canon6_A           0.1283
    canon6_AlowLR      0.0483
    canon6_B           0.1153
    canon6_B6          0.1367
    canon6_C           0.4132
    canon6_mlp+edits   0.2355

RECALL — AI images caught
  augmentations  MLP (shipped     canon6_A canon6_AlowL     canon6_B    canon6_B6     canon6_C canon6_mlp+e
  ---------------------------------------------------------------------------------------------------------
  0 (clean)             98.4%        98.8%       100.0%        98.8%        98.8%       100.0%        99.6%
  1                     98.8%        98.4%       100.0%        98.8%        98.8%        99.2%       100.0%
  2                     98.8%        98.4%       100.0%        98.8%        98.8%        98.4%        99.2%
  3                     98.8%        98.0%       100.0%        98.4%        98.8%        98.0%        99.2%
  4                     99.2%        98.8%       100.0%        98.8%        99.6%        97.6%        99.6%
  5                     99.2%        98.4%       100.0%        98.8%        99.2%        94.0%        98.8%
  6 (all)               98.0%        98.4%        99.6%        99.6%        98.8%        94.8%        98.8%
  ------------------------------------------------------------
  drop 0->6             -0.4         -0.4         -0.4          0.8         -0.0         -5.2         -0.8 

AUROC
  augmentations  MLP (shipped     canon6_A canon6_AlowL     canon6_B    canon6_B6     canon6_C canon6_mlp+e
  ---------------------------------------------------------------------------------------------------------
  0 (clean)            0.9991       0.9969       1.0000       0.9988       0.9995       1.0000       0.9998
  1                    0.9977       0.9951       0.9998       0.9971       0.9966       0.9995       0.9991
  2                    0.9973       0.9950       0.9997       0.9965       0.9964       0.9992       0.9975
  3                    0.9941       0.9934       0.9993       0.9921       0.9936       0.9972       0.9932
  4                    0.9965       0.9942       0.9997       0.9910       0.9941       0.9954       0.9938
  5                    0.9944       0.9887       0.9979       0.9886       0.9913       0.9883       0.9881
  6 (all)              0.9907       0.9904       0.9968       0.9880       0.9879       0.9815       0.9865

FALSE ALARMS — real photos wrongly flagged
  (a model can hold recall by drifting every score up; this is the check)
  augmentations  MLP (shipped     canon6_A canon6_AlowL     canon6_B    canon6_B6     canon6_C canon6_mlp+e
  ---------------------------------------------------------------------------------------------------------
  0 (clean)              1.3%         1.3%         1.3%         1.3%         1.3%         1.3%         1.3%
  1                      2.6%         2.0%         5.3%         3.9%         2.6%         2.6%         9.2%
  2                      3.9%         3.9%        14.5%        13.2%         7.2%         1.3%        15.8%
  3                      9.9%         7.2%        25.0%        24.3%         9.2%         2.6%        25.7%
  4                      9.2%        10.5%        32.2%        32.2%        15.1%         3.3%        31.6%
  5                     13.8%        12.5%        42.8%        41.4%        17.1%         3.9%        34.9%
  6 (all)               19.1%        11.8%        50.7%        48.0%        22.4%         5.3%        40.8%
```
