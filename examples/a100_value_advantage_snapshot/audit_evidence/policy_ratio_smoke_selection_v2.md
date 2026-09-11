# Attempt-aware ACP v2 policy replay ratio selection

Selected replay mixture: **50% Base + 50% attempt-aware**. Selection used only the fixed validation anchors; held-out test was not accessed.

| Model | Val normalized chunk MAE | Val physical chunk MAE | First-step MAE | Flow loss | Smoothness error | Base retention normalized MAE |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.161819 | 8.411614 | 6.112664 | 0.109741 | 0.482678 | 0.204689 |
| acp25 | 0.043606 | 2.214490 | 1.086640 | 0.015314 | 0.517703 | 0.045830 |
| bc25 | 0.042979 | 2.182297 | 1.088019 | 0.014994 | 0.501243 | 0.046151 |
| acp50 | 0.041721 | 2.112989 | 0.990000 | 0.014850 | 0.498412 | 0.052349 |
| bc50 | 0.040726 | 2.068293 | 0.993473 | 0.014701 | 0.501553 | 0.052673 |

50% improved the primary validation metrics in both paired modes. Its Base-retention probe is weaker than 25%, but remains substantially better than the initial baseline probe. ACP Q-free ranking at 50% is weak/mixed, so tag utilization remains a formal-20k monitoring item. These are offline diagnostics, not robot success-rate estimates.
