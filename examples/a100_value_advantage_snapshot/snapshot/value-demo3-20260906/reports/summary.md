# Value experiment interim report

Fixed episode-disjoint 90/10 split; the 10% holdout is reused for checkpoint selection, not an untouched final test. Steps below are best OBSERVED so far, not a proven global optimum.

| Task | Latest eval step | Best observed step | Effective epoch | Train CE | Holdout CE | Holdout MAE | Beats mean baseline |
|---|---:|---:|---:|---:|---:|---:|---|
| piperx_insert_copper_screw | 5000 | 4750 | 0.1342 | 1.3536 | 3.0536 | 0.01210 | True |

All selected trajectories are successful demonstrations. These results cannot establish failure discrimination or real-robot RL benefits.
