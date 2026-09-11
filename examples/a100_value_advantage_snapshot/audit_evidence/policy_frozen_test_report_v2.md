# Attempt-aware ACP v2 frozen held-out test

This is the single test evaluation after mixture, training steps, and checkpoints were frozen. It uses 46 attempts and 184 fixed legal 50-step anchors. ACP models use the deployment-positive prompt.

| Model | Normalized chunk MAE (95% CI) | Physical chunk MAE | First-step MAE | Flow loss | Smoothness error |
|---|---:|---:|---:|---:|---:|
| base | 0.150134 [0.144145, 0.156463] | 7.734858 | 6.298904 | 0.108683 | 0.463950 |
| episode_acp_v1 | 0.037392 [0.034641, 0.040298] | 1.887521 | 0.824549 | 0.012954 | 0.441643 |
| bc20k | 0.032604 [0.030481, 0.034756] | 1.650281 | 0.737829 | 0.013389 | 0.425897 |
| acp20k | 0.032656 [0.030567, 0.034804] | 1.655195 | 0.731886 | 0.013874 | 0.405761 |

## Paired comparisons

- `acp20k` − `bc20k` normalized_chunk_mae: +0.000053 (paired attempt bootstrap 95% CI [-0.000463, +0.000488]).
- `acp20k` − `bc20k` physical_chunk_mae: +0.004914 (paired attempt bootstrap 95% CI [-0.017761, +0.026363]).
- `acp20k` − `episode_acp_v1` normalized_chunk_mae: -0.004736 (paired attempt bootstrap 95% CI [-0.006847, -0.002598]).
- `acp20k` − `base` normalized_chunk_mae: -0.117477 (paired attempt bootstrap 95% CI [-0.123298, -0.111618]).
- `bc20k` − `base` normalized_chunk_mae: -0.117530 (paired attempt bootstrap 95% CI [-0.123450, -0.111597]).

Negative deltas favor the left model. These offline imitation diagnostics are not robot success-rate estimates; the test result was not used to alter the selected 50% mixture, 20k duration, or checkpoint.
