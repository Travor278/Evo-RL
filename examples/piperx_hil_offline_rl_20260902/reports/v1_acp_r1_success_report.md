# Official value/ACP v1-r2 + ACP-r1 completion report

## Outcome

- Value training job `v2sam-ego2exo-official-v1-r2` produced the complete Pi*0.6 step-8000 checkpoint at `checkpoints/v2sam-ego2exo-official-v1-value/checkpoints/008000`.
- ACP retry job `v2sam-ego2exo-official-v1-acp-r1` (`job-d1b6708e-d435-4c90-ac85-41420eeac470`) completed successfully at `2026-09-02T13:12:39Z`.
- The retry reused the exact step-8000 checkpoint and did not retrain value.
- All 1,688 distributed inference batches completed in 5:49:20 and produced 431,920 predictions.
- Annotation writing completed for 76/76 episodes.

## Runtime gates

- PyAV three-camera gate: passed for `left_wrist`, `right_wrist`, and `top`.
- NCCL gate: all ranks 0-7 reported `NCCL_RANK_OK`, world size 8, NVIDIA H100 80GB HBM3.
- Video backend patch: `scripts/0003-value-infer-video-backend.patch`, SHA-256 `3e1811d22b99ffc4e12c45c3519e075a900bb9cc3429a4454016513702d370c4`.
- Value checkpoint used: `checkpoints/v2sam-ego2exo-official-v1-value/checkpoints/008000/pretrained_model`.

## Written fields

- `complementary_info.value_evorl_official_v1`
- `complementary_info.advantage_evorl_official_v1`
- `complementary_info.acp_indicator_evorl_official_v1`

The policy annotation view contains 76 parquet files, 431,920 frames, and all three fields are declared in `meta/info.json`.

## Offline gate

Evidence: `reports/v2sam-ego2exo-official-v1-offline-gate.json`.

All six gates passed:

- finite value
- finite advantage
- value not collapsed
- advantage not collapsed
- ACP indicator not all the same
- held-out autonomous-success value above autonomous-failure value

Global results:

- value mean/std: -0.169828 / 0.116310
- value range: [-0.659297, -0.000611]
- advantage mean/std: -0.00000565 / 0.013955
- advantage range: [-0.376481, 0.325670]
- ACP positive fraction: 0.377973
- validation autonomous success-minus-failure mean value: +0.092401

The observed ACP-positive fraction exceeds the 0.30 quantile target because intervention frames were explicitly forced positive, as configured.

## Statistical limitation

The autonomous validation separation gate is supported by only one autonomous-success episode and one autonomous-failure episode. It is a useful preflight sanity check, not a statistically strong effectiveness claim. The policy comparison and robot evaluation must carry the paper-level conclusion.
