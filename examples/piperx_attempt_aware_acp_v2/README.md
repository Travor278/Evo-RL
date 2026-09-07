# attempt-aware ACP v2

This experiment treats every independent copper-screw insertion attempt as one Value trajectory. It is intentionally separate from the earlier episode-level ACP v1 experiment.

## Frozen inputs

- Attempt dataset: `Travor278/evorl-piperx-copper-screw-attempt-aware-clean-v2`
- Revision: `d8c2c0f0115ce3456dbfa38f5529276c0e4bbe16`
- Evo-RL base commit: `4bf0675bbc2569e43047e1ac0c640b563c855ea8`
- Development branch: `codex/piperx-attempt-aware-acp-v2`
- Seed: `20260906`

## Outcome gate

The source snapshot is immutable. `episode_success` was not broadcast from a physical five-insertion Episode into attempts. The derived `attempt_value_view` contains one LeRobot Episode per attempt and preserves the original label as `source_episode_success`.

Evidence reviewed:

- all 17 attempts from seven source-marked failures;
- all 50 attempts from ten old source-success Episodes;
- one physical Episode from every new collection, all five attempts (50 attempts total).

Adjudicated totals:

- known success: 475
- known failure: 7
- unknown: 0
- physical/attempt outcome conflicts: 10
- autonomous success: 101
- intervention-recovered success: 374
- autonomous failure: 5
- operator abort: 1
- timeout failure: 1

Split coverage after outcome adjudication:

- train: 381 success / 5 failure
- validation: 49 success / 1 failure
- test: 45 success / 1 failure

The split remains collection- and physical-Episode-disjoint. The held-out test split is not used to choose replay ratio, step count, or checkpoint.

## Mainline invariants

- Value data is attempt-aware train data only.
- Base 558 is policy replay / anti-forgetting regularization only.
- Base 558 does not receive Value, Advantage, or ACP labels.
- HIL Value sampling is attempt-uniform.
- Policy replay is source-first; HIL then uses attempt-uniform legal 50-step chunk starts.
- Thresholds are computed from train known-outcome valid transitions only.
- Validation selects 25% versus 50% attempt-aware replay.
- ACP and Data-only paired runs share initialization, data, sampler, seed, batch size, steps, and resources; only ACP conditioning differs.

## Platform paths

- Experiment root: `/inspire/hdd/project/luojianlan/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906`
- Verified source snapshot: `/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906/attempt_aware_snapshot_d8c2c0f0115ce3456dbfa38f5529276c0e4bbe16`
- Value view: `/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906/attempt_value_view`
- Base 558: `/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/pi05_full558_044/dataset_v3_training_view`

Machine-readable evidence is in `manifests/` and `reports/outcome_contract/`; logs are in `logs/`.
