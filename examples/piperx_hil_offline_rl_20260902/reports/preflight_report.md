# Preflight report

Status is updated as gates complete. A formal training job must not be submitted while any required item is `PENDING` or `FAIL`.

## Experiment identity

- Evo-RL source: commit `6f2db449a21e1bac750b996f2e27cac6739aa63f`.
- Official first path: Pi*0.6 value → n-step advantage → binary ACP task-text tag → π0.5 continuation.
- Proposed IQL/double-Q/AWR weighted-flow-matching path: deferred extension, not labeled as the official mainline.

## Dataset gates

| Gate | Result | Evidence |
| --- | --- | --- |
| Revision | PASS | `18016184b09929643b6bd055b3f5833bfb2e7b85` |
| Exact primary inventory | PASS | 390 files / 8,162,275,443 bytes |
| Declared SHA256 | PASS | 388/388; undeclared primary files independently hashed |
| Episode/frame/FPS | PASS | 76 / 431,920 / 30 |
| Three-camera decode | PASS | 228/228 files, decoded frames exactly equal Parquet rows |
| State/action shape | PASS | 14 / 14 |
| Timestamp monotonicity | PASS | 76/76 episodes |
| TD segment boundaries | PASS | zero valid transitions crossing segment IDs |
| Segment/episode terminals | PASS | 1,103 segment boundaries + 76 episode terminals invalidated |
| Safety clipping | PASS | zero clipped frames |

Full evidence: `manifests/dataset_inventory.json`, `reports/dataset_audit.md`, `reports/field_coverage.md`.

## Outcome semantics and statistical limit

- 3 autonomous complete successes.
- 7 autonomous failures, usually early.
- 66 intervention-assisted recovered successes.
- 0 intervention-assisted unrecovered failures.
- Failure trajectories contribute 13,357 frames, 3.09% of all frames.

The official value target estimates outcome/progress under the behavior represented in the dataset, including human recovery. It must not be called a calibrated autonomous-success probability. Uniform frame sampling may underweight early failures; the official-default run is retained as the baseline, and any balanced correction must be separately named.

## Split

The split is grouped by capture `policy_hil.session_id` and has no session leakage.

| Split | Episodes | Frames | Success | Failure | Autonomous success | Autonomous failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 60 | 344,910 | 55 | 5 | 2 | 5 |
| Validation | 8 | 43,954 | 7 | 1 | 1 | 1 |
| Test | 8 | 43,056 | 7 | 1 | 0 | 1 |

Only two independent sessions contain autonomous successes, so all three splits cannot contain one without leakage. Validation carries the autonomous success/failure separation gate; test is not used to claim a two-class autonomous metric.

## Dataset compatibility adapters

| Adapter | Result | Scope |
| --- | --- | --- |
| Writable derived view | PASS | Parquet/meta copied inside workspace; 8GB videos absolute-symlinked to immutable SSD source |
| Parquet physical column order | PASS | 76/76 derived files reordered; semantic columns/rows unchanged; before/after SHA recorded |
| Task metadata index | PASS | Explicit `task` column adapted to the DataFrame index expected by commit `6f2db449` |
| Train-only normalization | PASS | Derived `meta/stats.json` uses only train `transition_valid` state and `transition_valid & action_valid` action rows; q01/q10/q50/q90/q99 included |
| Real sample | PASS | Correct task text, three `(3,480,640)` images, state/action `(14,)` |
| Null-free value view | PASS | 76 Parquet files / 431,920 rows; seven required scalar/tensor fields plus three linked videos |
| Null-free ACP/policy view | PASS | 76 Parquet files / 431,920 rows; value fields plus non-null `is_intervention`; source evidence view is unchanged |

## Base checkpoint provenance

- Source checkpoint: `.../train_state/pi05_piperx_full558_h100_8gpu_50k_20260828_r13/checkpoints/050000`.
- Model payload SHA256: `d85c7cd84060a924b6ef10d055491c500c5714a21ddbb49e1dbdc828b7a74147`.
- Training state includes optimizer, scheduler, RNG and `step=50000`.
- Source code marker: LeRobot `0.6.0`, commit `eb74530ea345c942212820952e46b4d186f9a9f0`.
- Model: PI0.5, no PEFT/LoRA, chunk/action steps 50, FP32, three cameras, state/action 14.
- Compatibility config view: PASS. Four configuration-only fields unsupported by the current official class were removed and recorded; model weights are an absolute symlink and byte-identical.
- Full policy weight load: PASS with the official Transformers fork. All state-dict keys loaded; 3,616,757,520 FP32 parameters on `cuda:0`, 14,483,422,720 allocated bytes.
- Continuation optimizer decision: reinitialize for official ACP/data-only comparison; retain the exact step-50000 training state as provenance because the old completed scheduler is not semantically suitable for a new continuation budget.

## Official value path

| Gate | Status |
| --- | --- |
| Official source pinned | PASS |
| Value trainer imports | PASS |
| 4090 CUDA visibility | PASS |
| Official Transformers fork | PASS — `dcddb970176382c0fcf4521b0c0e6fc15894dfe0` |
| SigLIP revision/download | PASS — `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` |
| Gemma revision/download | PASS — fixed revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`; 10 files / 575,484,103 bytes; model SHA256 `abb58eb73aece5163624090d5383c59cbcdb9e9539ec84cb864f74386c4b21d5` |
| One-batch forward/backward | PASS — loss `5.1824183464`, 694 finite gradient tensors, grad norm `58.8468924` |
| 100-step smoke | PASS — 60 train episodes / 344,910 frames, BF16 batch 4; step 100 loss `4.989`, finite grad norm `10.545` |
| Finite loss/gradient | PASS |
| Save/load/resume | PASS — checkpoints 50/100; patched official Pi*0.6 resume from 100 to 102, scheduler/optimizer/global step restored, `last -> 000102` |
| Held-out autonomous value separation | PENDING |
| Advantage/ACP distribution not collapsed | PENDING |

## H100 distributed gate

- Failed v1 and r1 attempts are recorded in `reports/v1_failure_report.md` and `reports/v1_r1_failure_report.md`; neither produced a training checkpoint or experimental result.
- Active retry: `v2sam-ego2exo-official-v1-r2`, job `job-b32f52e3-21c9-4122-803a-570cb7bc9ceb`.
- Pinned Transformers metadata/runtime 4.53.3: PASS.
- CUDA 12.8 / eight H100 visibility / NCCL 8-rank all-reduce: PASS.
- Pi*0.6 DDP unused-parameter patch: PASS; recorded as `scripts/0002-pistar06-ddp-find-unused.patch`, SHA256 `4641be6c154a6bce41d0bd960d949f0e609fd727243d97fb0f6e349611e42ef5`.
- Eight-GPU 200-step benchmark: PASS. Checkpoints 100 and 200 were written; step-200 loss `3.959`, grad norm `4.633`; all logged losses and gradients finite.
- Formal 8,000-step value training: PASS. Final checkpoint `checkpoints/v2sam-ego2exo-official-v1-value/checkpoints/008000`; final loss `0.598`, grad norm `5.219`, LR `1.0e-06`; `FORMAL_VALUE_PASS` recorded.
- The combined r2 job then failed before the first ACP inference batch because inference defaulted to TorchCodec and the container lacked compatible FFmpeg shared libraries. Training remains valid and is not rerun. Full evidence: `reports/v1_r2_failure_report.md`.
- PyAV value-inference patch: PASS on 4090 real-data gate, 76 episodes / 431,920 frames / three `(3,480,640)` images. Patch SHA256 `3e1811d22b99ffc4e12c45c3519e075a900bb9cc3429a4454016513702d370c4`.
- ACP-only recovery: `v2sam-ego2exo-official-v1-acp-r1`, job `job-d1b6708e-d435-4c90-ac85-41420eeac470`, submitted on the same 8×H100 type group; reuses step 8000 and does not train value.

## Submission gate

No H100 job has been submitted. Formal submission now waits only for full-dataset value inference, held-out value separation, non-collapsed ACP labels, and paired policy smoke/benchmark gates. The small Pi*0.6 resume compatibility patch is recorded as `scripts/0001-pistar06-resume-postprocessor-override.patch`; its failed pre-patch attempt is preserved in the logs.

The train split contains 9,757 failure frames (2.83% under uniform-frame sampling). Official target means are −0.566 for autonomous failures, −0.085 for autonomous successes and −0.162 for HIL recovered successes. These are target statistics only; the learned model must reproduce the ordering on held-out data.
