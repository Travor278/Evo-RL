# Pure-demo SFT + RL

This is the new, independent implementation of **pure human demonstrations → visual Value → global positive Top10% windows → explicit D/D_high resampling → ordinary Pi0.5 flow-matching continuation**. It does not use PPO, ACP text tags, intervention data, or the historical mixed-Value selected clips.

The old A100 experiments remain unchanged in `../a100_value_advantage_snapshot/`. Their original audit predates this implementation. **Code tests are available; no formal training run or real-robot result is claimed by this example.**

## What can be reused from the old runs

| Old component | Role here |
|---|---|
| Pi0.5 step50000 SFT model | Shared starting checkpoint, verified by SHA before fresh policy continuation |
| 20k data-only BC trainer | Same flow loss, optimizer/scheduler and parameter scope; intervention replay is replaced by D/D_high |
| ACP Value/Advantage machinery | Mathematical building blocks only; its labels and intervention-trained weights are ineligible |
| A100 pure-vision SigLIP architecture | Standalone image-only encoder/head with 201 bins and two-hot CE |
| Global A50/Top10%/20-window export | Reimplemented with a frozen pure-demo contract and final-policy-test exclusion |

The checkpoint called `baseline` in the historical continuation is already SFT-trained. BC20k is additional supervised training with intervention data, corresponding to SFT+Int in the requested ablation terminology. ACP20k additionally conditions on positive/negative Advantage text. Those trained artifacts cannot be renamed to the new SFT+RL arm.

## Verified control evidence

`control_evidence/bc20k_train_config.json` was downloaded from the pinned published model revision and SHA-verified against the existing deployment manifest. `bc20k_run.json` is the corresponding actual run provenance.

- Source model repository: `Travor278/pi05-piperx-attempt-v2-bc-20k-r2`, revision `e7f9e3185c2c0e5741f6d4201df8773f8b910c3d`.
- Shared SFT model SHA: `d85c7cd84060a924b6ef10d055491c500c5714a21ddbb49e1dbdc828b7a74147`.
- 20,000 policy updates, 8 ranks × batch8 = global64; action horizon50.
- FP32 parameters with BF16 Accelerate autocast; full parameter scope (`train_expert_only=false`, `freeze_vision_encoder=false`).
- AdamW LR2.5e-5, betas(0.9,0.95), weight decay0.01, epsilon1e-8, gradient clipping1.
- Saved scheduler says warmup1200/decay50000; the actual scheduler source at run commit `37de4c0b59ec152a2626f9c3d1652bc444018438` matches this branch's scheduler and rescales to **warmup480/decay20000** for a20k run. The original cosine step origin is preserved.
- Checkpoints at1000/5000/10000/15000/20000, log every5 updates.

The old50:50 mixture is Base/intervention-attempt. A D/D_high50:50 continuation can be proposed when this is confirmed as the paper counterpart, but the CLI requires an explicit reviewed fraction. It never derives a mixture from directory sizes. The actual Value budget, allowed pure-demo scope and final policy test identities must also be supplied before training.

The historical counterpart used8×H100. The inspected machine has8×A10040GB; identical full-parameter DDP settings have not been memory-validated there. Compute topology and any memory-sharding strategy belong in the pre-launch review. No resource or runtime change is silently selected by this implementation.

## Implementation map

| Module under `src/lerobot/rl/sft_rl/` | Function |
|---|---|
| `manifest.py`, `protocol.py` | Reviewed human-only complete-episode contract, final-test exclusion, train-only Z, n50 residual, global Top10% then positivity, merged20-action intervals |
| `value_model.py`, `value_data.py`, `value_training.py` | Pretrained SigLIP, no language/state inputs, 201-bin two-hot CE, fixed-budget training, train/holdout CE+MAE, checkpoint selection, save/resume/inference |
| `replay.py`, `config.py` | D and D_high indexed views, explicit source probability, update-indexed reproducible sampler, control/config/checkpoint provenance gates |
| `loss.py` | Pi0.5 temporal padding exclusion, including per-sample loss; output horizon is unchanged |
| `export.py` | Physical LeRobot video/Parquet export, exact action/state checks, all-frame video reads, timestamps, camera mapping, real batch reads, example preview videos |
| `policy_validation.py`, `reporting.py` | Reload/inference checks, measured curves/distributions, protocol-bound operator trial aggregation |
| `provenance.py` | Source hashes, Git identity when available, numerical library versions |

`lerobot_train.py` only activates the new replay when `sft_rl.enable=true`. ACP, HIL replay and RA-BC weights are rejected in that mode. Pi0.5's normal flow-matching objective remains intact; padded temporal targets are excluded from the supervised reduction. The generic padding correction also applies when another Pi0.5 dataset supplies `action_is_pad`.

D_high is trained through an indexed view of the original videos: every retained frame may be an origin, not only the Top10% anchors. Action queries are clamped at that selected interval's boundary before preprocessing, and `action_is_pad` reaches the loss through the real processor. Physical exports are audit artifacts of the same intervals; they do not create extra re-encoding noise in policy inputs. Statistics are never silently recomputed from D+D_high.

## Stage commands

Use this branch's installed source in an environment matched to the selected control. The commands below do not run as a queue: each stage is explicitly invoked. All outputs should be new directories. Variables refer to reviewed paths/values, not hidden defaults.

1. Fill `scope.template.json`, `final_policy_test.template.json` and `value_run.template.json`. Unknown outcomes require explicit reviewed per-episode evidence; the builder does not invent success labels.

```bash
python -m lerobot.scripts.lerobot_sft_rl build-contract \
  --scope "$SCOPE" --final-policy-test "$FINAL_POLICY_TEST" --output "$CONTRACT"

python -m lerobot.scripts.lerobot_sft_rl prepare-value \
  --contract "$CONTRACT" --value-config "$VALUE_CONFIG" \
  --holdout-count "$VALUE_HOLDOUT_COUNT" --output "$RUN/prepared"
```

`split_report.json` records whole-episode membership and Z=max_train(length−1). A held-out return outside[-1,0] produces an offending-episode report and an error. Do not silently clip it or enlarge Z using held-out data. The paired Value control file must match the requested training budget/optimizer/selection rule; there are no guessed formal defaults.

`source_inventory.json` records the pinned dataset revision, metadata/Parquet hashes and video paths/sizes. Preparation verifies the actual source frame indices, intervention columns and camera mapping before any Value model is initialized. Video payload bytes are not rehashed by this inventory; export performs real frame/alignment checks. Source identities in the scope and final-policy-test list must refer to the same original recording identities across dataset copies/revisions.

2. Review the prepared configuration before starting GPU work. After that review, train and select the declared Value checkpoint:

```bash
python -m torch.distributed.run --standalone --nproc_per_node="$VALUE_GPUS" \
  --module lerobot.scripts.lerobot_sft_rl train-value \
  --prepared "$RUN/prepared/prepared.json" --output "$RUN/value"
```

`selected.json` identifies the predeclared final or minimum-holdout-MAE checkpoint. Resume uses `--resume /path/to/the/last/checkpoint.pt` with the same prepared file and output; source, stack, world size and last-checkpoint hash must match. It does not rewind over later checkpoints.

3. Set VALUE_CHECKPOINT to the file named by `selected.json`, then infer on complete allowed trajectories:

```bash
python -m lerobot.scripts.lerobot_sft_rl infer-value \
  --prepared "$RUN/prepared/prepared.json" --checkpoint "$VALUE_CHECKPOINT" \
  --device cuda --batch-size "$VALUE_INFER_BATCH" --output "$RUN/predictions.json"

python -m lerobot.scripts.lerobot_sft_rl select \
  --prepared "$RUN/prepared/prepared.json" --predictions "$RUN/predictions.json" \
  --output "$RUN/selection.json"

python -m lerobot.scripts.lerobot_sft_rl export \
  --contract "$CONTRACT" --selection "$RUN/selection.json" \
  --repo-id local/pure-demo-high --output "$RUN/high-export" \
  --output-horizon "$POLICY_HORIZON" --preview-count 3

python -m lerobot.scripts.lerobot_sft_rl report \
  --metrics "$RUN/value/metrics.jsonl" --predictions "$RUN/predictions.json" \
  --selection "$RUN/selection.json" --output "$RUN/plots"
```

Ranking is global across all nonterminal origins. First take ceil(10%×N), then requireA>0; tied values use stable episode/frame order. Each anchor retains[t,t+20), truncated at the original endpoint; overlap and adjacency merge. A crop endpoint is never labeled successful. An empty positive selection is reported and policy training is refused.

4. Generate the policy configuration from the actual resolved counterpart `train_config.json`:

```bash
python -m lerobot.scripts.lerobot_sft_rl prepare-policy \
  --control "$POLICY_CONTROL" --control-run "$POLICY_CONTROL_RUN" --contract "$CONTRACT" \
  --selection "$RUN/selection.json" --export-validation "$RUN/high-export/export_validation.json" \
  --sft-checkpoint "$SHARED_SFT_CHECKPOINT" --sft-sha256 "$SHARED_SFT_SHA256" \
  --high-fraction "$REVIEWED_HIGH_FRACTION" --global-batch "$CONTROL_GLOBAL_BATCH" \
  --mixed-precision "$CONTROL_PRECISION" --normalization-source "$CONTROL_NORMALIZATION_SOURCE" \
  --output-dir "$RUN/policy" --config-output "$RUN/policy_config.json"
```

When normalization_source=control_dataset, also provide`--stats-sha256` for the canonical reviewed dataset statistics. With checkpoint normalization, the saved SFT normalizer/unnormalizer is loaded without replacing its statistics. Both choices require matching the counterpart protocol.

**Review `policy_config.review.json` before launching.** It includes shared SFT identity, data/version, Value provenance and Z, n50/Top10%/20-window settings, source fractions, updates, global batch, optimizer/scheduler, trainable flags, horizon, precision, output and save policy. It also contains the exact Accelerate command. The new training entry checks the effective configuration against the frozen control; it does not silently accept CLI changes to the scientific settings.

5. After the reviewed native policy run completes:

```bash
python -m lerobot.scripts.lerobot_sft_rl verify-policy \
  --checkpoint "$POLICY_CHECKPOINT" --contract "$CONTRACT" \
  --device cuda --batch-size "$VERIFY_BATCH" --seed "$CONTROL_SEED" \
  --output "$RUN/policy_reload_inference.json"
```

This verifies reload, action shape and finiteness on training-source sanity inputs. It is not a final policy test or a success-rate estimate. Each saved policy checkpoint carries `sft_rl_protocol.json` alongside native model/processors/train config.

`POLICY_CHECKPOINT` means the directory containing `model.safetensors`, normally `checkpoints/<step>/pretrained_model`. `source_code.json` records source hashes and Git identity; runtime library versions are recorded in the run/checkpoint provenance.

## Real robot acceptance remains separate

Use the same paper task, initialization, time limit, camera mapping, RTC and scoring rules. Record actual trials with checkpoint SHA and the canonical hash of `sft_rl_protocol.json`. `robot_protocol.template.json` deliberately leaves the paper-specific TP definition empty; it is not inferred from the wordTP.

```bash
python -m lerobot.scripts.lerobot_sft_rl robot-report \
  --protocol "$ROBOT_PROTOCOL" --trials "$MEASURED_TRIALS" \
  --policy-checkpoint "$POLICY_CHECKPOINT" --output "$RUN/robot_results.json"
```

The aggregator reports stage successes/attempts/SR and the explicitly definedTP. It rejects offline metrics, mismatched checkpoints/protocols and duplicate trials. Do not fill the paper SFT+RL row until these real-robot records exist.

## Tests

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests/rl -p 'test_sft_rl*.py' -v
```

The tests use synthetic local videos and a tiny randomly initialized SigLIP fixture. They require no pretrained download, GPU, or robot. They test source guards, global selection, short chunks, actual Pi0.5 forward loss wiring, actual processor mask preservation, Accelerate sharding, control config decoding, checkpoint restoration and video/action/state alignment. Test fixtures and numbers are not experimental evidence.
