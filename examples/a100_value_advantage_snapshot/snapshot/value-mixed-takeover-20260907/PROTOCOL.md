# PiperX mixed full-trajectory value pilot — preparation, not launched

User approved successful-demo + complete-HIL 1:1 training with takeover penalty.
User 2026-09-07 says the 75 trajectories from 4090A should all be successful;
record this as user-confirmed/provisional outcome provenance, not existing metadata.
The 66 HF-source trajectories already have success metadata. No failure outcomes
are fabricated. Train only a pure-vision value; no policy training in this pilot.

## Proposed frozen protocol

- Fresh existing seed1000 text-free SigLIP value initialization; 201 bins [-1,0],
  two-hot cross entropy, FP32 parameters and BF16 autocast, gradient checkpointing.
- 4 GPUs × per-GPU batch8 = global32, exactly 4 demo + 4 complete-HIL frames per
  GPU. Within each training pool, uniformly sample frames with replacement.
- 1500 optimizer steps; warmup200, cosine peak5e-5 to1e-6; save/evaluate every250.
- Preserve existing Demo558 split502/56. HIL141 split127/14 by stable source
  identity, not by frame or intervention segment; never split a source episode.
- HIL identity is source_episode_uuid if present, otherwise
  source_dataset + ':' + source_episode_index. Rank ascending SHA256 of
  'value-mixed-takeover-split-v1:1000:' + identity; first14 held out.
- HIL held-out output episode IDs:
  [11,17,18,22,35,42,51,53,82,91,114,118,121,136].
  Audited training809459 frames / held-out95789. Training source counts HF60,
  4090A67; held-out HF6,4090A8. Cross-pool content-duplicate audit still required.
- Nonterminal reward -1; successful terminal0. Extra -150 only on the preceding
  autonomous frame of a 0→1 human-control transition. Initial human control is
  not a takeover. Terminal boundary excluded per current manuscript convention.
- Raw return = -(remaining transitions + 150 × remaining takeovers). Gamma1.
- Audited616 takeovers, zero terminal-boundary exclusions. Maximum HIL training
  cost15337 (episode43 length13088 with15 takeovers). Max held-out cost12316.
- User explicitly corrected normalization: Z=max training cumulative cost=15337,
  multiplier=1 (NOT 2), pending final combined-data verification. Demo max training
  length7441 does not enlarge it. Fit using TRAIN only, shared across both sources
  and all checkpoints. The minimum training target is -1, successful terminal0.
  Max held-out cost12316 fits the support. Any out-of-support target must fail
  explicitly; never silently clip labels or enlarge Z using held-out data.
- One event is -150/15337=-0.00978026993545, about0.978% of full value range and
  1.956 bin intervals. C150 means5 seconds at30fps; this is a pilot cost preference,
  NOT a literature optimum. Costs accumulate across repeated takeovers.
- Report CE/MAE by demo/HIL and HIL source, as well as 50:50 aggregate. MAE in
  cost-equivalent seconds is not literal remaining-time error once penalty exists.
  New CE/normalized MAE are not directly numerically comparable to old Z14882 runs.

## Advantage validation required

Evaluate held-out autonomous frames with n=50 (plus sensitivity windows15/30),
terminal bootstrap zero. Keep full A, time-only reward + value delta, and explicit
takeover cost as separate outputs. A_full=A_time_delta-C*N_window/Z.
This inference decomposition is NOT a no-penalty training group.
Report pre-takeover negative fractions, matched non-takeover false alarms,
event recall and PR/AUROC with episode-level uncertainty. Event proximity is a
proxy label, not proof every earlier action is bad. Do not select C just to force
all pre-event A negative. Low Monte Carlo return MAE alone does not validate A:
perfect reproduction of each realized return would yield zero n-step residual.
No claim that policy improvement is demonstrated without downstream policy tests.

## Current execution state

### Latest override — 2026-09-07 direct v4 authorized and running

The user explicitly instructed autonomous completion after the temporary-key
question. Authorization is now granted; do not keep waiting for permission.
4090A tmux `value-direct-nonhf-v4` runs WORK/direct_nonhf_v4.py send;
WORK=/home/zhaobo/value-direct-transfer-20260907, log transfer-v4.log.
Only direct rsync to A100, no proxy/jump host/Mac relay. HF198 files are already
SHA-verified complete. NonHF38 files/3106109840 bytes verified; remaining24 files/
3368881768 bytes are transferring using immutable direct-v4-plan.json.
Single-connection throughput fell to20–30KB/s with many TCP retransmissions.
After stopping that sender and checking both ends, source send was changed to
four direct connections with disjoint missing[shard::4] lists. Progress logs are
WORK/transfer-v4-shard0.log through shard3.log. No additional network hop.
Temporary key WORK/key-v4 uses A100 marker value-direct-nonhf-v4-20260907,
restrict + rrsync -wo -no-del -no-lock scoped to the destination dataset. The
no-lock option permits the disjoint shards, not wider filesystem access. Sender removes
key-v4 and key-v4.pub after receipt upload. A100 value-direct-verify-v4 runs
ROOT/direct_nonhf_v4.py verify, checks all62 SHA values and removes only this
marker before publishing relay_nonhf_verified.json. Original unified260-SHA
verifier and preflight/smoke/1500 queue stay unchanged. Do not restart healthy
writers or any old relay. Preserve all files and parts. The following acquisition
bullets describe historical v3 state and are superseded by this override.

- Latest user-approved acquisition is split by byte-identical source. Old
  value-mixed-relay-v2 and both remote v2 processes were deliberately stopped.
  Never restore old full-stream/v2/direct jobs. Direct SSH keys remain removed.
- Full immutable manifest: direct_source_manifest.json,260files,249MP4,
  13944913650bytes. hf_routes.json pins 198 exact-SHA video mappings,
  7469922042bytes, repo Travor278/evorl-piperx-copper-screw-hil-clean,
  revision18016184b09929643b6bd055b3f5833bfb2e7b85 via hf-mirror.com.
- A100 tmux value-hf-acquire-v3 runs hf_acquire_v3.py download with
  /data/envs/lerobot-0.6.1/bin/python; hf-acquire-v3.log. Four workers,
  Range-resumable .hf-v3.part; existing SHA-valid files skipped. Destination
  paths come from output mapping, not original HF camera/chunk names.
  Completion marker hf_verified.json does NOT release training.
- Disjoint complement62files/6474991608bytes uses Mac tmux
  value-nonhf-relay-v3 and local relay_nonhf_v3.py relay; relay-nonhf-v3.log.
  Source helper /home/zhaobo/value_relay_nonhf_v3_20260907.py; A100 helper
  ROOT/relay_nonhf_v3.py. Missing-only plan relay_nonhf_plan.json.
  At split switch,28files/2072717168bytes already valid;34files/4402274440
  bytes remain on this route. Final relay_nonhf_verified.json is subset-only.
- A100 tmux value-acquire-verify-v3 runs hf_acquire_v3.py verify, log
  acquire-verify-v3.log. Waits BOTH subset markers, checks immutable manifests,
  then rehashes all260 files before publishing original transfer_verified.json.
  Original preflight/smoke/1500step experiment queue is unchanged.
- No overlapping writers, no deletion of any data/part/rsync partial, no
  overwriting mismatched existing files. Healthy routes stay running; only
  genuinely failed routes restart after checking own processes have stopped.
  Mac must stay online for nonHF route. Report sustained observed speed only.
- A100 experiment directory /data/experiments/value-mixed-takeover-20260907.
- Metadata preflight now passed: train Demo1132703/HIL809459 frames, held-out
  Demo123026/HIL95789, no exact whole-trajectory action/state SHA duplicates.
  Reward-boundary and balanced deterministic sampler tests passed. Transfer and
  all-episode video sample decoding still pending; no mixed training started.
- USER STOP 2026-09-07: all old 4+4 GPU experiments and both cross-task queues
  deliberately terminated on request, confirmed GPUs0-7 memory0MiB/util0%.
  Do NOT resume old queues/experiments or wait for their completion markers.
  PiperX constant last saved7000; new stationery short had no formal checkpoint.
  Preserve all old results.
- 2026-09-07 11:51: new train_mixed.py, evaluate_advantage.py and run_experiment.py
  implemented/deployed. Syntax/import tests, reward endpoints, two-hot support,
  cosine LR endpoints, realized-return telescoping identity, tied/perfect ROC/AP,
  autonomous-only evaluation origins and event-window alignment tests passed.
- A100 tmux value-mixed-pilot is running run_experiment.py and active.json says
  waiting_for_verified_transfer. Pipeline: transfer SHA completion → full preflight
  → four-GPU actual 2-step smoke/save/reload → fresh1500 training →128-frame-per-
  episode train/held-out review → final-checkpoint held-out advantage review.
  GPU0-3 only, perGPU8/global32; no cancelled experiment is a dependency. No mixed
  GPU training or GPU smoke has happened yet; never claim otherwise before logs.
- Advantage plan has57378 windows at15/30/50 frames and22676 unique nonterminal
  image inputs from14 held-out HIL episodes. ROC/AP uses only fixed every5th
  autonomous frames (natural grid prevalence); extra dense pre-event frames are
  retained in records for event plots, not included in classification scores.
  300 episode bootstrap resamples; A<0 threshold fixed; final1500 first. Comparing
  intermediate checkpoints on the SAME advantage set remains a later analysis.
