# PiperX HIL offline RL experiment

This directory records the first-stage PiperX copper-screw experiment run on
Shanghai Qizhi in September 2026. The implementation is based on upstream
Evo-RL commit `6f2db449a21e1bac750b996f2e27cac6739aa63f`.

The completed mainline is **Pi*0.6 value inference plus Advantage-Conditioned
Policy (ACP) training of a Pi0.5 actor**. It is not an IQL double-Q/value or
continuous advantage-weighted flow-matching implementation.

## Pipeline

1. Audit the clean 76-episode LeRobot v3 HIL dataset and split by episode or
   collection into 60 train, 8 validation, and 8 held-out comparison episodes.
2. Train the official `pistar06` distributional value model for 8,000 steps on
   the 60 HIL training episodes.
3. Run value inference over all 76 HIL episodes. Compute 50-step advantages,
   mark the top 30% per task positive, and force intervention frames positive.
4. Build a policy replay view with 558 base episodes plus the 60 HIL training
   episodes. Freeze the base normalization statistics and sample 25% HIL / 75%
   base with deterministic weighted replacement.
5. Start both policy branches from the exact same Pi0.5 full558 step-50000
   checkpoint:
   - ACP: inject `Advantage: positive` or `Advantage: negative` into HIL task
     text; leave base replay untagged; use 30% tag dropout.
   - Data-only: disable ACP and ignore value, advantage and intervention labels.
6. Train matched 10k and 20k policy pairs with global batch 64. The 10k pair
   used 4 H100s; the independent 20k pair used 8 H100s.
7. Compare checkpoints on the locked eight-episode offline comparison set, then
   use randomized real-robot evaluation for effectiveness claims.

At ACP inference, append the positive condition to the task:

```text
Insert the copper screw into the black sleeve.
Advantage: positive
```

## Repository changes

The code changes are applied directly under `src/lerobot`. They add:

- robust Pi*0.6 resume and DDP handling;
- forwarding of the selected video backend during value inference;
- an ACP apply mask so base replay remains untagged;
- deterministic base/HIL weighted replay sampling;
- explicit checkpoint step lists and finite policy-loss/gradient gates;
- direct PyAV stream seeking, deferred RGB conversion, and uint8 image transfer
  through DataLoader workers.

The final three changes address a measured input-pipeline bottleneck. The failed
configuration spent about 60.7 seconds assembling one global batch without model
compute; the fixed loader averaged about 1.24 seconds and 51.8 samples/s.

## Contents

- `configs/`: exact recorded experiment contracts. Paths are platform-specific
  provenance and must be adapted for another installation.
- `scripts/`: exact data audit, preparation, launch, supervision and offline
  evaluation programs used by the run. They intentionally preserve the recorded
  Qizhi directory layout and should be copied and parameterized before reuse.
- `reports/`: reward/value contract, preflight, completion and performance
  diagnosis.
- `patches/`: the nine byte-identical patches applied during the experiment.
  The same changes are already present in this branch's source tree.

No dataset, checkpoint, optimizer state, token, cookie, password or platform
credential is included.

## Scientific scope

The value target is an official normalized time-to-go/outcome target in `[-1, 0]`.
Because most successful episodes include human recovery and only seven episodes
are explicit failures, it must not be presented as a calibrated autonomous-success
probability. Offline action MAE is useful for checkpoint comparison but does not
replace real-robot success and intervention-rate evaluation.

IQL double Q/V, expectile regression, transition masks and continuous advantage
weights remain a separately named future extension.
