# Attempt-aware Value failure LOO gate v1

Status: preregistered before inference on the original test failure (attempt episode 481).

## Purpose

Replace the unstable single-validation-failure ranking gate with a seven-fold,
group-safe leave-one-failure-out (LOO) gate. This protocol is diagnostic and
selection-bearing: the original train/validation/test labels are consumed by the
cross-validation procedure and the original test split is no longer an untouched
held-out test.

## Frozen inputs

- Outcome manifest SHA-256: `d7d20b68032e5237a4bbc2e08d531dafcbd19e1e4b7e3eddb09fbb9534139a79`
- Fold manifest SHA-256: `d91d4e408c1556fc654834b944a0308fe4ebc9fa378e9f48d95420563f4f94a5`
- Failure attempts: 454, 459, 471, 472, 474, 475, 481.
- Seed: 20260906.
- Grouping field: `source_collection_id`; physical-episode leakage is also
  impossible because every physical episode belongs to exactly one collection.
- Value data: attempt-aware known-outcome attempts only. Base 558 remains excluded.

## Fold construction

For each failure attempt, exclude its complete collection from training. Also
exclude one failure-free success-control collection. Success controls are matched
without model predictions, using this fixed score in ascending order:

1. current number of uses of the candidate control collection;
2. same `source_dataset` preferred;
3. a success attempt with the same `logical_attempt_index` preferred;
4. minimum absolute log-duration difference;
5. SHA-256 tie-break of seed, failure episode, and collection id.

Only three leakage-safe old success-control collections exist, so they are reused
in a balanced 2/2/3 pattern. Bootstrap resampling is therefore performed over the
seven folds, never over repeated success frames or control attempts.

## Training and evaluation

- Seven independently initialized Pi*0.6 Value models.
- Attempt-uniform mainline sampler; no outcome oversampling.
- All known-outcome attempts outside the two held-out collections form that fold's
  training set.
- Same initial weights, optimizer, normalization, global batch 64, bf16, seed, and
  1,000-step diagnostic budget for every fold.
- Each fold is evaluated only on its one held-out failure and all known successes
  in the two held-out collections.
- Primary per-fold statistic:
  `mean(success_control_end_value) - failure_end_value`.

## Frozen pass rule

The LOO gate passes only if every condition is true:

1. All seven folds and predictions are finite.
2. At least five of seven fold separations are strictly positive.
3. Median fold separation is strictly positive.
4. Mean fold separation is strictly positive.
5. Pooled out-of-fold success-vs-failure rank AUC is greater than 0.5.

Report, but do not gate on, the fold-bootstrap 95% confidence interval of the mean
separation because seven failures cannot support a stable CI-based acceptance
threshold. Bootstrap uses 10,000 resamples and seed 20260906.

If the gate fails, formal Value 8k, Advantage generation, and ACP policy training
remain blocked. If it passes, train the final 8k Value model on all known-outcome
attempts with the attempt-uniform sampler; report that no untouched offline test
failure remains and do not present the cross-validated result as held-out test
performance.
