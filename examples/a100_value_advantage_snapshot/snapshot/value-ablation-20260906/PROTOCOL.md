# Fixed endpoint value ablations — 2026-09-06

Question: choose architecture and a predeclared total step budget. Every new formal run stops exactly at its fixed horizon, NOT at a selected intermediate checkpoint. Intermediate evaluation is diagnostic only. Failures stop the queue.

## Controlled conditions

PiperX successful human demonstrations, unchanged episode-disjoint train 502 / held-out 56 split and train-only quantiles from value-demo3-20260906. Three RGB cameras, identical image preprocessing and two-hot 201-bin countdown targets. The held-out partition is used for model selection and is therefore a validation set, despite legacy filenames saying test. It is not an untouched final test.

Same local SigLIP and Gemma pretrained weights, seed1000, stateless uniform frame sampling with replacement, per-GPU batch4 × 8 GPUs = global32. Native DDP, FP32 parameters/AdamW states with BF16 autocast, gradient checkpointing, peak LR 5e-5, floor 1e-6, weight decay 1e-5, clip10, warmup200 steps. Cosine decay reaches floor at each run's own final step. Original LeRobot/Evo-RL source and datasets remain unchanged.

## Five variants

| Variant | Images | Language input | Trainable encoders |
| --- | --- | --- | --- |
| full | three cameras | task + frame-varying state tokens | vision + Gemma |
| vision_only | three cameras | none; Gemma removed, no replacement state MLP | vision |
| frozen_language | three cameras | task + state | vision only |
| frozen_both | three cameras | task + state | neither; projections and value head train |
| task_only | three cameras | task text only; no state | vision + Gemma |

Pure vision changes fusion/head width (512 rather than 1024) and removes both task/state; it is not a single-factor text-only ablation. Task-only versus full isolates the state input while keeping the architecture. Frozen encoders stay in eval mode and have no gradients. Unused SigLIP text parameters exist in the pretrained container as in the original baseline but do not receive gradients.

## Staged fixed-budget plan

1. Each variant: two-step actual eight-GPU smoke checking video loading, backward, intended encoder gradients, save and strict reload/equal logits.
2. Existing step3000 baseline: paired unmodified / neutral-state / marginally shuffled-state inference on identical holdout images. These probes measure reliance and can be OOD; the task-only retraining tests whether state is useful.
3. Five variants each freshly initialized for a fixed1500 run. Evaluate/save every250 and final. Dense128 frames/episode review of FINAL1500.
4. Provisional two leaders by final episode-MAE (RMSE tie-break), each freshly initialized for fixed2500 and fixed3000. Do not continue a1500 run into a new schedule. Screening may miss slow-starting variants; report this limit.
5. Compare FINAL endpoints using MAE, RMSE, CE, CRPS, bias, predicted90% coverage/width, entropy, episode/time-position errors, ranking and value-difference errors. Paired episode bootstrap uncertainty; small differences do not establish a winner. For near ties, repeat seeds before a strong recommendation. Report training time too.
6. Use winning recipe for follow-up fold and stationery checks; a PiperX winner is not automatically the winner for other tasks. Preserve the old pilot's outputs and pending extended review.

No exact final winner or step equivalence is promised ahead of data. Countdown labels from all-success trajectories measure remaining time, not independently labelled physical progress or failure discrimination.
