# Pure-vision value: bounded learning-rate and fixed-horizon study

User authorized 2026-09-07. Scope: PiperX insertion only. No paper main.tex edits, no upstream LeRobot edits, no policy training, no other tasks. Preserve old five-way results.

## Gates and model

Remove only the unused SigLIP text tower and logit scale/bias from the old pure-vision model. Retain the identical vision tower, image projector, 512-wide head, preprocessing and 201-bin two-hot target. Export exact used pretrained tensors and head initialization under seed1000. Validate old step1500 against compact inference weights on real three-camera frames in FP32 and BF16, strict reload and gradients. No quantization. Formal runs begin from the initial fixture, NOT the old trained checkpoint.

Run two simultaneous four-GPU, batch8-per-GPU smoke tests, including forward/backward/AdamW/save/reload. Require both gates before formal launch. GPU allocations are disjoint; global batch remains32. User explicitly declined rerunning1500: use old1500 as reference, flag old8x4 versus new4x8 dropout/reduction differences. The global frame sequence remains seed1000 and identical across schedules. No new1500 formal run is authorized.

## Runs

Lane A GPUs0-3: 200-step warmup then constant LR5e-5 to8000, diagnostic long curve. Bounded, never infinite.
Lane B GPUs4-7: independent cosine schedules3000,6000, sequentially. Every run uses the SAME initial tensors and ends at LR1e-6; warmup200, peak5e-5. No continuation from a shorter trained run. Existing1500 remains the reference.

Shared: AdamW beta(.9,.999), eps1e-8, weight decay1e-5, clip10, FP32 parameters/optimizer with BF16 autocast, gradient checkpointing, dropout.1, global32, frame-uniform replacement sampler, no image augmentation, three384x384 cameras. Episode split502/56, original train-only normalization and target scale14882. All episodes eventually successful. Constant-LR long-run intermediate checkpoints are not substitutes for shorter complete cosine schedules.

## Evaluation and selection

Every250steps save full resumable checkpoint and evaluate the same56 training episodes x32 frames plus56 held-out episodes x32 frames. Final strict reload evaluates56 held-out x128frames. Report train/held-out CE, MAE, RMSE, CRPS, bias, nominal90% coverage/width; diagnostics of temporal difference and ordering can reuse saved predictions. Allcheckpoints retained. Approx350GB storage for68 formal checkpoints is within observed5.5TB free.

Primary comparison: final held-out episode MAE of each independent cosine schedule. Pairwise episode bootstrap95% CIs, CRPS/RMSE and temporal diagnostics guard against selecting only CE minima or noisy value differences. Prefer smaller budget if differences are within uncertainty and no meaningful auxiliary improvement. One seed and one held-out split are pilot evidence, not a global optimum. Because the10% split is repeatedly used for selection, call it validation in the paper; it is not an untouched final test.

Overfit diagnosis: training error falls while held-out error increases or calibration degrades. A generalization gap or a CE plateau alone does not prove the scalar predictions are unusable. Do not predeclare1500or3000 optimal. Safety failures/NaN/OOM stop only the affected lane, preserve evidence; resume from its own recent checkpoint only after diagnosis. No auto-expansion beyond these three formal runs.

## Paper use

Current paper Methods explicitly leaves value-architecture selection TBD. The five-way1500 comparison and this budget study can support a small value-design subsection/appendix. Report all five variants, including the full baseline, even if not selected for future training. Compression is an equivalence/efficiency result, not a new predictive architecture contribution. These success-demo return-prediction experiments cannot populate the main BC+RL policy SR/TP cells; downstream advantage labels and real-robot outcomes remain necessary.

Before future manuscript edits reconcile current one-hot description with actual two-hot targets, success-demo target scaling with intervention-cost targets, and paper's generic8x80GB hardware statement with this A10040GB ablation. No paper body modifications made in this task.
