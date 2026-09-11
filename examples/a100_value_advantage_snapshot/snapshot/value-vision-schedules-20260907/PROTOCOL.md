# Pure-vision value: bounded learning-rate and fixed-horizon study

User authorized 2026-09-07. Scope: PiperX insertion only. No paper main.tex edits, no upstream LeRobot edits, no policy training, no other tasks. Preserve old five-way results.

## Gates and model

Remove only the unused SigLIP text tower and logit scale/bias from the old pure-vision model. Retain the identical vision tower, image projector, 512-wide head, preprocessing and 201-bin two-hot target. Export exact used pretrained tensors and head initialization under seed1000. Validate old step1500 against compact inference weights on real three-camera frames in FP32 and BF16, strict reload and gradients. No quantization. Formal runs begin from the initial fixture, NOT the old trained checkpoint.

Run two simultaneous four-GPU, batch8-per-GPU smoke tests, including forward/backward/AdamW/save/reload. Require both gates before formal launch. GPU allocations are disjoint; global batch remains32. User explicitly declined rerunning1500: use old1500 as reference, flag old8x4 versus new4x8 dropout/reduction differences. The global frame sequence remains seed1000 and identical across schedules. No new1500 formal run is authorized.

## Runs

NEW USER AUTHORIZATION after4500pause: run four SO101 purevision cross-task epoch comparisons after the existing GPU lanes finish, no sharing. Full instructions in cross-task-epoch/PROTOCOL.md; immutable plan cross-task-epoch/plan.json. Two CPU queue workers value-cross-cosine/value-cross-constant wait for respective parent completion markers AND free GPUs. Folding1491/2981steps,stationery1496/2991steps match PiperX1500/3000 effectiveepochs. No changes to existingPiperX training. Existing monitor must continue until these additional4runs are evaluated and summarized too.

LATEST USER OVERRIDE: do NOT run4500 now. It was briefly launched on shared GPU0-3 and logged step10 before the user declined GPU sharing. Stop only its launcher, retain all files, no automatic restart or queue after3000. Only constant8000 and cosine3000 remain active, each on disjoint4 GPUs. 6000 stays cancelled. run_lane.py cosine4500 is explicitly blocked pending new approval. The brief4500 run has no250-step checkpoint and is NOT a completed endpoint; its successful2-step smoke checkpoint is not a formal-training resume checkpoint. Earlier parallel4500 plan below is superseded.

Lane A GPUs0-3: 200-step warmup then constant LR5e-5 to8000, diagnostic long curve. Bounded, never infinite.
Lane B GPUs4-7: independent cosine schedule3000. Every run uses the SAME initial tensors and ends at LR1e-6; warmup200, peak5e-5. No continuation from a shorter trained run. Existing1500 remains the reference.

User amendment 2026-09-07: CANCEL6000 and replace it with independent4500, starting concurrently after a real shared-GPU smoke test. Lane C `run_lane.py cosine4500`, tmux=value-lean-cosine4500, GPUs0-3 shared with constant8000, unique port29733, same4x8/global32. Run name=piperx-visionlean-cosine-fixed4500-seed1000. Training source/architecture, initialization, data and optimizer are unchanged; only its cosine horizon is4500. GPU sharing changes wall-clock throughput, so do not use overlapped wall times for architecture-speed comparisons. Never re-run1500 or6000.

Safe live-controller handoff: old CPU queue PID1908829 had already loaded [3000,6000]. `retire_old_cosine_controller.py` in tmux=value-lean-cosine-finish intentionally SIGSTOPs ONLY that CPU queue while torchrun PID1908832 and all3000 workers continue untouched. It waits for that exact torchrun to exit, checks no active children, retires only the stopped queue, and invokes updated3000-only lane for strict final review. Inspect cosine-controller-handoff.json/log; do NOT resume the old stopped queue (it could enqueue6000). If3000 exits without completion, the handoff reports failure, does not restart. Current constant controller can continue unchanged (its8000 budget was not amended). The old runner and protocol are preserved as run_lane.before4500.py and PROTOCOL.before4500.md.

Shared: AdamW beta(.9,.999), eps1e-8, weight decay1e-5, clip10, FP32 parameters/optimizer with BF16 autocast, gradient checkpointing, dropout.1, global32, frame-uniform replacement sampler, no image augmentation, three384x384 cameras. Episode split502/56, original train-only normalization and target scale14882. All episodes eventually successful. Constant-LR long-run intermediate checkpoints are not substitutes for shorter complete cosine schedules.

## Evaluation and selection

Every250steps save full resumable checkpoint and evaluate the same56 training episodes x32 frames plus56 held-out episodes x32 frames. Final strict reload evaluates56 held-out x128frames. Report train/held-out CE, MAE, RMSE, CRPS, bias, nominal90% coverage/width; diagnostics of temporal difference and ordering can reuse saved predictions. Allcheckpoints retained. Approx320GB storage for62 formal checkpoints is within observed5.5TB free.

Primary comparison: final held-out episode MAE of each independent cosine schedule. Pairwise episode bootstrap95% CIs, CRPS/RMSE and temporal diagnostics guard against selecting only CE minima or noisy value differences. Prefer smaller budget if differences are within uncertainty and no meaningful auxiliary improvement. One seed and one held-out split are pilot evidence, not a global optimum. Because the10% split is repeatedly used for selection, call it validation in the paper; it is not an untouched final test.

Overfit diagnosis: training error falls while held-out error increases or calibration degrades. A generalization gap or a CE plateau alone does not prove the scalar predictions are unusable. Do not predeclare1500or3000 optimal. Safety failures/NaN/OOM stop only the affected lane, preserve evidence; resume from its own recent checkpoint only after diagnosis. No auto-expansion beyond these three formal runs.

## Paper use

Current paper Methods explicitly leaves value-architecture selection TBD. The five-way1500 comparison and this budget study can support a small value-design subsection/appendix. Report all five variants, including the full baseline, even if not selected for future training. Compression is an equivalence/efficiency result, not a new predictive architecture contribution. These success-demo return-prediction experiments cannot populate the main BC+RL policy SR/TP cells; downstream advantage labels and real-robot outcomes remain necessary.

Before future manuscript edits reconcile current one-hot description with actual two-hot targets, success-demo target scaling with intervention-cost targets, and paper's generic8x80GB hardware statement with this A10040GB ablation. No paper body modifications made in this task.
