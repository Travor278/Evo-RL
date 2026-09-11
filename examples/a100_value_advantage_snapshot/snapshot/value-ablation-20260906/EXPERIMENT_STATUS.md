# COMPLETED — all five fixed1500 variants — 2026-09-07

All five variants finished step1500, saved final checkpoints, and completed strict-reload dense128 evaluation. FULL1500_COMPLETION.json is present; reporter confirms formal_runs=5, evaluations=30, dense_endpoints=5. A100 has no active GPU compute processes. Final Chinese plots were generated and visually checked. Current report: COMPLETE_COMPARISON.md; do not use the historical mixed-budget FINAL_REPORT.md for this five-way comparison. No additional training is authorized. The old value automation was deleted; no completion-only replacement was created. Preserve all results and do not restart any queue.

## Superseded completion authorization — 2026-09-06

The user has now explicitly asked to finish the missing full-input variant and redraw a COMPLETE, FAIR five-variant graph with unambiguous legends. This overrides the stop below ONLY for full/fixed1500. Do not resume the broad run_pipeline.py main queue, fixed2500/3000, seed repeats, old8000 pilot, fold, or stationery.

Use complete_full_only.py in tmux value-full1500-completion; log full1500-completion.log. It uses the existing full-fixed1500-seed1000-v1 run, restores model/AdamW/per-rank RNG from checkpoint250 with --resume, keeps the exact same split, train-only stats, global32, seed1000, BF16 autocast, warmup200, peak5e-5/floor1e-6 cosine ending at1500, then strictly reloads FINAL1500 for dense128 review. Existing checkpoints and metrics remain preserved. The training implementation is unchanged (SHA256 be2837bfb0f219dca19478b982608d87ddc4739ed778d52fdda426cd84341ad6).

Only completion marker FULL1500_COMPLETION.json plus final.json step1500 and dense-review-001500-frames128.json strict_ok establishes completion. After completion copy freshly generated reports to Mac, render Chinese legends with plot_complete_comparison.py, verify images, deliver five NEW fixed1500 endpoints and curves (never substitute old full3000), then delete the completion-only monitor. Normal progress does not require restarts. If failure, inspect before resuming the same run/config. Mac artifacts: /Users/elvin/.codex/visualizations/2026/08/28/01a04880-fb8c-7e80-ac49-3813ce17c051/value-ablation-20260906.

## Superseded stop record — history, not current full1500 authorization

# STOPPED BY USER — 2026-09-06

The user explicitly ended further experiments and requested a final comparison using existing results. Do not restart either pipeline or any pending fixed2500/fixed3000, seed-repeat, fold, or stationery runs. Automation `value` was deleted. Scheduler PID499771 was terminated and active torchrun PID2234507 was interrupted; all eight workers exited and all GPUs were verified at 0MiB/0% utilization. Stop-related SIGINT/SIGHUP tracebacks are intentional, not a failure to recover.

Four fixed1500 variants completed training and strict-reload dense final evaluation: frozen_both, frozen_language, task_only, vision_only. The fifth full-input fixed1500 run was stopped with last logged optimizer step460; its last complete checkpoint and evaluation are step250. It is NOT a completed fixed1500 result. Fixed2500/fixed3000 runs never started. Preserve all existing checkpoints, logs, data, models and caches. CPU-only report generation is allowed; no further training is authorized by the old plan below.

Practical recommendation from completed fixed endpoints: vision_only final step1500 (simpler, lowest observed MAE/RMSE/CRPS among four completed variants). Task_only is close and not significantly worse in paired episode MAE. Frozen_language has lowest CE. This is not a proven five-variant optimum. Separately, the old full-input 8k-schedule pilot has a balanced checkpoint at3000, not evidence that a fresh fixed3000 schedule was tested.

## Historical plan below — superseded; DO NOT execute

# Previous priority: five architectures and fixed total step selection

User explicitly requested actual comparative experiments, not continuing the old8k/early-stop pilot. Old pipeline was intentionally stopped after checkpoint006000 was safely written. Its tmux/processes exited; no checkpoint, dataset, log or upstream source was removed. SIGINT at the end of its log is intentional.

New root: /data/experiments/value-ablation-20260906

## Launch verification snapshot (2026-09-06 14:53 China)

All five eight-GPU two-step smokes completed successfully: actual forward/backward, expected frozen/non-frozen encoder gradients, strict save/reload exact logits. See smokes_complete.json. State probe also completed: original step3000 MAE0.01217783095, neutral-state0.01217859655, permuted-state0.01217795746. Paired images/labels identical; a sampled neutral/permuted prompt changed64/54 tokenizer positions respectively. Mean absolute prediction shift was1.8277e-5 /1.9369e-5. This model is insensitive to these interventions on this split; it does not establish that no-state retraining wins.

Formal fixed1500 frozen_both run is now genuinely training, observed step10 loss5.3794279 / grdn3.4498417 / lr2.5e-6 / effective_epoch0.0002825101, with no formal held-out evaluation yet. This is startup evidence, NOT convergence performance. Current state must always come from active.json, actual logs and metrics, not this historical snapshot.

- Queue: tmux value-ablation-pipeline, run_pipeline.py, pipeline.log, active.json.
- Runner: ablation_train.py; original upstream LeRobot/Evo files are unchanged.
- State diagnostic: state_probe.py, existing original step3000, no training mutation.
- Complete scientific plan: PROTOCOL.md. Formal stop is EXACT fixed horizon; endpoint model primary.
- CPU preflight passed: three camera inputs, PiperX14-dimensional state, task-only prompt constant across frames and without State, vision-only prompt empty, same decoded images and labels, reproducible sampled frames.
- Pipeline first checks every variant with two-step eight-GPU forward/backward/frozen-gradient/save/reload smoke, then state probes, then all five fresh1500-step runs. Dense128 final review; provisional two leaders each fresh2500 and3000. Single-seed staged screening is not a full factorial or a global optimum claim. User wants a final comparison table and train/held-out Loss+MAE plots; evaluate whether extra controls are needed before concluding.
- Never restart the old value-demo3-pipeline automatically. Old extended probability/50-frame reviews remain pending and preserved; do not claim they ran.
- Normal queue progress does not need restarts. Run_pipeline.py checks GPU idle before each launch and fails closed. Resume only the same schedule and seed from existing last.json. A run failed before first checkpoint needs inspection and a new clearly labelled output, not deletion/overwrite.
- Environment: /data/experiments/value-demo3-20260906/venv-py312-torch210-cu128; Python3.12/torch2.10.0+cu128/LeRobot0.6.1. Models and manifests/quantiles reused read-only from value-demo3-20260906.
- Reports: run this environment's python analyze_experiments.py (CPU-only). It creates reports CSV/JSON/Markdown and paired-episode metrics. Matplotlib is deliberately not added to this locked training environment; copy reports/curves.json and this script to Mac and use cached uv with numpy/matplotlib to render --plot-only curves.json --out REPORTDIR.
- Monitor id=value continues every15 minutes, but now follows the new fixed-horizon experiments. Quiet while healthy; notify substantive results/failure/completion. No HF upload/PTS/policy work.

Old6000 heldout32 metrics: CE3.3612558701 / MAE0.0122011843 / RMSE0.0201226431; old3000 CE2.636523 / MAE0.012178295. Neither is equivalent to a freshly trained shortened LR schedule.
