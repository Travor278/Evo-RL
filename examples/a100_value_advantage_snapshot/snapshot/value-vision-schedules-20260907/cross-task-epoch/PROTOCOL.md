# Cross-task epoch-matched pure-vision value pilot

User authorized overnight cross-task validation, interpreting their spoken "11500" as the previously discussed1500 (explicitly disclosed). Compare PiperX1500 and3000 against the SAME effective epochs on the existing duration-matched SO101 folding108 and stationery397 episode subsets. No new1500PiperX, no4500/6000, no policy runs, no paper edits, no upstream source/environment edits, no data/checkpoint deletion.

## Fixed budget before looking at results

Effective epoch = optimizer steps ×32 / TRAIN frames; uniform frame sampling WITH replacement. It counts sampled examples, not unique coverage or full trajectories. It is not episode count. PiperX502train episodes contain1132703frames; target epochs1500×32/1132703=0.04237651 and3000×32/1132703=0.08475302.

Folding97train/11heldout episodes,1125359train frames:1491 and2981steps. Warmup199, eval/save every249. Stationery357train/40heldout episodes,1129226train frames:1496 and2991steps. Warmup200, eval/save every250. Formula ceil(reference_steps×task_train_frames/1132703); warmup and eval interval also scaled from200 and250 in epoch space. Ceiling introduces <1step mismatch. Similar totals are expected: data were duration matched.

## Training controls

Four independent fresh runs from SAME initial seed1000 SigLIP-vision/head fixture, not continuation from short endpoints. Pure vision only (no task/state model inputs); same201bins/two-hot,3cameras384x384,AdamW,peak5e-5,cosine floor1e-6 at finalstep,global32=4GPUs×8,BF16autocast/FP32parameters,gradient checkpointing,dropout.1, no augmentations. Unmodified parent train_lean_value.py is reused through a namespace-only entrypoint; assets is a symlink to parent verified initialization, no copying/replacing trained weights.

Keep existing seed1000 episode-disjoint90/10 split and archive-selected subsets; no redraw based on performance. Use existing TRAIN-only quantile files. Value scale remains2×max TRAIN episode length (PiperX14882,fold35886,stationery17936), not heldout fitted. State is loaded by the shared dataset code but NOT fed to the purevision model. Camera order follows each immutable manifest; mean camera pooling is permutation-invariant in eval, do not modify source data. Allselectedepisodes were user-confirmed finallysuccessful.

## Scheduling and checks

No shared GPUs. Two CPU queue tmux sessions value-cross-constant and value-cross-cosine wait for parent constant-complete.json or cosine-complete.json, which requires original final checkpoint AND strict dense review. Queue first checks group memory<1000MiB on each GPU. Lane constant uses0-3, cosine4-7, each4processes; file locks prevent duplicate job claims. Fourjobs order foldshort,foldlong,stationeryshort,stationerylong; first free lane works, other may pick remaining unlocked work after its own dependency completes. Waiting is expected. Do not kill original3000/8000 or SIGCONT the intentionally stopped old cosine CPU controller. Preserve parent controller-handoff.

CPU preflight verifies immutable manifests, frame counts, disjoint splits, train-only quantiles and real first/middle/last PyAV samples in train/heldout (all3cameras,tolerance1e-4). Once GPUs available, each job first passes2-step realbatch forward/backward/AdamW/checkpoint/strictreload smoke, then starts independentformalrun. Allformalcheckpoints preserved, final step saved even off eval cadence. Final strictreload evaluates128 evenly spaced frames per heldout episode; regular train/heldout evaluates32 per episode with equalnumber of fixedtrainepisodes. Do not mix densities. Failures stop that lane and report; inspect before --retry-failed, which can resume only its own checkpoint; no checkpoint requires explicit safe recovery plan, no overwrites.

## Decision and reporting

Per task compare SHORT FINAL versus LONG FINAL at identical128-frame evaluation; no posthoc beststep substitution. Mainmetric paired episode MAE difference(long-short), bootstrap95%CI over episodes, percentimprovement. Also CE,RMSE,CRPS,bias,90%coverage/width,MAEseconds,train-heldoutgap. Report per-task effect directions and uncertainty. Scales differ, so do not compare raw normalized MAE across tasks as inherenttaskdifficulty; use within-task relativegain and MAEseconds as context. Fixedtrainmean baseline helps detect a model that merely predicts a constant.

Consistency means similar short-vs-long directions, plausibleeffectsize and uncertainty, not that every smallimprovement is significant.11foldheldout episodes mean wideuncertainty; one seed cannot establish universality or bestbudget. Existing PiperX1500used8×4 vs new4×8, sameglobalbatch butdropout/reductiondifferences are a disclosedconfound. Because available data were duration matched, framecounts/steps are nearlyequal: this study can test cross-task reproducibility of the recipe, but CANNOT distinguish whether steps or epochs are the better scaling rule. No inference of RLpolicy benefit from value MAEalone. If evidence is inconclusive, say so, do not autoexpand beyond4runs.

Produce a3task×2budget table, curves with explicit colors for budgets and solidtrain/dashedheldout, normalizedepochxaxis, pairedCI table and recommended practicalbudget. Include oldPiperX1500 and newPiperX3000; constant8000 is diagnostic not an equivalentshortcosine. Update existing15min monitor, do not create duplicate.
