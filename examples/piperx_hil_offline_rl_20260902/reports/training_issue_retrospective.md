# EvoRL training issue retrospective

## Conclusion

The initial failures and extreme ETAs were primarily software/data-pipeline and
storage issues, not insufficient H100 compute and not a fundamental CUDA 13.2
compute-group incompatibility. The final jobs passed 8-rank NCCL on H100 and
completed with finite losses using the NGC 25.02 CUDA 12.8 runtime under the
platform's CUDA 13.2-labelled H100 group.

| Symptom | Root cause | Category | Resolution/evidence |
|---|---|---|---|
| Value job failed before training | Pinned Transformers 4.53.3 source imported, but the installed distribution metadata had been removed, so LeRobot considered Transformers unavailable | Environment packaging | Install the pinned fork as an editable distribution and assert both runtime and metadata versions before launch |
| Value DDP failed on iteration 2 | Pi*0.6 value forward has conditionally unused parameters, while the official value trainer set `find_unused_parameters=False` | Distributed-training code | Patch 0002 enabled `find_unused_parameters=True`; 3-step 4090 and 200-step 8-rank H100 gates passed |
| Value training succeeded but ACP inference failed | Inference defaulted to TorchCodec; the image lacked compatible FFmpeg shared libraries | Backend integration | Patch 0003 exposed and forwarded `video_backend=pyav`; ACP recovery reused the completed value checkpoint |
| Policy benchmark projected roughly 188.45 hours for 10k | Random-access PyAV decoding was the bottleneck. Loader-only tests reproduced a mean 60.71 seconds/global batch without policy forward/backward; an 8.33-second GOP caused many unnecessary RGB conversions | Data pipeline | Direct stream-relative seek, deferred RGB conversion, uint8 DataLoader IPC, RAM staging, and a controlled prefetch budget |
| 4090 smoke reached backward but failed when Adam initialized moments | The 48GB card reached 47.31GiB used with only 48.69MiB free | 4090 capacity | Use SGD only for the 4090 execution gate; retain exact Adam on 80GB H100 for formal training |
| Early H100 jobs had dependency/startup failures | PyAV wheel/source and package versions did not match the Python 3.12 NGC image | Environment assembly | Reuse the verified locked wheelhouse and assert exact package versions/import paths |
| Original RL-aware 20k job failed at final save | Project HDD was full; training had reached 20k but the partial checkpoint was invalid | Storage | Quarantine the partial save, offload complete intermediate checkpoints to SSD, then resume exact optimizer/scheduler/RNG state from step 15k |

## Throughput evidence

- Failed policy r3: mean step time `67.841s`; projected 10k time `188.45h`.
- Loader-only r2: `424.9566s` for seven global batches, or `60.71s/batch`,
  reproducing almost all of the apparent training slowdown without model compute.
- Spawn-only control did not help: first batch `254.62s`, ruling out a Polars/fork
  deadlock as the main cause.
- Corrected loader (RAM staged, uint8 IPC, verified package stack): mean global-max
  batch `1.236s`, throughput `51.77 samples/s`.
- Formal RL-aware 10k on 4xH100: `22,657s` (`6.29h`).
- Formal Data-only 20k on 8xH100: `23,749s` (`6.60h`).

## Interpretation

H100 compatibility was proven by CUDA visibility, finite forward/backward,
8-rank NCCL, checkpoint save/resume, and two completed formal runs. The long
initial ETA should therefore be described as a pathological input-pipeline ETA,
not as evidence that EvoRL/PI0.5 cannot run efficiently on H100. The only clear
card-capacity problem occurred on the 48GB RTX 4090 when exact Adam state was
allocated.

Primary evidence is recorded in:

- `reports/v1_failure_report.md`
- `reports/v1_r1_failure_report.md`
- `reports/v1_r2_failure_report.md`
- `reports/v2_policy_smoke_r2_resource_report.md`
- `reports/v2_r3_supervisor_failure_report.md`
- `reports/v2sam_loaderbench_r2_stop_report.md`
- `reports/v2sam_loaderbench_r3_stop_report.md`
- `reports/v2sam-ego2exo-official-loaderbench-r9-r4-4gpu-origstack.md`
- `reports/v2sam-ego2exo-official-v2-r5-4gpu.md`
- `reports/v2sam-ego2exo-official-v3-20k-r2-8gpu.md`
