# Policy input-pipeline diagnosis and r1 retry contract

## Root cause

The first 8×H100 v2 benchmark was numerically healthy but input-bound: step 10 reported
77.151 s of data time and step 20 reported 73.226 s. The current Evo-RL checkout routed
`video_backend=pyav` through `torchvision.io.VideoReader.seek`. On the 4090 reproducer,
one late-timestamp call remained in I/O wait for more than 50 seconds.

The same files and timestamps decoded through a stream-relative native PyAV seek in
approximately 0.51–0.53 seconds in the isolated reproducer. The implementation is the
same seek strategy used by the validated full558 training source. That full558 r13 run
also used `video_backend=pyav`, batch 8 per rank, four workers per rank, and 8×H100;
its persisted metrics show about 1.28–1.35 s update time and 0.006–0.009 s visible
dataloader wait after warm-up through step 50,000.

## Correctness gate

Patch `0007-direct-pyav-stream-seek.patch` changes only the PyAV decoding wrapper. It
retains nearest-timestamp selection, tolerance checks, CHW float32 output, and [0, 1]
normalization. On one late base frame and one late HIL frame, its tensors were bitwise
identical to TorchCodec:

- base `file-000.mp4`, 120.0 s: SHA-256
  `2573d4100a1899c2974e420a6bb8e83bb12762bf341aa5ab1206f1b69fb0c3f1`
- HIL `file-558.mp4`, 150.0 s: SHA-256
  `70622b9e1e6631e30025c1d2df9b827d3b208965cd5e49064ab1adba0142e050`

TorchCodec itself is not selected for H100: the NGC 25.02 environment lacks the
system FFmpeg shared libraries required by its available wheel, and the prior full558
preflight recorded that incompatibility before switching to native PyAV.

## Retry gate

The paired retries are named `v2sam-ego2exo-official-v2-r1` and
`v2sam-ego2exo-official-v3-r1`. Each must pass imports, two fixed-frame hashes,
8-rank NCCL, finite loss/gradient checks, checkpoint 200, and a 200-step wall-clock
limit of 1,800 seconds. A retry exceeding that limit stops before formal training.
The formal 10,000-step jobs retain global batch 64, 25% HIL sampling, seed 20260902,
and identical replay/checkpoint semantics.
