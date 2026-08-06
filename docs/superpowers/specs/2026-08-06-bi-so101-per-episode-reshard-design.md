# Bi-SO101 Per-Episode Reshard Design

## Goal

Convert `Elvinky/bi-so101-insert-screw-562ep` on the remote host into the
per-episode LeRobot v3 storage convention used by
`MINT-SJTU/RW-RL-Dataset`, without downloading dataset payloads locally.

## Confirmed source and target invariants

- Source revision: `1d426b48382c82b1afb0689e6f29d037cae3cfb3`.
- Source metadata declares 562 episodes, 2,070,776 rows, 30 FPS, and three
  video features.
- Target repository examples use one Parquet file per episode and one MP4
  file per episode per video feature while retaining LeRobot v3 metadata.
- Episode ordering, frame ordering, feature schemas, task mapping, and total
  frame counts must not change.
- No upload, push, or merge request is part of this conversion step.

## Architecture

Download the pinned source revision directly on the remote host with the
official Hugging Face CLI. Load and validate it with LeRobot 0.6.0. Write a
new sibling output directory rather than mutating the source.

Low-dimensional data is sliced at the episode boundaries recorded in
`meta/episodes` and written as one Snappy Parquet file per episode. Video is
split on the same boundaries. A small stream-copy trial is accepted only if
decoded frame counts match exactly; otherwise each aggregate input is decoded
once sequentially and emitted as per-episode H.264/YUV420P files. Episode
metadata is rewritten so every data/video file index equals its episode index
and every per-file video range starts at zero.

## Failure handling

All work is written under a new `.partial` directory. The final output name is
created only after every validation passes. The downloaded source remains
unchanged. Any mismatch stops processing and preserves logs and partial output
for diagnosis.

## Validation

1. Load source and output with the official `LeRobotDataset` API.
2. Require 562 episodes and 2,070,776 total data rows in both.
3. For every episode, compare Parquet row count, schema, and a canonical Arrow
   content digest for all columns.
4. Require exactly 562 MP4 files for each of the three video features.
5. Decode-count every MP4 and require its frame count to equal the episode
   length; each camera must total 2,070,776 decoded frames.
6. Check contiguous `episode_index`, `frame_index`, global `index`, metadata
   ranges, and successful random samples through `LeRobotDataset`.

