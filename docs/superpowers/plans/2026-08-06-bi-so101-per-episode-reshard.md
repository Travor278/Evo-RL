# Bi-SO101 Per-Episode Reshard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified one-episode-per-Parquet-and-MP4 copy of the 562-episode Bi-SO101 screw insertion dataset on the remote host.

**Architecture:** Pin and download the Hub source on the remote host, then use LeRobot metadata with PyArrow and FFmpeg/PyAV to reshard without changing episode or frame ordering. Build into a partial directory and publish the final local directory only after exhaustive count and content validation.

**Tech Stack:** Hugging Face CLI/Hub, LeRobot 0.6.0, PyArrow 24, FFmpeg/ffprobe or PyAV, Python 3.12.

## Global Constraints

- Source revision is `1d426b48382c82b1afb0689e6f29d037cae3cfb3`.
- Dataset payloads are downloaded only to `zhaobo@192.168.105.96`.
- Source is immutable; output is built in a new sibling directory.
- Expected totals are 562 episodes and 2,070,776 frames at 30 FPS.
- No upload, push, or merge request is authorized in this plan.

---

### Task 1: Pin and fetch the source remotely

**Files:**
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_source/`
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_work/download.log`

**Interfaces:**
- Consumes: local Hugging Face token over process stdin only.
- Produces: immutable remote source tree at the pinned revision.

- [ ] Verify local Hugging Face authentication without printing the token.
- [ ] Pass the token through stdin to a remote `hf download` process without persisting it.
- [ ] Download the pinned dataset revision into the remote source directory.
- [ ] Compare the downloaded file list and sizes with Hub metadata.

### Task 2: Audit source metadata and test video splitting

**Files:**
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_work/source_audit.json`
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_work/video_trial/`

**Interfaces:**
- Consumes: pinned source tree.
- Produces: exact episode boundary table and selected video split mode.

- [ ] Load the source with `LeRobotDataset` and record schema, episode lengths, file mappings, and totals.
- [ ] Verify metadata ranges are contiguous and cover exactly 2,070,776 frames.
- [ ] Trial stream-copy extraction on episodes at the beginning, middle, and end of source shards.
- [ ] Decode-count each trial output and accept stream copy only if every result exactly matches its episode length.
- [ ] Otherwise select single-pass decode and per-episode H.264 encoding.

### Task 3: Build the per-episode output

**Files:**
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_per_episode.partial/`
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_work/convert.log`

**Interfaces:**
- Consumes: source audit and selected video split mode.
- Produces: complete partial LeRobot v3 dataset with one data/video file per episode.

- [ ] Copy stable metadata files into the partial tree.
- [ ] Slice each episode into `data/chunk-000/file-{episode_index:03d}.parquet` with Snappy compression.
- [ ] Split each camera into `videos/{video_key}/chunk-000/file-{episode_index:03d}.mp4`.
- [ ] Rewrite data/video file indices and per-file timestamp ranges in episode metadata.
- [ ] Preserve source task, stats, feature, FPS, robot, split, and global-count metadata.

### Task 4: Exhaustively validate and finalize

**Files:**
- Create remotely: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_work/validation.json`
- Rename remotely after success: `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_per_episode.partial/` to `/home/zhaobo/lerobot_datasets/bi_so101_insert_screw_562ep_per_episode/`

**Interfaces:**
- Consumes: source and partial output.
- Produces: finalized remote dataset and machine-readable validation evidence.

- [ ] Load source and output through the official LeRobot API.
- [ ] Compare episode count, total rows, feature schemas, and each episode length.
- [ ] Compare canonical Arrow content digests episode by episode.
- [ ] Decode-count all 1,686 MP4 files and compare every result with its episode length.
- [ ] Require each camera total to equal 2,070,776 frames and reject missing/extra files.
- [ ] Validate contiguous indices, metadata file mappings, timestamps, and random API samples.
- [ ] Write a JSON report containing all commands, versions, counts, and pass/fail results.
- [ ] Atomically rename the partial directory only when every check passes.
