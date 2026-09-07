#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import lerobot.policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import ReplaySamplingConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.rl.replay_sampler import build_replay_sampler
from lerobot.utils.import_utils import register_third_party_plugins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampler-samples", type=int, default=50_000)
    parser.add_argument("--video-batches", type=int, default=2)
    args = parser.parse_args()

    register_third_party_plugins()
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
    metadata = LeRobotDatasetMetadata("local/attempt-mixed-v2", root=args.dataset)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata)
    dataset = LeRobotDataset(
        "local/attempt-mixed-v2",
        root=args.dataset,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    raw = dataset.hf_dataset.with_format(None)
    source = np.asarray(raw["replay_source"], dtype=np.int64)
    mask = np.asarray(raw["acp_apply_mask"], dtype=np.int64)
    indicator = np.asarray(raw["acp_indicator_attempt_v2"], dtype=np.int64)
    valid = np.asarray(raw["logical_action_chunk_valid_50"], dtype=bool)
    attempt_ids = np.asarray(raw["logical_attempt_id"], dtype=str)
    assert len(dataset) == 1_565_209
    assert dataset.num_episodes == 944
    assert int(np.sum(source == 0)) == 1_255_729
    assert int(np.sum(source == 1)) == 309_480
    assert np.all(mask[source == 0] == 0) and np.all(indicator[source == 0] == 0)
    assert len(set(attempt_ids[source == 1])) == 386

    sampler_reports = {}
    for target in (0.25, 0.50):
        cfg = ReplaySamplingConfig(
            enable=True,
            strategy="attempt_balanced",
            source_field="replay_source",
            hil_value=1,
            target_hil_fraction=target,
            num_samples=args.sampler_samples,
            attempt_id_field="logical_attempt_id",
            valid_chunk_field="logical_action_chunk_valid_50",
        )
        sampler, stats = build_replay_sampler(dataset, cfg, 20260906)
        samples = list(sampler)
        assert np.all(valid[samples])
        observed = float(np.mean(source[samples] == 1))
        assert abs(observed - target) < 0.015
        counts = {}
        for index in samples:
            if source[index] == 1:
                attempt_id = attempt_ids[index]
                counts[attempt_id] = counts.get(attempt_id, 0) + 1
        sampler_reports[str(target)] = {
            **stats.__dict__,
            "observed_hil_fraction": observed,
            "min_attempt_samples": min(counts.values()),
            "max_attempt_samples": max(counts.values()),
        }

    paired_cfg = ReplaySamplingConfig(
        enable=True,
        strategy="attempt_balanced",
        source_field="replay_source",
        hil_value=1,
        target_hil_fraction=0.5,
        num_samples=64,
        attempt_id_field="logical_attempt_id",
        valid_chunk_field="logical_action_chunk_valid_50",
    )
    acp_sampler, _ = build_replay_sampler(dataset, paired_cfg, 20260906)
    data_only_sampler, _ = build_replay_sampler(dataset, paired_cfg, 20260906)
    assert list(acp_sampler) == list(data_only_sampler)

    loader_sampler, _ = build_replay_sampler(dataset, paired_cfg, 20260906)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=loader_sampler,
        num_workers=2,
        pin_memory=True,
    )
    expected_cameras = [
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
        "observation.images.camera_wrist_right",
    ]
    batches = []
    for batch in itertools.islice(loader, args.video_batches):
        cameras = sorted(key for key in batch if key.startswith("observation.images."))
        assert cameras == expected_cameras
        assert tuple(batch["action"].shape) == (2, 50, 14)
        assert torch.isfinite(batch["action"]).all()
        assert all(batch[key].dtype == torch.uint8 for key in cameras)
        batches.append({"action_shape": list(batch["action"].shape), "cameras": cameras})

    payload = {
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "base_frames": int(np.sum(source == 0)),
        "attempt_frames": int(np.sum(source == 1)),
        "distinct_attempts": len(set(attempt_ids[source == 1])),
        "base_tagged_frames": int(np.sum(mask[source == 0])),
        "samplers": sampler_reports,
        "real_video_batches": batches,
        "acp_data_only_sampler_identical": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
