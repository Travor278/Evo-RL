#!/usr/bin/env python3
"""Preflight gates for the paired full558+HIL replay experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.configs.train import ACPConfig, ReplaySamplingConfig
from lerobot.datasets.factory import make_dataset
from lerobot.rl.acp_hook import ACPPromptHook
from lerobot.rl.replay_sampler import build_replay_sampler
from lerobot.utils.import_utils import register_third_party_plugins


PROMPT = "Insert the copper screw into the black sleeve."
ACP_FIELD = "complementary_info.acp_indicator_evorl_official_v1"
MASK_FIELD = "complementary_info.acp_apply_mask_evorl_official_v1"
SOURCE_FIELD = "complementary_info.replay_source_evorl_official_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    return parser.parse_args()


def gate_files(view: Path) -> dict[str, int | float]:
    info = json.loads((view / "meta" / "info.json").read_text())
    data_files = sorted((view / "data").rglob("*.parquet"))
    videos = sorted((view / "videos").rglob("*.mp4"))
    rows = sum(pq.read_metadata(path).num_rows for path in data_files)
    assert len(data_files) == 618
    assert len(videos) == 1854
    assert all(path.is_symlink() and path.resolve().is_file() for path in videos)
    assert rows == info["total_frames"] == 1_600_639
    assert info["total_episodes"] == 618

    episodes = pq.read_table(view / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert episodes.num_rows == 618
    assert episodes["dataset_from_index"][0].as_py() == 0
    assert episodes["dataset_to_index"][-1].as_py() == rows

    source_counts = {0: 0, 1: 0}
    positive_counts = {0: 0, 1: 0}
    for path in data_files:
        table = pq.read_table(path, columns=[ACP_FIELD, MASK_FIELD, SOURCE_FIELD])
        source = table[SOURCE_FIELD].combine_chunks().to_numpy()
        mask = table[MASK_FIELD].combine_chunks().to_numpy()
        indicator = table[ACP_FIELD].combine_chunks().to_numpy()
        source_value = int(source[0])
        assert source_value in (0, 1)
        assert (source == source_value).all()
        assert (mask == source_value).all()
        assert ((indicator == 0) | (indicator == 1)).all()
        if source_value == 0:
            assert (indicator == 0).all()
        source_counts[source_value] += len(source)
        positive_counts[source_value] += int(indicator.sum())

    assert source_counts[0] == 1_255_729
    assert source_counts[1] == 344_910
    return {
        "data_files": len(data_files),
        "video_symlinks": len(videos),
        "total_frames": rows,
        "base_frames": source_counts[0],
        "hil_frames": source_counts[1],
        "hil_indicator_positive": positive_counts[1],
        "hil_indicator_fraction": positive_counts[1] / source_counts[1],
    }


def gate_hook() -> list[str]:
    hook = ACPPromptHook(
        ACPConfig(enable=True, indicator_field=ACP_FIELD, apply_mask_field=MASK_FIELD), seed=20260902
    )
    batch = {
        "task": [PROMPT] * 4,
        ACP_FIELD: torch.tensor([1, 0, 0, 1], dtype=torch.int64),
        MASK_FIELD: torch.tensor([1, 1, 0, 0], dtype=torch.int64),
    }
    tasks = hook(batch, 0)["task"]
    assert tasks == [
        f"{PROMPT}\nAdvantage: positive",
        f"{PROMPT}\nAdvantage: negative",
        PROMPT,
        PROMPT,
    ]
    return tasks


def gate_dataset_and_sampler(view: Path, policy: Path) -> dict[str, int | float | str]:
    import lerobot.policies  # noqa: F401  # register built-in policy choices
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig

    policy_cfg = PreTrainedConfig.from_pretrained(policy)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/piperx_full558_hil_train60_replay_v1",
            root=str(view),
            use_imagenet_stats=False,
            video_backend="pyav",
        ),
        policy=policy_cfg,
        output_dir=view.parent / "_not_created_contract_test",
        job_name="mixed-replay-contract-test",
        steps=1,
        num_workers=0,
        batch_size=4,
        rename_map={
            "observation.images.camera_top": "observation.images.base_0_rgb",
            "observation.images.camera_wrist_left": "observation.images.left_wrist_0_rgb",
            "observation.images.camera_wrist_right": "observation.images.right_wrist_0_rgb",
        },
    )
    dataset = make_dataset(cfg)
    assert dataset.num_frames == 1_600_639
    assert dataset.num_episodes == 618
    assert dataset.meta.tasks.iloc[0].name == PROMPT
    assert "observation.state" in dataset.meta.stats and "action" in dataset.meta.stats

    replay_cfg = ReplaySamplingConfig(
        enable=True,
        source_field=SOURCE_FIELD,
        target_hil_fraction=0.25,
        num_samples=64_000,
    )
    sampler_a, stats = build_replay_sampler(dataset, replay_cfg, 20260902)
    indices_a = list(sampler_a)
    sampled_source = np.asarray(dataset.hf_dataset.with_format(None)[SOURCE_FIELD], dtype=np.int64)
    sampled_ratio = float(sampled_source[np.asarray(indices_a, dtype=np.int64)].mean())
    assert abs(sampled_ratio - 0.25) < 0.01, sampled_ratio
    sampler_b, _ = build_replay_sampler(dataset, replay_cfg, 20260902)
    assert indices_a == list(sampler_b)

    for index in (0, 1_255_728, 1_255_729, 1_600_638):
        item = dataset[index]
        assert tuple(item["observation.state"].shape) == (14,)
        assert tuple(item["action"].shape) == (50, 14)
        assert item["task"] == PROMPT
        for key in (
            "observation.images.camera_top",
            "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right",
        ):
            assert tuple(item[key].shape) == (3, 480, 640)

    return {
        "dataset_frames": dataset.num_frames,
        "dataset_episodes": dataset.num_episodes,
        "sampler_base_count": stats.base_count,
        "sampler_hil_count": stats.hil_count,
        "sampler_target_hil_fraction": stats.target_hil_fraction,
        "sampled_hil_fraction_64000": sampled_ratio,
        "policy_type": policy_cfg.type,
    }


def main() -> None:
    args = parse_args()
    register_third_party_plugins()
    results = {
        "files": gate_files(args.view),
        "hook_tasks": gate_hook(),
        "dataset_sampler": gate_dataset_and_sampler(args.view, args.policy),
    }
    print("MIXED_REPLAY_CONTRACT_PASS")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
