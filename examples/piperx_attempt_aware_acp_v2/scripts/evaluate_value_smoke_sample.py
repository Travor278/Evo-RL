#!/usr/bin/env python3
"""Sampled train/validation gate for an attempt-aware Value smoke checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main() -> int:
    register_third_party_plugins()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    split_info = json.loads(args.split_manifest.read_text(encoding="utf-8"))["splits"]
    train_ids = set(split_info["train"]["episode_indices"])
    validation_ids = set(split_info["validation"]["episode_indices"])
    dataset = LeRobotDataset(
        "local/evorl-piperx-copper-screw-attempt-value-v2",
        root=args.dataset_root,
        episodes=sorted(train_ids | validation_ids),
        video_backend="pyav",
    )
    frames = dataset.hf_dataset.with_format(None)
    episode_values = np.asarray(frames["episode_index"], dtype=np.int64)
    frame_values = np.asarray(frames["frame_index"], dtype=np.int64)
    success_values = np.asarray(frames["logical_attempt_success"], dtype=np.bool_)
    attempt_index_values = np.asarray(frames["logical_attempt_index"], dtype=np.int64)
    transition_valid_values = np.asarray(frames["logical_transition_valid"], dtype=np.bool_)

    train_failure_ids = sorted(
        int(ep) for ep in np.unique(episode_values) if ep in train_ids and not bool(success_values[np.where(episode_values == ep)[0][0]])
    )
    train_success_by_index: dict[int, list[int]] = defaultdict(list)
    for ep in sorted(int(value) for value in np.unique(episode_values) if value in train_ids):
        positions = np.where(episode_values == ep)[0]
        if bool(success_values[positions[0]]):
            train_success_by_index[int(attempt_index_values[positions[0]])].append(ep)
    selected_train_success = sorted(ep for index in range(5) for ep in train_success_by_index[index][:5])
    selected_episodes = sorted(validation_ids | set(train_failure_ids) | set(selected_train_success))

    selected_positions = []
    position_metadata = []
    for ep in selected_episodes:
        positions = np.where(episode_values == ep)[0]
        if positions.size == 0:
            raise ValueError(f"Missing episode {ep} in loaded dataset")
        valid_positions = positions[transition_valid_values[positions]]
        if valid_positions.size < 3:
            raise ValueError(f"Episode {ep} has fewer than three valid transition frames")
        points = [
            int(valid_positions[0]),
            int(valid_positions[len(valid_positions) // 2]),
            int(valid_positions[-1]),
        ]
        for phase_index, position in enumerate(points):
            selected_positions.append(position)
            position_metadata.append(
                {
                    "episode_index": ep,
                    "frame_index": int(frame_values[position]),
                    "phase": ("start", "middle", "end")[phase_index],
                    "split": "train" if ep in train_ids else "validation",
                    "success": bool(success_values[position]),
                    "attempt_index": int(attempt_index_values[position]),
                }
            )

    checkpoint = args.checkpoint.resolve()
    value_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(value_cfg, Pistar06Config):
        raise TypeError(f"Expected Pistar06Config, got {type(value_cfg).__name__}")
    value_cfg.pretrained_path = checkpoint
    value_cfg.device = "cuda"
    model = make_policy(cfg=value_cfg, ds_meta=dataset.meta)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=value_cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    model.to("cuda").eval()
    loader = DataLoader(
        Subset(dataset, selected_positions),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    predictions = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = preprocessor(raw_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model.predict_value(batch)
            predictions.extend(float(value) for value in values.detach().cpu().numpy().reshape(-1))
    if len(predictions) != len(position_metadata):
        raise RuntimeError("Prediction count mismatch")
    for record, value in zip(position_metadata, predictions, strict=True):
        record["value"] = value

    by_episode: dict[int, dict[str, dict]] = defaultdict(dict)
    for record in position_metadata:
        by_episode[record["episode_index"]][record["phase"]] = record
    episode_summaries = []
    for ep, phases in sorted(by_episode.items()):
        start = phases["start"]
        middle = phases["middle"]
        end = phases["end"]
        episode_summaries.append(
            {
                "episode_index": ep,
                "split": start["split"],
                "success": start["success"],
                "attempt_index": start["attempt_index"],
                "start_value": start["value"],
                "middle_value": middle["value"],
                "end_value": end["value"],
                "end_minus_start": end["value"] - start["value"],
            }
        )

    def mean_for(split: str, success: bool, field: str) -> float | None:
        values = [row[field] for row in episode_summaries if row["split"] == split and row["success"] == success]
        return float(np.mean(values)) if values else None

    all_values = np.asarray(predictions, dtype=np.float64)
    progress_deltas = [row["end_minus_start"] for row in episode_summaries]
    start_values = [row["start_value"] for row in episode_summaries]
    attempt_indices = [float(row["attempt_index"]) for row in episode_summaries]
    metrics = {
        "sampled_episodes": len(episode_summaries),
        "sampled_frames": len(predictions),
        "train_failures_unique": len(train_failure_ids),
        "train_successes_sampled": len(selected_train_success),
        "validation_attempts": len(validation_ids),
        "value_min": float(np.min(all_values)),
        "value_max": float(np.max(all_values)),
        "value_mean": float(np.mean(all_values)),
        "value_std": float(np.std(all_values)),
        "mean_end_minus_start": float(np.mean(progress_deltas)),
        "positive_progress_fraction": float(np.mean(np.asarray(progress_deltas) > 0)),
        "attempt_index_start_value_correlation": pearson(attempt_indices, start_values),
        "train_success_mean_end": mean_for("train", True, "end_value"),
        "train_failure_mean_end": mean_for("train", False, "end_value"),
        "validation_success_mean_end": mean_for("validation", True, "end_value"),
        "validation_failure_mean_end": mean_for("validation", False, "end_value"),
    }
    metrics["train_end_separation"] = metrics["train_success_mean_end"] - metrics["train_failure_mean_end"]
    metrics["validation_end_separation"] = (
        metrics["validation_success_mean_end"] - metrics["validation_failure_mean_end"]
    )
    gates = {
        "finite": bool(np.isfinite(all_values).all()),
        "not_collapsed": metrics["value_std"] > 0.005,
        "mean_progress_positive": metrics["mean_end_minus_start"] > 0,
        "train_success_above_failure": metrics["train_end_separation"] > 0,
        "validation_success_above_failure": metrics["validation_end_separation"] > 0,
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "selection_policy": "all validation; all five train failures; five train successes per attempt index; no test",
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
        "episodes": episode_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "episodes"}, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
