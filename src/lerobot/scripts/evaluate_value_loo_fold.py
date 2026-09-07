#!/usr/bin/env python

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


def main() -> None:
    register_third_party_plugins()
    parser = argparse.ArgumentParser(description="Evaluate one preregistered Value failure-LOO fold.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--loo-manifest", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(args.loo_manifest.read_text(encoding="utf-8"))
    fold = manifest["folds"][args.fold_index]
    failure_ids = [int(value) for value in fold["eval_failure_episode_indices"]]
    success_ids = [int(value) for value in fold["eval_success_episode_indices"]]
    if failure_ids != [int(fold["failure_episode_index"])]:
        raise ValueError("Each preregistered fold must contain exactly its named failure.")
    eval_ids = sorted(failure_ids + success_ids)

    dataset = LeRobotDataset(
        "local/evorl-piperx-copper-screw-attempt-value-v2",
        root=args.dataset_root,
        episodes=eval_ids,
        video_backend="pyav",
    )
    frames = dataset.hf_dataset.with_format(None)
    episode_values = np.asarray(frames["episode_index"], dtype=np.int64)
    frame_values = np.asarray(frames["frame_index"], dtype=np.int64)
    success_values = np.asarray(frames["logical_attempt_success"], dtype=np.bool_)
    transition_valid = np.asarray(frames["logical_transition_valid"], dtype=np.bool_)

    selected_positions: list[int] = []
    position_metadata: list[dict] = []
    for episode in eval_ids:
        positions = np.flatnonzero(episode_values == episode)
        valid_positions = positions[transition_valid[positions]]
        if valid_positions.size < 3:
            raise ValueError(f"Episode {episode} has fewer than three valid transition frames.")
        anchors = [
            int(valid_positions[0]),
            int(valid_positions[len(valid_positions) // 2]),
            int(valid_positions[-1]),
        ]
        for phase, position in zip(("start", "middle", "end"), anchors, strict=True):
            selected_positions.append(position)
            position_metadata.append(
                {
                    "episode_index": episode,
                    "frame_index": int(frame_values[position]),
                    "phase": phase,
                    "success": bool(success_values[position]),
                }
            )

    checkpoint = args.checkpoint.resolve()
    value_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(value_cfg, Pistar06Config):
        raise TypeError(f"Expected Pistar06Config, got {type(value_cfg).__name__}.")
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
    predictions: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = preprocessor(raw_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model.predict_value(batch)
            predictions.extend(float(value) for value in values.detach().cpu().numpy().reshape(-1))
    if len(predictions) != len(position_metadata):
        raise RuntimeError("Prediction count mismatch.")
    if not np.isfinite(np.asarray(predictions)).all():
        raise RuntimeError("Non-finite Value prediction in LOO evaluation.")
    for record, value in zip(position_metadata, predictions, strict=True):
        record["value"] = value

    by_episode: dict[int, dict[str, dict]] = defaultdict(dict)
    for record in position_metadata:
        by_episode[record["episode_index"]][record["phase"]] = record
    episode_summaries = []
    for episode, phases in sorted(by_episode.items()):
        episode_summaries.append(
            {
                "episode_index": episode,
                "success": phases["start"]["success"],
                "start_value": phases["start"]["value"],
                "middle_value": phases["middle"]["value"],
                "end_value": phases["end"]["value"],
                "end_minus_start": phases["end"]["value"] - phases["start"]["value"],
            }
        )
    failure_rows = [row for row in episode_summaries if not row["success"]]
    success_rows = [row for row in episode_summaries if row["success"]]
    if len(failure_rows) != 1 or not success_rows:
        raise RuntimeError("LOO evaluation requires one failure and at least one success.")

    failure_end = float(failure_rows[0]["end_value"])
    success_ends = [float(row["end_value"]) for row in success_rows]
    payload = {
        "schema_version": "attempt-aware-value-failure-loo-fold-result/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold_id": fold["fold_id"],
        "fold_index": args.fold_index,
        "checkpoint": str(checkpoint),
        "failure_episode_index": failure_ids[0],
        "failure_end_value": failure_end,
        "success_end_values": success_ends,
        "fold_separation": float(np.mean(success_ends) - failure_end),
        "episode_summaries": episode_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "episode_summaries"}, indent=2))


if __name__ == "__main__":
    main()
