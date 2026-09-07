#!/usr/bin/env python3
"""Stage-0 real-data Pi*0.6 forward/backward gate for attempt-aware Value."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from lerobot.configs.value_train import ValueTargetsConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.attempt_sampler import build_attempt_uniform_sampler
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_episodes = split_manifest["splits"]["train"]["episode_indices"]
    probe_episodes = [train_episodes[0], 454]
    assets = json.loads(args.asset_manifest.read_text(encoding="utf-8"))["assets"]
    dataset = LeRobotDataset(
        "local/evorl-piperx-copper-screw-attempt-value-v2",
        root=args.dataset_root,
        episodes=probe_episodes,
        video_backend="pyav",
    )
    sampler, sampler_stats = build_attempt_uniform_sampler(
        dataset,
        attempt_field="logical_attempt_id",
        valid_field="logical_transition_valid",
        outcome_known_field="logical_attempt_outcome_known",
        num_samples=32,
        seed=20260906,
    )
    config = Pistar06Config(
        device="cuda",
        dtype="bfloat16",
        vision_repo_id=assets["vision"]["snapshot_path"],
        language_repo_id=assets["language"]["snapshot_path"],
        use_gradient_checkpointing=True,
        freeze_vision_encoder=False,
        freeze_language_model=False,
        push_to_hub=False,
    )
    model = make_policy(cfg=config, ds_meta=dataset.meta)
    model.to("cuda")
    preprocessor, _ = make_pre_post_processors(policy_cfg=config, dataset_stats=dataset.meta.stats)
    hook = model.build_training_raw_batch_hook(dataset, ValueTargetsConfig(success_field="episode_success"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=2)
    raw_batch = next(iter(loader))
    if not bool(torch.as_tensor(raw_batch["logical_transition_valid"]).all()):
        raise RuntimeError("Attempt sampler selected an invalid transition")
    batch = preprocessor(hook(raw_batch, 0))
    model.train()
    loss, metrics = model.forward(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss}")
    loss.backward()
    grad_sq = 0.0
    grad_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Non-finite gradient")
        grad_sq += float(parameter.grad.detach().float().pow(2).sum().item())
        grad_tensors += 1
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()),
        "probe_episodes": probe_episodes,
        "sampler_stats": sampler_stats.__dict__,
        "loss": float(loss.detach().item()),
        "metrics": metrics,
        "gradient_norm": math.sqrt(grad_sq),
        "gradient_tensors": grad_tensors,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "finite_loss": True,
        "finite_gradients": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
