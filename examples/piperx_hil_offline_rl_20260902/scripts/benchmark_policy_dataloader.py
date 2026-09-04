#!/usr/bin/env python3
"""Measure the policy input pipeline concurrently on every distributed rank.

This deliberately performs no model forward/backward. A barrier before every
batch makes the global maximum batch latency directly comparable to the DDP
straggler penalty seen by policy training.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator

import lerobot.policies  # noqa: F401 - registers built-in policy configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import ReplaySamplingConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.rl.replay_sampler import build_replay_sampler
from lerobot.utils.import_utils import register_third_party_plugins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piperx-loaderbench")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--backend", default="pyav")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--sampler", choices=("shuffle", "weighted"), required=True)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("default", "spawn"),
        default="default",
    )
    parser.add_argument("--target-hil-fraction", type=float, default=0.25)
    parser.add_argument(
        "--source-field",
        default="complementary_info.replay_source_evorl_official_v1",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"Expected exactly {args.expected_world_size} ranks, got {world_size}"
        )
    cuda_device = accelerator.device
    register_third_party_plugins()
    torch.manual_seed(args.seed)

    init_started = time.perf_counter()
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset,
        delta_timestamps=delta_timestamps,
        video_backend=args.backend,
        return_uint8=True,
    )
    generator = torch.Generator()
    # Every rank starts from the same sampler stream. Accelerator then wraps
    # the batch sampler exactly as it does in lerobot_train.py and assigns a
    # distinct local batch from each global group of eight batches.
    generator.manual_seed(args.seed)
    sampling_stats: dict[str, object]
    if args.sampler == "weighted":
        sampling_cfg = ReplaySamplingConfig(
            enable=True,
            source_field=args.source_field,
            hil_value=1,
            target_hil_fraction=args.target_hil_fraction,
            num_samples=len(dataset),
        )
        sampler, stats = build_replay_sampler(dataset, sampling_cfg, args.seed)
        sampling_stats = stats.__dict__
    else:
        sampler = None
        sampling_stats = {"mode": "shuffle_true"}

    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=args.sampler == "shuffle",
        pin_memory=True,
        drop_last=args.sampler == "weighted",
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
        generator=generator,
        multiprocessing_context=(
            args.multiprocessing_context
            if args.num_workers and args.multiprocessing_context != "default"
            else None
        ),
    )
    loader = accelerator.prepare(loader)
    iterator = iter(loader)
    init_seconds = time.perf_counter() - init_started
    init_tensor = torch.tensor([init_seconds], device=cuda_device, dtype=torch.float64)
    init_gather = [torch.zeros_like(init_tensor) for _ in range(world_size)]
    dist.all_gather(init_gather, init_tensor)
    if rank == 0:
        print(
            "LOADERBENCH_INIT",
            json.dumps(
                {
                    "variant": args.variant,
                    "dataset": str(args.dataset),
                    "world_size": world_size,
                    "per_rank_seconds": [round(float(item.item()), 6) for item in init_gather],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    expected_cameras = {
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
        "observation.images.camera_wrist_right",
    }
    per_rank_timings: list[list[float]] = [[] for _ in range(world_size)]
    global_max_timings: list[float] = []
    measured_hil = 0
    measured_samples = 0
    all_finite = True
    for batch_index in range(args.batches):
        dist.barrier()
        started = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - started

        camera_keys = {key for key in batch if key.startswith("observation.images.")}
        cameras_ok = camera_keys == expected_cameras and all(
            int(batch[key].shape[0]) == args.batch_size for key in expected_cameras
        )
        action = torch.as_tensor(batch["action"])
        images_uint8 = all(batch[key].dtype == torch.uint8 for key in expected_cameras)
        finite = bool(torch.isfinite(action).all().item()) and cameras_ok and images_uint8
        if args.sampler == "weighted":
            source = torch.as_tensor(batch[args.source_field]).reshape(-1)
            hil = int((source == 1).sum().item())
            samples = int(source.numel())
        else:
            hil = 0
            samples = int(action.shape[0])

        local = torch.tensor(
            [elapsed, float(hil), float(samples), 1.0 if finite else 0.0],
            device=cuda_device,
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        rank_seconds = [float(item[0].item()) for item in gathered]
        step_finite = all(bool(item[3].item()) for item in gathered)
        if not step_finite:
            raise RuntimeError(f"Non-finite action or camera contract failure at batch {batch_index}")
        if batch_index >= args.warmup_batches:
            for source_rank, seconds in enumerate(rank_seconds):
                per_rank_timings[source_rank].append(seconds)
            global_max_timings.append(max(rank_seconds))
            measured_hil += sum(int(item[1].item()) for item in gathered)
            measured_samples += sum(int(item[2].item()) for item in gathered)
            all_finite = all_finite and step_finite
        if rank == 0:
            print(
                "LOADERBENCH_BATCH",
                json.dumps(
                    {
                        "variant": args.variant,
                        "index": batch_index,
                        "warmup": batch_index < args.warmup_batches,
                        "per_rank_seconds": [round(value, 6) for value in rank_seconds],
                        "global_max_seconds": round(max(rank_seconds), 6),
                        "slowest_rank": max(range(world_size), key=rank_seconds.__getitem__),
                        "hil": sum(int(item[1].item()) for item in gathered),
                        "samples": sum(int(item[2].item()) for item in gathered),
                        "finite_contract": step_finite,
                        "images_uint8": images_uint8,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if not global_max_timings:
        raise RuntimeError("No measured batches")
    summary = {
        "schema": "evorl_distributed_loaderbench/v1",
        "variant": args.variant,
        "dataset": str(args.dataset),
        "backend": args.backend,
        "sampler": args.sampler,
        "multiprocessing_context": args.multiprocessing_context,
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "num_workers_per_rank": args.num_workers,
        "total_workers": args.num_workers * world_size,
        "prefetch_factor": args.prefetch_factor,
        "total_prefetch_samples": (
            args.num_workers * world_size * args.prefetch_factor * args.batch_size
        ),
        "warmup_batches": args.warmup_batches,
        "measured_batches": len(global_max_timings),
        "global_max_mean_seconds": statistics.fmean(global_max_timings),
        "global_max_median_seconds": statistics.median(global_max_timings),
        "global_max_p90_seconds": percentile(global_max_timings, 0.90),
        "global_max_worst_seconds": max(global_max_timings),
        "global_samples_per_second": (
            args.batch_size * world_size * len(global_max_timings) / sum(global_max_timings)
        ),
        "per_rank_mean_seconds": [statistics.fmean(values) for values in per_rank_timings],
        "per_rank_worst_seconds": [max(values) for values in per_rank_timings],
        "measured_hil_fraction": (
            measured_hil / measured_samples if args.sampler == "weighted" else None
        ),
        "finite_contract": all_finite,
        "images_uint8": True,
        "sampling_stats": sampling_stats,
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("DISTRIBUTED_LOADERBENCH_PASS", json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
