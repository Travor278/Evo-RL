#!/usr/bin/env python3
"""Deterministic held-out action-chunk MAE comparison for PiperX π0.5 checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


ROOT = Path(
    "/inspire/hdd/project/luojianlan/zhubingwen-253108120125/"
    "codex_remote_ops/evorl_hil_rl_piperx_20260902"
)
RAW_DATASET_ROOT = Path(
    "/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/"
    "evorl_datasets/evorl-piperx-copper-screw-hil-clean/"
    "18016184b09929643b6bd055b3f5833bfb2e7b85"
)
DATASET_ROOT = ROOT / "derived/hil_official_policy_v1"
TEST_MANIFEST = ROOT / "manifests/test_episodes.json"
CHECKPOINTS = {
    "base": ROOT / "checkpoints/pi05_base_step50000_policy_compat_6f2db_v2",
    "rlaware_10k": ROOT
    / "checkpoints/v2sam-ego2exo-official-v2-r5-4gpu-acp-policy/checkpoints/010000/pretrained_model",
    "rlaware_20k_run_step10k": ROOT
    / "checkpoints/v2sam-ego2exo-official-v2-20k-r1-8gpu-acp-policy/checkpoints/010000/pretrained_model",
    "rlaware_20k": ROOT
    / "checkpoints/v2sam-ego2exo-official-v2-20k-r1-8gpu-acp-policy/checkpoints/020000/pretrained_model",
    "data_only_20k": ROOT
    / "checkpoints/v2sam-ego2exo-official-v3-20k-r2-8gpu-data-only-policy/checkpoints/020000/pretrained_model",
}
EXPECTED_SHA256 = {
    "rlaware_10k": "e97ca20f967afe6a3021c865c786c7084e9bce76c2b1bbf81bbbf36cf1d1453c",
    "rlaware_20k_run_step10k": "89e6ee61192823a92bc420b9d3f795e99afd78a4f5c7029ac10da38b93a9a348",
    "rlaware_20k": "89e9445826e3d7fcde353ea4f38f5b5d2924e3c06c7905314b0945b393bd71d6",
    "data_only_20k": "9ac060fb590610c0bc7a3865c807c4f2bad99e1e2ea675c4f3b85f85cd2879a4",
}
RENAME_MAP = {
    "observation.images.top": "observation.images.base_0_rgb",
    "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
}
IMAGE_KEYS = tuple(RENAME_MAP)
TASK = "Insert the copper screw into the black sleeve."


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def numpy_value(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def evenly_pick(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=count)
    chosen = []
    for position in positions:
        idx = indices[int(round(float(position)))]
        if idx not in chosen:
            chosen.append(idx)
    for idx in indices:
        if len(chosen) >= count:
            break
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def build_anchors(
    dataset: LeRobotDataset,
    episode_indices: list[int],
    chunk_size: int,
    samples_per_episode: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hf = dataset.hf_dataset
    view_global_indices = [int(scalar(v)) for v in hf["index"]]
    global_to_local = {value: position for position, value in enumerate(view_global_indices)}
    if len(global_to_local) != len(view_global_indices):
        raise RuntimeError("duplicate global index in evaluation view")

    columns = [
        "episode_index",
        "frame_index",
        "index",
        "action",
        "complementary_info.segment_id",
        "complementary_info.action_valid",
        "complementary_info.is_intervention",
        "complementary_info.control_state",
    ]
    raw: dict[str, list[Any]] = {name: [] for name in columns}
    for parquet_path in sorted((RAW_DATASET_ROOT / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path, columns=columns)
        for name in columns:
            raw[name].extend(table[name].to_pylist())
    selected_positions = [
        position
        for position, value in enumerate(raw["episode_index"])
        if int(value) in episode_indices
    ]
    episodes = np.asarray([int(raw["episode_index"][i]) for i in selected_positions], dtype=np.int64)
    frames = np.asarray([int(raw["frame_index"][i]) for i in selected_positions], dtype=np.int64)
    global_indices = np.asarray([int(raw["index"][i]) for i in selected_positions], dtype=np.int64)
    segments = [str(raw["complementary_info.segment_id"][i]) for i in selected_positions]
    action_valid = np.asarray(
        [bool(raw["complementary_info.action_valid"][i]) for i in selected_positions], dtype=bool
    )
    interventions = np.asarray(
        [bool(raw["complementary_info.is_intervention"][i]) for i in selected_positions], dtype=bool
    )
    control_states = [str(raw["complementary_info.control_state"][i]) for i in selected_positions]
    actions = np.asarray([raw["action"][i] for i in selected_positions], dtype=np.float32)
    order = np.argsort(global_indices)
    episodes = episodes[order]
    frames = frames[order]
    global_indices = global_indices[order]
    segments = [segments[i] for i in order]
    action_valid = action_valid[order]
    interventions = interventions[order]
    control_states = [control_states[i] for i in order]
    actions = actions[order]
    finite = np.isfinite(actions).all(axis=1)

    candidates_by_episode: dict[int, list[int]] = defaultdict(list)
    n = len(episodes)
    for index in range(0, n - chunk_size + 1):
        end = index + chunk_size
        ep = int(episodes[index])
        if ep not in episode_indices:
            continue
        if int(episodes[end - 1]) != ep:
            continue
        if int(frames[end - 1]) - int(frames[index]) != chunk_size - 1:
            continue
        if not np.all(np.diff(frames[index:end]) == 1):
            continue
        segment = segments[index]
        if any(value != segment for value in segments[index:end]):
            continue
        control = control_states[index]
        if any(value != control for value in control_states[index:end]):
            continue
        if not bool(np.all(action_valid[index:end] & finite[index:end])):
            continue
        candidates_by_episode[ep].append(index)

    anchors: list[dict[str, Any]] = []
    candidate_summary: dict[str, Any] = {}
    for ep in episode_indices:
        candidates = candidates_by_episode[ep]
        policy = [idx for idx in candidates if not interventions[idx]]
        human = [idx for idx in candidates if interventions[idx]]
        human_target = min(samples_per_episode // 2, len(human))
        chosen = evenly_pick(human, human_target)
        chosen += evenly_pick(policy, samples_per_episode - len(chosen))
        if len(chosen) < samples_per_episode:
            remaining = [idx for idx in candidates if idx not in chosen]
            chosen += evenly_pick(remaining, samples_per_episode - len(chosen))
        chosen = sorted(chosen[:samples_per_episode], key=lambda idx: int(frames[idx]))
        if len(chosen) != samples_per_episode:
            raise RuntimeError(
                f"episode {ep} has only {len(chosen)} valid anchors; required {samples_per_episode}"
            )
        candidate_summary[str(ep)] = {
            "valid_candidates": len(candidates),
            "policy_candidates": len(policy),
            "intervention_candidates": len(human),
            "selected": len(chosen),
            "selected_intervention": sum(bool(interventions[idx]) for idx in chosen),
        }
        for idx in chosen:
            anchors.append(
                {
                    "dataset_local_index": global_to_local[int(global_indices[idx])],
                    "global_index": int(global_indices[idx]),
                    "episode_index": int(episodes[idx]),
                    "frame_index": int(frames[idx]),
                    "segment_id": segments[idx],
                    "control_state": control_states[idx],
                    "is_intervention": bool(interventions[idx]),
                }
            )
    return anchors, candidate_summary


def bootstrap_episode_ci(records: list[dict[str, Any]], field: str, seed: int) -> list[float]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_episode[int(record["episode_index"])].append(float(record[field]))
    episode_means = np.asarray(
        [np.mean(values) for _, values in sorted(by_episode.items())], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000, dtype=np.float64)
    for i in range(draws.size):
        draws[i] = rng.choice(episode_means, size=len(episode_means), replace=True).mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    physical_dim = np.asarray([r["physical_mae_per_dim"] for r in records], dtype=np.float64)
    normalized_dim = np.asarray([r["normalized_mae_per_dim"] for r in records], dtype=np.float64)
    groups = {}
    for label, flag in (("policy", False), ("intervention", True)):
        subset = [r for r in records if bool(r["is_intervention"]) is flag]
        groups[label] = {
            "samples": len(subset),
            "physical_chunk_mae": float(np.mean([r["physical_chunk_mae"] for r in subset]))
            if subset
            else None,
            "normalized_chunk_mae": float(np.mean([r["normalized_chunk_mae"] for r in subset]))
            if subset
            else None,
        }
    return {
        "samples": len(records),
        "episodes": len({int(r["episode_index"]) for r in records}),
        "physical_chunk_mae": float(np.mean([r["physical_chunk_mae"] for r in records])),
        "physical_chunk_mae_episode_bootstrap_95ci": bootstrap_episode_ci(
            records, "physical_chunk_mae", seed
        ),
        "physical_first_step_mae": float(
            np.mean([r["physical_first_step_mae"] for r in records])
        ),
        "normalized_chunk_mae": float(np.mean([r["normalized_chunk_mae"] for r in records])),
        "normalized_chunk_mae_episode_bootstrap_95ci": bootstrap_episode_ci(
            records, "normalized_chunk_mae", seed + 1
        ),
        "flow_matching_loss": float(np.mean([r["flow_matching_loss"] for r in records])),
        "physical_mae_per_dim": physical_dim.mean(axis=0).tolist(),
        "normalized_mae_per_dim": normalized_dim.mean(axis=0).tolist(),
        "groups": groups,
    }


def model_file_metadata(name: str, checkpoint: Path) -> dict[str, Any]:
    model_file = checkpoint / "model.safetensors"
    if not model_file.is_file():
        raise RuntimeError(f"missing model file: {model_file}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "model_file": str(model_file.resolve()),
        "model_size": model_file.stat().st_size,
        "expected_sha256": EXPECTED_SHA256.get(name),
    }


def evaluate_model(
    name: str,
    checkpoint: Path,
    dataset: LeRobotDataset,
    anchors: list[dict[str, Any]],
    anchor_sha256: str,
    part_path: Path,
    seed: int,
) -> dict[str, Any]:
    if part_path.is_file():
        existing = json.loads(part_path.read_text(encoding="utf-8"))
        if existing.get("anchor_sha256") == anchor_sha256 and len(existing.get("records", [])) == len(
            anchors
        ):
            print(f"MAE_MODEL_REUSE name={name} samples={len(anchors)}", flush=True)
            return existing

    print(f"MAE_MODEL_LOAD_START name={name} checkpoint={checkpoint}", flush=True)
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if not isinstance(config, PI05Config):
        raise RuntimeError(f"unexpected policy config for {name}: {type(config).__name__}")
    config.device = "cuda"
    config.gradient_checkpointing = False
    config.compile_model = False
    policy = PI05Policy.from_pretrained(
        str(checkpoint), config=config, strict=True, local_files_only=True
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    torch.set_float32_matmul_precision("high")
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    required_keys = (*IMAGE_KEYS, "observation.state", "action", "task")
    for position, anchor in enumerate(anchors):
        item = dataset[int(anchor["dataset_local_index"])]
        if item["task"] != TASK:
            raise RuntimeError(f"task mismatch: {item['task']!r}")
        batch = default_collate([{key: item[key] for key in required_keys}])
        target_physical = batch["action"].detach().float().cpu()
        processed = preprocessor(batch)
        target_normalized = processed["action"].detach().float().cpu()

        deterministic_seed = seed + int(anchor["global_index"])
        torch.manual_seed(deterministic_seed)
        torch.cuda.manual_seed_all(deterministic_seed)
        with torch.inference_mode():
            flow_loss, _ = policy.forward(processed)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(1_000_000 + deterministic_seed)
            noise = torch.randn(
                (1, config.chunk_size, config.max_action_dim),
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            predicted_normalized = policy.predict_action_chunk(processed, noise=noise)
            predicted_physical = postprocessor(predicted_normalized)
        if isinstance(predicted_physical, dict):
            predicted_physical = predicted_physical["action"]
        predicted_physical = predicted_physical.detach().float().cpu()
        predicted_normalized = predicted_normalized.detach().float().cpu()
        if predicted_physical.shape != target_physical.shape:
            raise RuntimeError(
                f"physical shape mismatch: pred={predicted_physical.shape} target={target_physical.shape}"
            )
        if predicted_normalized.shape != target_normalized.shape:
            raise RuntimeError(
                f"normalized shape mismatch: pred={predicted_normalized.shape} target={target_normalized.shape}"
            )
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (predicted_physical, target_physical, predicted_normalized, target_normalized)
        ):
            raise RuntimeError("non-finite prediction or target")
        physical_abs = (predicted_physical - target_physical).abs()[0]
        normalized_abs = (predicted_normalized - target_normalized).abs()[0]
        record = {
            **anchor,
            "deterministic_seed": deterministic_seed,
            "physical_chunk_mae": float(physical_abs.mean().item()),
            "physical_first_step_mae": float(physical_abs[0].mean().item()),
            "normalized_chunk_mae": float(normalized_abs.mean().item()),
            "flow_matching_loss": float(flow_loss.detach().float().item()),
            "physical_mae_per_dim": physical_abs.mean(dim=0).tolist(),
            "normalized_mae_per_dim": normalized_abs.mean(dim=0).tolist(),
        }
        if not all(math.isfinite(float(record[key])) for key in (
            "physical_chunk_mae",
            "physical_first_step_mae",
            "normalized_chunk_mae",
            "flow_matching_loss",
        )):
            raise RuntimeError(f"non-finite metric: {record}")
        records.append(record)
        if (position + 1) % 8 == 0 or position + 1 == len(anchors):
            payload = {
                "model": name,
                "anchor_sha256": anchor_sha256,
                "checkpoint": model_file_metadata(name, checkpoint),
                "records": records,
                "complete": position + 1 == len(anchors),
            }
            part_path.parent.mkdir(parents=True, exist_ok=True)
            part_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"MAE_MODEL_PROGRESS name={name} samples={position+1}/{len(anchors)} "
                f"physical_mae={np.mean([r['physical_chunk_mae'] for r in records]):.6f} "
                f"normalized_mae={np.mean([r['normalized_chunk_mae'] for r in records]):.6f}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    result = {
        "model": name,
        "anchor_sha256": anchor_sha256,
        "checkpoint": model_file_metadata(name, checkpoint),
        "records": records,
        "summary": summarize(records, seed + 10_000),
        "elapsed_seconds": elapsed,
        "complete": True,
    }
    part_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del policy, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"MAE_MODEL_PASS name={name} seconds={elapsed:.1f}", flush=True)
    return result


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for name in payload["model_order"]:
        summary = payload["models"][name]["summary"]
        ci = summary["physical_chunk_mae_episode_bootstrap_95ci"]
        rows.append(
            f"| {name} | {summary['physical_chunk_mae']:.6f} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] | {summary['physical_first_step_mae']:.6f} | "
            f"{summary['normalized_chunk_mae']:.6f} | {summary['flow_matching_loss']:.6f} |"
        )
    base = payload["models"]["base"]["summary"]["physical_chunk_mae"] if "base" in payload["models"] else None
    comparisons = []
    if base is not None:
        for name in payload["model_order"]:
            if name == "base":
                continue
            value = payload["models"][name]["summary"]["physical_chunk_mae"]
            comparisons.append(
                f"- `{name}` vs `base`: absolute MAE change `{value-base:+.6f}`, relative change `{(value/base-1)*100:+.2f}%`."
            )
    path.write_text(
        "# Held-out PiperX action-chunk MAE comparison\n\n"
        "Evaluation uses the fixed test Episodes 17, 21, 22, 23, 72, 73, 74, and 75. "
        "Anchors require a contiguous 50-frame action chunk inside one segment and one control state, "
        "with valid finite executed actions. Identical anchors and deterministic diffusion noise are used for every model.\n\n"
        "| Model | Physical chunk MAE | Episode bootstrap 95% CI | First-step MAE | Normalized chunk MAE | Flow loss |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        + "\n".join(comparisons)
        + "\n\nPhysical MAE averages joint-angle and gripper dimensions with different units/ranges, "
        "so normalized MAE and per-dimension MAE should be read alongside it. These are offline imitation diagnostics, not success-rate estimates.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-episode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--models", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/pi05_base_rlaware10k_rlaware20k_heldout_mae.json",
    )
    parser.add_argument(
        "--anchor-output",
        type=Path,
        default=ROOT / "manifests/pi05_heldout_mae_anchors.json",
    )
    parser.add_argument(
        "--part-dir",
        type=Path,
        default=ROOT / "reports/pi05_heldout_mae_parts",
    )
    args = parser.parse_args()
    if args.samples_per_episode <= 0:
        raise ValueError("samples-per-episode must be positive")
    test = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    episode_indices = [int(value) for value in test["episode_indices"]]
    if episode_indices != [17, 21, 22, 23, 72, 73, 74, 75]:
        raise RuntimeError(f"held-out test manifest drift: {episode_indices}")

    reference_config = PreTrainedConfig.from_pretrained(str(CHECKPOINTS["rlaware_10k"]))
    if not isinstance(reference_config, PI05Config):
        raise RuntimeError(f"unexpected reference config: {type(reference_config).__name__}")
    if reference_config.chunk_size != 50:
        raise RuntimeError(f"unexpected chunk size: {reference_config.chunk_size}")
    delta_timestamps = {
        "action": [index / 30.0 for index in reference_config.action_delta_indices]
    }
    dataset = LeRobotDataset(
        "local/evorl-piperx-copper-screw-hil-clean",
        root=DATASET_ROOT,
        episodes=episode_indices,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    anchors, candidate_summary = build_anchors(
        dataset, episode_indices, reference_config.chunk_size, args.samples_per_episode
    )
    anchor_contract = {
        "schema": "pi05_heldout_mae_anchors/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(DATASET_ROOT),
        "raw_dataset_root": str(RAW_DATASET_ROOT),
        "dataset_revision": RAW_DATASET_ROOT.name,
        "test_manifest": str(TEST_MANIFEST),
        "episode_indices": episode_indices,
        "chunk_size": reference_config.chunk_size,
        "samples_per_episode": args.samples_per_episode,
        "total_samples": len(anchors),
        "filters": [
            "contiguous_frame_index",
            "single_segment",
            "single_control_state",
            "action_valid_true_for_all_50",
            "finite_executed_action_for_all_50",
        ],
        "rename_map": RENAME_MAP,
        "candidate_summary": candidate_summary,
        "anchors": anchors,
    }
    canonical = json.dumps(anchor_contract, sort_keys=True, separators=(",", ":")).encode()
    anchor_sha256 = hashlib.sha256(canonical).hexdigest()
    anchor_contract["anchor_sha256"] = anchor_sha256
    args.anchor_output.parent.mkdir(parents=True, exist_ok=True)
    args.anchor_output.write_text(
        json.dumps(anchor_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"MAE_ANCHOR_GATE_OK episodes={len(episode_indices)} samples={len(anchors)} "
        f"sha256={anchor_sha256}",
        flush=True,
    )

    results = {}
    for name in args.models:
        results[name] = evaluate_model(
            name,
            CHECKPOINTS[name],
            dataset,
            anchors,
            anchor_sha256,
            args.part_dir / f"{name}.json",
            args.seed,
        )
    final = {
        "schema": "pi05_heldout_mae_comparison/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "held_out_test_episodes": episode_indices,
            "same_anchors": True,
            "same_preprocessing_contract": True,
            "saved_checkpoint_normalization": True,
            "same_deterministic_noise_per_anchor": True,
            "target": "actual executed action chunk",
            "chunk_size": 50,
            "inference_steps": 10,
            "physical_mae_mixes_joint_and_gripper_units": True,
            "offline_metric_not_success_rate": True,
        },
        "anchor_manifest": str(args.anchor_output),
        "anchor_sha256": anchor_sha256,
        "model_order": args.models,
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(final, args.output.with_suffix(".md"))
    print(f"MAE_COMPARISON_PASS output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
