#!/usr/bin/env python3
"""Pre-registered split diagnostics for attempt-aware ACP v2 policies.

Validation mode is used for replay-ratio selection. Test mode is allowed only after
the final training choices are frozen. It evaluates identical deterministic anchors
and diffusion noise for every checkpoint, plus a fixed Base-558 retention probe.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as pads
import torch
from torch.utils.data._utils.collate import default_collate

import lerobot.policies  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.import_utils import register_third_party_plugins


ROOT = Path(
    "/inspire/hdd/project/luojianlan/zhubingwen-253108120125/"
    "codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906"
)
ANNOTATED_ROOT = Path(
    "/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/"
    "codex_remote_ops/evorl_attempt_aware_rl_piperx_20260906/attempt_acp_annotated_view"
)
BASE_ROOT = Path(
    "/inspire/ssd/project/luojianlan/public/zhubingwen-253108120125/"
    "pi05_full558_044/dataset_v3_training_view"
)
CHECKPOINTS = {
    "base": Path(
        "/inspire/hdd/project/luojianlan/zhubingwen-253108120125/"
        "codex_remote_ops/evorl_hil_rl_piperx_20260902/checkpoints/"
        "pi05_base_step50000_policy_compat_6f2db_v2"
    ),
    "acp25": ROOT / "checkpoints/v2sam-ego2exo-attempt-v2-acp_25pct_smoke2k_r2/checkpoints/002000/pretrained_model",
    "bc25": ROOT / "checkpoints/v2sam-ego2exo-attempt-v2-bc_25pct_smoke2k_r1/checkpoints/002000/pretrained_model",
    "acp50": ROOT / "checkpoints/v2sam-ego2exo-attempt-v2-acp_50pct_smoke2k_r1/checkpoints/002000/pretrained_model",
    "bc50": ROOT / "checkpoints/v2sam-ego2exo-attempt-v2-bc_50pct_smoke2k_r1/checkpoints/002000/pretrained_model",
    "episode_acp_v1": Path(
        "/inspire/hdd/project/luojianlan/zhubingwen-253108120125/"
        "codex_remote_ops/evorl_hil_rl_piperx_20260902/checkpoints/"
        "v2sam-ego2exo-official-v2-20k-r1-8gpu-acp-policy/checkpoints/020000/pretrained_model"
    ),
    "acp20k": ROOT / "checkpoints/v2sam-ego2exo-attempt-v2-acp-20k-r2/checkpoints/020000/pretrained_model",
    "bc20k": ROOT / "checkpoints/v2sam-ego2exo-attempt-v2-bc-20k-r2/checkpoints/020000/pretrained_model",
}
EXPECTED_SHA256 = {
    "acp25": "8a871364791d930ca31246b0f6ae26393283c5d9df87aa4cf6345621e24563ad",
    "bc25": "f07e818ace9977600b498a06256ad44f384e41cefc693d7cf8f4c2a723ae45c2",
    "acp50": "656cfe44fc232eb93371bf30e3cb499bb310bdc48251e8fb75e0fe09cf0c82c8",
    "bc50": "c25de3d7350fe6a7358e22fa9c24d71316f279224ef4a8ee1479370235e9edc6",
}
ACP_MODELS = {"acp25", "acp50", "episode_acp_v1", "acp20k"}
RENAME_MAP = {
    "observation.images.camera_top": "observation.images.base_0_rgb",
    "observation.images.camera_wrist_left": "observation.images.left_wrist_0_rgb",
    "observation.images.camera_wrist_right": "observation.images.right_wrist_0_rgb",
}
IMAGE_KEYS = tuple(RENAME_MAP)
ANNOTATED_IMAGE_MAP = {
    "observation.images.top": "observation.images.camera_top",
    "observation.images.left_wrist": "observation.images.camera_wrist_left",
    "observation.images.right_wrist": "observation.images.camera_wrist_right",
}
TASK = "Insert the copper screw into the black sleeve."


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def evenly_pick(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=count)
    out: list[int] = []
    for pos in positions:
        value = indices[int(round(float(pos)))]
        if value not in out:
            out.append(value)
    for value in indices:
        if len(out) >= count:
            break
        if value not in out:
            out.append(value)
    return out


def split_episode_indices(split_name: str) -> list[int]:
    files = sorted((ANNOTATED_ROOT / "data").rglob("*.parquet"))
    table = pads.dataset([str(path) for path in files], format="parquet").to_table(
        columns=["episode_index", "dataset_split"]
    )
    episodes = sorted(
        {
            int(ep)
            for ep, split in zip(table["episode_index"].to_pylist(), table["dataset_split"].to_pylist())
            if str(split) == split_name
        }
    )
    expected = {"validation": 50, "test": 46}[split_name]
    if len(episodes) != expected:
        raise RuntimeError(f"{split_name} episode drift: expected {expected}, got {len(episodes)}")
    return episodes


def make_dataset(root: Path, repo_id: str, episodes: list[int], policy_cfg: PI05Config) -> LeRobotDataset:
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, metadata)
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )


def build_attempt_anchors(
    dataset: LeRobotDataset, samples_per_attempt: int, split_name: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = dataset.hf_dataset.with_format(None)
    episodes = np.asarray(raw["episode_index"], dtype=np.int64)
    frames = np.asarray(raw["frame_index"], dtype=np.int64)
    actions = np.asarray(raw["action"], dtype=np.float32)
    attempts = np.asarray(raw["logical_attempt_id"], dtype=str)
    valid = np.asarray(raw["logical_action_chunk_valid_50"], dtype=bool)
    apply_mask = np.asarray(raw["acp_apply_mask"], dtype=bool)
    indicators = np.asarray(raw["acp_indicator_attempt_v2"], dtype=np.int64)
    interventions = np.asarray(raw["complementary_info.is_intervention"], dtype=bool)
    attempt_intervened = np.asarray(raw["logical_attempt_intervened"], dtype=bool)
    outcomes = np.asarray(raw["logical_attempt_outcome"], dtype=str)
    known = np.asarray(raw["logical_attempt_outcome_known"], dtype=bool)
    splits = np.asarray(raw["dataset_split"], dtype=str)
    global_indices = np.asarray(raw["index"], dtype=np.int64)
    finite = np.isfinite(actions).all(axis=1)
    if not np.all(splits == split_name):
        raise RuntimeError(f"row outside {split_name} entered evaluation dataset")
    if not np.all(known):
        raise RuntimeError("unknown outcome entered fixed validation split")

    duration_by_episode = {int(ep): int(np.sum(episodes == ep)) for ep in np.unique(episodes)}
    candidates: dict[int, list[int]] = defaultdict(list)
    n = len(episodes)
    for i in range(n):
        if not valid[i] or not apply_mask[i] or not finite[i]:
            continue
        end = i + 49
        if end >= n or episodes[end] != episodes[i] or attempts[end] != attempts[i]:
            raise RuntimeError(f"declared valid chunk crosses attempt at local row {i}")
        if frames[end] - frames[i] != 49 or not np.all(np.diff(frames[i : end + 1]) == 1):
            raise RuntimeError(f"declared valid chunk is non-contiguous at local row {i}")
        if not np.isfinite(actions[i : end + 1]).all():
            raise RuntimeError(f"declared valid chunk contains non-finite action at local row {i}")
        candidates[int(episodes[i])].append(i)

    anchors: list[dict[str, Any]] = []
    candidate_summary: dict[str, Any] = {}
    for ep in sorted(np.unique(episodes).tolist()):
        ep_candidates = candidates[int(ep)]
        chosen = evenly_pick(ep_candidates, samples_per_attempt)
        if not chosen:
            raise RuntimeError(f"validation attempt {ep} has no legal 50-step anchor")
        candidate_summary[str(ep)] = {
            "valid_candidates": len(ep_candidates),
            "selected": len(chosen),
            "duration_frames": duration_by_episode[int(ep)],
        }
        for i in chosen:
            anchors.append(
                {
                    "dataset_local_index": int(i),
                    "global_index": int(global_indices[i]),
                    "episode_index": int(episodes[i]),
                    "frame_index": int(frames[i]),
                    "attempt_id": str(attempts[i]),
                    "attempt_duration_frames": duration_by_episode[int(ep)],
                    "acp_indicator": int(indicators[i]),
                    "is_intervention_action": bool(interventions[i]),
                    "attempt_intervened": bool(attempt_intervened[i]),
                    "outcome": str(outcomes[i]),
                }
            )
    return anchors, candidate_summary


def build_base_anchors(
    dataset: LeRobotDataset, selected_episodes: list[int], samples_per_episode: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = dataset.hf_dataset.with_format(None)
    episodes = np.asarray(raw["episode_index"], dtype=np.int64)
    frames = np.asarray(raw["frame_index"], dtype=np.int64)
    actions = np.asarray(raw["action"], dtype=np.float32)
    global_indices = np.asarray(raw["index"], dtype=np.int64)
    finite = np.isfinite(actions).all(axis=1)
    candidates: dict[int, list[int]] = defaultdict(list)
    for i in range(max(0, len(episodes) - 49)):
        end = i + 49
        if episodes[end] != episodes[i] or frames[end] - frames[i] != 49:
            continue
        if not np.all(np.diff(frames[i : end + 1]) == 1) or not np.all(finite[i : end + 1]):
            continue
        candidates[int(episodes[i])].append(i)
    anchors: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for ep in selected_episodes:
        chosen = evenly_pick(candidates[int(ep)], samples_per_episode)
        if len(chosen) != samples_per_episode:
            raise RuntimeError(f"Base retention episode {ep} lacks {samples_per_episode} anchors")
        summary[str(ep)] = {"valid_candidates": len(candidates[int(ep)]), "selected": len(chosen)}
        for i in chosen:
            anchors.append(
                {
                    "dataset_local_index": int(i),
                    "global_index": int(global_indices[i]),
                    "episode_index": int(episodes[i]),
                    "frame_index": int(frames[i]),
                }
            )
    return anchors, summary


def prompt_for_anchor(
    model_name: str, anchor: dict[str, Any], prompt_mode: str, swapped: bool = False
) -> str:
    if model_name not in ACP_MODELS:
        return TASK
    positive = True if prompt_mode == "deployment_positive" else bool(anchor["acp_indicator"])
    if swapped:
        positive = not positive
    return TASK + ("\nAdvantage: positive" if positive else "\nAdvantage: negative")


def bootstrap_attempt_ci(records: list[dict[str, Any]], field: str, seed: int) -> list[float]:
    by_attempt: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_attempt[int(record["episode_index"])].append(float(record[field]))
    means = np.asarray([np.mean(values) for _, values in sorted(by_attempt.items())], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000, dtype=np.float64)
    for i in range(draws.size):
        draws[i] = rng.choice(means, size=len(means), replace=True).mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def metric_summary(records: list[dict[str, Any]], seed: int, include_groups: bool) -> dict[str, Any]:
    fields = [
        "physical_chunk_mae",
        "physical_first_step_mae",
        "normalized_chunk_mae",
        "flow_matching_loss",
        "predicted_action_total_variation",
        "smoothness_error",
    ]
    out: dict[str, Any] = {
        "samples": len(records),
        "attempts_or_episodes": len({int(r["episode_index"]) for r in records}),
    }
    for offset, field in enumerate(fields):
        out[field] = float(np.mean([float(r[field]) for r in records]))
        out[field + "_attempt_macro"] = float(
            np.mean(
                [
                    np.mean([float(r[field]) for r in records if int(r["episode_index"]) == ep])
                    for ep in sorted({int(r["episode_index"]) for r in records})
                ]
            )
        )
        out[field + "_attempt_bootstrap_95ci"] = bootstrap_attempt_ci(records, field, seed + offset)
    if not include_groups:
        return out

    duration_median = float(np.median([r["attempt_duration_frames"] for r in records]))
    predicates = {
        "acp_positive": lambda r: int(r["acp_indicator"]) == 1,
        "acp_negative": lambda r: int(r["acp_indicator"]) == 0,
        "intervention_action": lambda r: bool(r["is_intervention_action"]),
        "non_intervention_action": lambda r: not bool(r["is_intervention_action"]),
        "intervened_attempt": lambda r: bool(r["attempt_intervened"]),
        "autonomous_attempt": lambda r: not bool(r["attempt_intervened"]),
        "long_attempt": lambda r: float(r["attempt_duration_frames"]) > duration_median,
        "short_attempt": lambda r: float(r["attempt_duration_frames"]) <= duration_median,
        "known_failure": lambda r: "failure" in str(r["outcome"]) or "abort" in str(r["outcome"]),
        "known_success": lambda r: "success" in str(r["outcome"]),
    }
    groups = {}
    for label, predicate in predicates.items():
        subset = [r for r in records if predicate(r)]
        groups[label] = {
            "samples": len(subset),
            "attempts": len({int(r["episode_index"]) for r in subset}),
            "normalized_chunk_mae": float(np.mean([r["normalized_chunk_mae"] for r in subset])) if subset else None,
            "physical_chunk_mae": float(np.mean([r["physical_chunk_mae"] for r in subset])) if subset else None,
            "flow_matching_loss": float(np.mean([r["flow_matching_loss"] for r in subset])) if subset else None,
        }
    out["duration_median_frames"] = duration_median
    out["groups"] = groups
    if any("swapped_flow_matching_loss" in r for r in records):
        matched = np.asarray([r["flow_matching_loss"] for r in records], dtype=np.float64)
        swapped = np.asarray([r["swapped_flow_matching_loss"] for r in records], dtype=np.float64)
        out["acp_qfree_ranking"] = {
            "matched_minus_swapped_flow_loss": float(np.mean(matched - swapped)),
            "matched_better_fraction": float(np.mean(matched < swapped)),
            "interpretation": "negative delta and fraction above 0.5 favor the registered matched ACP condition",
        }
    return out


def one_pass(
    policy: PI05Policy,
    preprocessor: Any,
    postprocessor: Any,
    config: PI05Config,
    batch: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    target_physical = batch["action"].detach().float().cpu()
    processed = preprocessor(batch)
    target_normalized = processed["action"].detach().float().cpu()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        flow_loss, _ = policy.forward(processed)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(1_000_000 + seed)
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
    if predicted_physical.shape != target_physical.shape or predicted_normalized.shape != target_normalized.shape:
        raise RuntimeError("prediction/target shape mismatch")
    tensors = (predicted_physical, target_physical, predicted_normalized, target_normalized)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("non-finite prediction or target")
    physical_abs = (predicted_physical - target_physical).abs()[0]
    normalized_abs = (predicted_normalized - target_normalized).abs()[0]
    pred_delta = torch.diff(predicted_physical[0], dim=0)
    target_delta = torch.diff(target_physical[0], dim=0)
    return {
        "physical_chunk_mae": float(physical_abs.mean()),
        "physical_first_step_mae": float(physical_abs[0].mean()),
        "normalized_chunk_mae": float(normalized_abs.mean()),
        "flow_matching_loss": float(flow_loss.detach().float()),
        "predicted_action_total_variation": float(pred_delta.abs().mean()),
        "smoothness_error": float((pred_delta - target_delta).abs().mean()),
    }


def evaluate_model(
    name: str,
    checkpoint: Path,
    validation_dataset: LeRobotDataset,
    validation_anchors: list[dict[str, Any]],
    base_dataset: LeRobotDataset,
    base_anchors: list[dict[str, Any]],
    anchor_sha256: str,
    part_path: Path,
    seed: int,
    acp_prompt_mode: str,
) -> dict[str, Any]:
    if part_path.is_file():
        existing = json.loads(part_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("anchor_sha256") == anchor_sha256:
            print(f"VALIDATION_MODEL_REUSE name={name}", flush=True)
            return existing

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    if not isinstance(config, PI05Config) or config.chunk_size != 50:
        raise RuntimeError(f"invalid PI05 config for {name}")
    config.device = "cuda"
    config.gradient_checkpointing = False
    config.compile_model = False
    print(f"VALIDATION_MODEL_LOAD_START name={name} checkpoint={checkpoint}", flush=True)
    policy = PI05Policy.from_pretrained(str(checkpoint), config=config, strict=True, local_files_only=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    validation_records: list[dict[str, Any]] = []
    base_records: list[dict[str, Any]] = []
    started = time.monotonic()

    for domain, dataset, anchors in (
        ("validation", validation_dataset, validation_anchors),
        ("base_retention", base_dataset, base_anchors),
    ):
        records = validation_records if domain == "validation" else base_records
        for position, anchor in enumerate(anchors):
            source_item = dataset[int(anchor["dataset_local_index"])]
            item = {
                "observation.state": source_item["observation.state"],
                "action": source_item["action"],
                "task": source_item["task"],
            }
            if domain == "validation":
                for source_key, target_key in ANNOTATED_IMAGE_MAP.items():
                    item[target_key] = source_item[source_key]
            else:
                for key in IMAGE_KEYS:
                    item[key] = source_item[key]
            if domain == "validation":
                item["task"] = prompt_for_anchor(name, anchor, acp_prompt_mode)
            else:
                item["task"] = TASK
            batch = default_collate([item])
            deterministic_seed = seed + int(anchor["global_index"])
            metrics = one_pass(policy, preprocessor, postprocessor, config, batch, deterministic_seed)
            record = {**anchor, **metrics, "deterministic_seed": deterministic_seed}
            if domain == "validation" and name in ACP_MODELS:
                swapped_item = dict(item)
                swapped_item["task"] = prompt_for_anchor(
                    name, anchor, acp_prompt_mode, swapped=True
                )
                swapped_batch = default_collate([swapped_item])
                swapped = one_pass(
                    policy, preprocessor, postprocessor, config, swapped_batch, deterministic_seed
                )
                record.update({f"swapped_{key}": value for key, value in swapped.items()})
            if not all(math.isfinite(float(value)) for key, value in record.items() if key.endswith(("mae", "loss", "variation", "error"))):
                raise RuntimeError(f"non-finite metric for {name}: {record}")
            records.append(record)
            if (position + 1) % 10 == 0 or position + 1 == len(anchors):
                payload = {
                    "model": name,
                    "anchor_sha256": anchor_sha256,
                    "validation_records": validation_records,
                    "base_retention_records": base_records,
                    "complete": False,
                }
                part_path.parent.mkdir(parents=True, exist_ok=True)
                part_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(
                    f"VALIDATION_MODEL_PROGRESS name={name} domain={domain} samples={position+1}/{len(anchors)}",
                    flush=True,
                )
    result = {
        "model": name,
        "anchor_sha256": anchor_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "model_file_size": (checkpoint / "model.safetensors").stat().st_size,
        "model_sha256_from_training_manifest": EXPECTED_SHA256.get(name),
        "validation_records": validation_records,
        "base_retention_records": base_records,
        "validation_summary": metric_summary(validation_records, seed + 10_000, True),
        "base_retention_summary": metric_summary(base_records, seed + 20_000, False),
        "elapsed_seconds": time.monotonic() - started,
        "complete": True,
    }
    part_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del policy, preprocessor, postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VALIDATION_MODEL_PASS name={name} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-attempt", type=int, default=4)
    parser.add_argument("--base-episodes", type=int, default=32)
    parser.add_argument("--base-samples-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--acp-prompt-mode",
        choices=("label_matched", "deployment_positive"),
        default=None,
    )
    parser.add_argument("--models", nargs="+", choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--anchor-output", type=Path, default=None)
    parser.add_argument("--part-dir", type=Path, default=None)
    args = parser.parse_args()
    register_third_party_plugins()
    torch.set_float32_matmul_precision("high")

    reference_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINTS["acp25"]))
    if not isinstance(reference_cfg, PI05Config) or reference_cfg.chunk_size != 50:
        raise RuntimeError("reference checkpoint is not a 50-step PI05 policy")
    if args.output is None:
        args.output = ROOT / f"reports/policy_{args.split}_evaluation_v2.json"
    if args.anchor_output is None:
        args.anchor_output = ROOT / f"manifests/policy_{args.split}_evaluation_anchors_v2.json"
    if args.part_dir is None:
        args.part_dir = ROOT / f"reports/policy_{args.split}_evaluation_parts_v2"
    if args.acp_prompt_mode is None:
        args.acp_prompt_mode = (
            "label_matched" if args.split == "validation" else "deployment_positive"
        )
    split_episodes = split_episode_indices(args.split)
    val_dataset = make_dataset(
        ANNOTATED_ROOT, "local/attempt-acp-annotated-v2", split_episodes, reference_cfg
    )
    val_anchors, val_summary = build_attempt_anchors(
        val_dataset, args.samples_per_attempt, args.split
    )

    rng = np.random.default_rng(args.seed)
    base_episodes = sorted(rng.choice(np.arange(558), size=args.base_episodes, replace=False).tolist())
    base_dataset = make_dataset(BASE_ROOT, "local/pi05-full558-v3", base_episodes, reference_cfg)
    base_anchors, base_summary = build_base_anchors(
        base_dataset, base_episodes, args.base_samples_per_episode
    )
    anchor_contract = {
        "schema": "attempt_policy_ratio_validation_anchors/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "evaluation_split": args.split,
        "test_data_accessed": args.split == "test",
        "episode_indices": split_episodes,
        "attempts": len(split_episodes),
        "anchors": val_anchors,
        "candidate_summary": val_summary,
        "base_retention_probe_note": "fixed deterministic sample from replay data; measures relative forgetting, not held-out generalization",
        "base_episode_indices": base_episodes,
        "base_anchors": base_anchors,
        "base_candidate_summary": base_summary,
        "chunk_size": 50,
        "same_anchors_all_models": True,
        "same_diffusion_noise_all_models": True,
        "acp_prompt_mode": args.acp_prompt_mode,
    }
    canonical = json.dumps(anchor_contract, sort_keys=True, separators=(",", ":")).encode()
    anchor_sha256 = hashlib.sha256(canonical).hexdigest()
    anchor_contract["anchor_sha256"] = anchor_sha256
    args.anchor_output.parent.mkdir(parents=True, exist_ok=True)
    args.anchor_output.write_text(json.dumps(anchor_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"EVALUATION_ANCHOR_GATE_OK split={args.split} attempts={len(split_episodes)} samples={len(val_anchors)} "
        f"base_episodes={len(base_episodes)} base_samples={len(base_anchors)} sha256={anchor_sha256}",
        flush=True,
    )

    models = {}
    for name in args.models:
        models[name] = evaluate_model(
            name,
            CHECKPOINTS[name],
            val_dataset,
            val_anchors,
            base_dataset,
            base_anchors,
            anchor_sha256,
            args.part_dir / f"{name}.json",
            args.seed,
            args.acp_prompt_mode,
        )
    result = {
        "schema": "attempt_policy_ratio_smoke_validation/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": args.split,
        "held_out_test_used_for_selection": False,
        "choices_frozen_before_test": args.split == "test",
        "acp_prompt_mode": args.acp_prompt_mode,
        "anchor_manifest": str(args.anchor_output),
        "anchor_sha256": anchor_sha256,
        "model_order": args.models,
        "models": models,
        "caveat": "offline diagnostics are not robot success-rate estimates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VALIDATION_COMPARISON_PASS output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
