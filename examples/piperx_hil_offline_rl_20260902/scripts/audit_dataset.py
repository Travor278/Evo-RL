#!/usr/bin/env python3
"""Strict audit for the PiperX Evo-RL HIL LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_FILES = 390
EXPECTED_BYTES = 8_162_275_443
EXPECTED_EPISODES = 76
EXPECTED_FRAMES = 431_920
EXPECTED_FPS = 30
EXPECTED_REVISION = "18016184b09929643b6bd055b3f5833bfb2e7b85"
CAMERA_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
CONTROL_STATE = {
    0: "POLICY",
    1: "HUMAN_BLEND",
    2: "HUMAN",
    3: "POLICY_BLEND",
    4: "SAFETY_HOLD",
    5: "ABORTED",
}
ACTION_SOURCE = {0: "POLICY", 1: "PEDAL", 2: "KEYBOARD", 3: "WEB", 4: "SAFETY_HOLD"}
POLICY_ACTION_ROLE = {0: "NONE", 1: "CONTROL", 2: "SHADOW"}
POLICY_INVALID_REASON = {
    0: "VALID",
    1: "NOT_REQUESTED",
    2: "WAITING_FOR_CHUNK",
    3: "TIMEOUT",
    4: "STALE_EPOCH",
    5: "WORKER_FAULT",
    6: "SHADOW_SUPPRESSED",
}
INTERVENTION_CAUSE = {
    0: "UNKNOWN",
    1: "PREDICTED_FAILURE",
    2: "MANIPULATION_FAILURE",
    3: "OBJECT_SLIP",
    4: "COLLISION_RISK",
    5: "WORKSPACE_LIMIT",
    6: "POLICY_STALL",
    7: "OPERATOR_CORRECTION",
    8: "OTHER",
}

FIELD_CONTRACT = {
    "observation_state": ("observation.state", None),
    "executed_action": ("action", "complementary_info.action_valid"),
    "policy_action": ("complementary_info.policy_action", "complementary_info.policy_action_valid"),
    "operator_action": (
        "complementary_info.operator_action",
        "complementary_info.operator_action_valid",
    ),
    "requested_action": (
        "complementary_info.requested_action",
        "complementary_info.requested_action_valid",
    ),
    "action_source": ("complementary_info.action_source", None),
    "control_state": ("complementary_info.control_state", None),
    "is_intervention": ("complementary_info.is_intervention", None),
    "intervention_id": ("complementary_info.intervention_id", None),
    "intervention_source": ("complementary_info.intervention_source", None),
    "blend_alpha": ("complementary_info.blend_alpha", None),
    "transition_valid": ("complementary_info.transition_valid", None),
    "transition_invalid_reason": (None, None),
    "segment_id": ("complementary_info.segment_id", None),
    "episode_success": ("meta/episodes:episode_success", None),
    "termination_reason": ("meta/episodes:termination_reason", None),
    "safety_clipped": ("complementary_info.safety_clipped", None),
    "safety_reason": ("complementary_info.safety_reason", None),
    "delta_norm": ("complementary_info.executed_action_delta_norm", None),
    "timestamp": ("timestamp", None),
    "frame_index": ("frame_index", None),
    "episode_index": ("episode_index", None),
}


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def list_array(table: pa.Table, key: str, width: int = 14) -> np.ndarray:
    array = table[key].combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64).reshape(len(array), width)


def scalar_array(table: pa.Table, key: str, dtype: Any | None = None) -> np.ndarray:
    values = table[key].combine_chunks().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def vector_stats(arrays: list[np.ndarray]) -> dict[str, Any]:
    if not arrays:
        return {"rows": 0}
    array = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
    finite = np.isfinite(array)
    result: dict[str, Any] = {
        "rows": int(array.shape[0]),
        "dimension": int(array.shape[1]),
        "nan_count": np.isnan(array).sum(axis=0).astype(int).tolist(),
        "inf_count": np.isinf(array).sum(axis=0).astype(int).tolist(),
        "finite_count": finite.sum(axis=0).astype(int).tolist(),
    }
    safe = np.where(finite, array, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        result.update(
            {
                "min": np.nanmin(safe, axis=0).tolist(),
                "max": np.nanmax(safe, axis=0).tolist(),
                "mean": np.nanmean(safe, axis=0).tolist(),
                "std": np.nanstd(safe, axis=0).tolist(),
            }
        )
    mean = np.asarray(result["mean"])
    std = np.asarray(result["std"])
    z = np.abs((array - mean) / np.where(std > 0, std, np.nan))
    result["outlier_abs_z_gt_8"] = np.nansum(z > 8.0, axis=0).astype(int).tolist()
    return result


def decode_video(path: Path) -> dict[str, Any]:
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=600)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "probe_returncode": probe.returncode,
        "probe_stderr": probe.stderr.strip(),
    }
    if probe.returncode == 0:
        payload = json.loads(probe.stdout)
        stream = payload.get("streams", [{}])[0]
        record.update(stream)
        for key in ("nb_frames", "nb_read_frames"):
            try:
                record[key] = int(stream[key])
            except (KeyError, TypeError, ValueError):
                record[key] = None
        for key in ("r_frame_rate", "avg_frame_rate"):
            try:
                record[f"{key}_float"] = float(Fraction(stream[key]))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                record[f"{key}_float"] = None

    decode_cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        "-threads",
        "1",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    decoded = subprocess.run(decode_cmd, capture_output=True, text=True, timeout=1800)
    progress = {}
    for line in decoded.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            progress[key] = value
    record.update(
        {
            "decode_returncode": decoded.returncode,
            "decode_stderr": decoded.stderr.strip(),
            "decoded_frames": int(progress["frame"]) if progress.get("frame", "").isdigit() else None,
            "decode_progress": progress.get("progress"),
        }
    )
    return record


@dataclass
class Group:
    name: str
    episodes: list[int]
    frames: int
    failures: int
    autonomous_successes: int
    autonomous_failures: int


def choose_group_split(groups: list[Group], seed: int, total_episodes: int, total_frames: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    best: tuple[float, dict[str, list[str]]] | None = None
    names = [group.name for group in groups]
    lookup = {group.name: group for group in groups}
    for _ in range(100_000):
        shuffled = names[:]
        rng.shuffle(shuffled)
        assignment = {"train": [], "val": [], "test": []}
        counts = {split: [0, 0, 0, 0] for split in assignment}
        for name in shuffled:
            group = lookup[name]
            candidates = []
            for split, target_ratio in (("val", 0.1), ("test", 0.1), ("train", 0.8)):
                ep_after = counts[split][0] + len(group.episodes)
                fr_after = counts[split][1] + group.frames
                target_ep = total_episodes * target_ratio
                target_fr = total_frames * target_ratio
                cost = abs(ep_after - target_ep) / max(target_ep, 1) + abs(fr_after - target_fr) / max(target_fr, 1)
                if split != "train" and counts[split][0] >= math.ceil(target_ep):
                    cost += 4.0
                candidates.append((cost + rng.random() * 0.01, split))
            split = min(candidates)[1]
            assignment[split].append(name)
            counts[split][0] += len(group.episodes)
            counts[split][1] += group.frames
            counts[split][2] += group.failures
            counts[split][3] += group.autonomous_successes
        if counts["val"][2] < 1 or counts["test"][2] < 1 or counts["train"][2] < 2:
            continue
        if counts["val"][3] < 1 or counts["train"][3] < 1:
            continue
        score = 0.0
        for split, ratio in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
            score += abs(counts[split][0] / total_episodes - ratio) * 5
            score += abs(counts[split][1] / total_frames - ratio) * 2
        if best is None or score < best[0]:
            best = (score, {key: sorted(value) for key, value in assignment.items()})
    if best is None:
        raise RuntimeError("Could not construct group-disjoint split with failure coverage in every split")
    return best[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--video-workers", type=int, default=4)
    parser.add_argument("--skip-video-decode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    output_root = args.output_root.resolve()
    started = datetime.now(timezone.utc)
    progress(f"audit_start dataset={root}")
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    download_manifest = json.loads((root / "download_manifest.json").read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in (root / "meta/source_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    source_by_episode = {int(row["episode_index"]): row for row in source_rows}
    provenance = json.loads((root / "meta/collection_provenance.json").read_text(encoding="utf-8"))
    collection_to_session = {}
    for collection in provenance["collections"]:
        policy_hil = collection.get("policy_hil") or {}
        collection_to_session[collection["collection_uuid"]] = policy_hil.get("session_id") or collection["collection_uuid"]

    sha_rows = [json.loads(line) for line in (root / "meta/file_sha256.jsonl").read_text(encoding="utf-8").splitlines() if line]
    sha_results = []
    progress(f"sha256_verify_start files={len(sha_rows)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(sha256_file, root / row["path"]): row for row in sha_rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            path = root / row["path"]
            actual_sha = future.result() if path.is_file() else None
            sha_results.append(
                {
                    "path": row["path"],
                    "expected_size": row["size"],
                    "actual_size": path.stat().st_size if path.is_file() else None,
                    "expected_sha256": row["sha256"],
                    "actual_sha256": actual_sha,
                    "ok": path.is_file() and path.stat().st_size == row["size"] and actual_sha == row["sha256"],
                }
            )
    sha_results.sort(key=lambda item: item["path"])
    progress(f"sha256_verify_done ok={sum(item['ok'] for item in sha_results)}/{len(sha_results)}")

    meta_files = sorted((root / "meta/episodes/chunk-000").glob("*.parquet"))
    data_files = sorted((root / "data/chunk-000").glob("*.parquet"))
    meta_table = pq.read_table(meta_files)
    meta = meta_table.to_pydict()
    meta_by_episode = {int(meta["episode_index"][i]): {key: meta[key][i] for key in meta} for i in range(meta_table.num_rows)}

    schema = pq.ParquetFile(data_files[0]).schema_arrow
    schema_map = {field.name: str(field.type) for field in schema}
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    episode_records: list[dict[str, Any]] = []
    global_counts: dict[str, Counter] = defaultdict(Counter)
    transition_reason_counts: Counter = Counter()
    total_rows = 0

    action_specs = {
        "observation.state": None,
        "action": "complementary_info.action_valid",
        "complementary_info.policy_action": "complementary_info.policy_action_valid",
        "complementary_info.operator_action": "complementary_info.operator_action_valid",
        "complementary_info.requested_action": "complementary_info.requested_action_valid",
    }

    for path in data_files:
        table = pq.read_table(path)
        n = table.num_rows
        total_rows += n
        episode_index_values = scalar_array(table, "episode_index", np.int64)
        episode_index = int(episode_index_values[0])
        frame_index = scalar_array(table, "frame_index", np.int64)
        timestamps = scalar_array(table, "timestamp", np.float64)
        segments = scalar_array(table, "complementary_info.segment_id", np.int64)
        transition_valid = scalar_array(table, "complementary_info.transition_valid", bool)
        control_state = scalar_array(table, "complementary_info.control_state", np.int64)
        action_source = scalar_array(table, "complementary_info.action_source", np.int64)
        intervention = scalar_array(table, "complementary_info.is_intervention", bool)
        safety_clipped = scalar_array(table, "complementary_info.safety_clipped", bool)
        safety_reason = scalar_array(table, "complementary_info.safety_reason", np.int64)
        policy_role = scalar_array(table, "complementary_info.policy_action_role", np.int64)
        policy_invalid = scalar_array(table, "complementary_info.policy_action_invalid_reason", np.int64)
        intervention_cause = scalar_array(table, "complementary_info.intervention_cause", np.int64)

        for value in control_state:
            global_counts["control_state"][CONTROL_STATE.get(int(value), f"UNKNOWN_{value}")] += 1
        for value in action_source:
            global_counts["action_source"][ACTION_SOURCE.get(int(value), f"UNKNOWN_{value}")] += 1
        for value in policy_role:
            global_counts["policy_action_role"][POLICY_ACTION_ROLE.get(int(value), f"UNKNOWN_{value}")] += 1
        for value in policy_invalid:
            global_counts["policy_action_invalid_reason"][POLICY_INVALID_REASON.get(int(value), f"UNKNOWN_{value}")] += 1
        for value in intervention_cause:
            global_counts["intervention_cause"][INTERVENTION_CAUSE.get(int(value), f"UNKNOWN_{value}")] += 1

        invalid_indices = np.flatnonzero(~transition_valid)
        episode_invalid_reasons = Counter()
        for index in invalid_indices:
            if index == n - 1:
                reason = "episode_terminal"
            elif segments[index] != segments[index + 1]:
                reason = "segment_boundary"
            else:
                reason = "unexplained_invalid"
            transition_reason_counts[reason] += 1
            episode_invalid_reasons[reason] += 1

        segment_last = np.r_[segments[1:] != segments[:-1], True]
        valid_cross_segment = int(np.sum(transition_valid[:-1] & (segments[:-1] != segments[1:])))
        segment_terminal_valid = int(np.sum(transition_valid[segment_last]))
        timestamp_diff = np.diff(timestamps)

        action_coverage = {}
        for field, valid_field in action_specs.items():
            values = list_array(table, field)
            mask = np.ones(n, dtype=bool) if valid_field is None else scalar_array(table, valid_field, bool)
            arrays[field].append(values[mask].astype(np.float32, copy=False))
            selected = values[mask]
            action_coverage[field] = {
                "valid_rows": int(mask.sum()),
                "invalid_rows": int((~mask).sum()),
                "nan": int(np.isnan(selected).sum()),
                "inf": int(np.isinf(selected).sum()),
            }

        camera_sequences = {}
        for camera in ("left_wrist", "right_wrist", "top"):
            sequence = scalar_array(table, f"complementary_info.camera_capture_sequence.{camera}", np.int64)
            diff = np.diff(sequence)
            camera_sequences[camera] = {
                "non_monotonic": int(np.sum(diff <= 0)),
                "gaps": int(np.sum(diff > 1)),
                "max_step": int(diff.max(initial=0)),
                "reported_gap_count_max": int(
                    scalar_array(table, f"complementary_info.camera_gap_count.{camera}", np.int64).max(initial=0)
                ),
            }

        meta_row = meta_by_episode[episode_index]
        source_row = source_by_episode[episode_index]
        group_id = collection_to_session.get(source_row["source_dir"], source_row["source_dir"])
        episode_records.append(
            {
                "episode_index": episode_index,
                "episode_uuid": meta_row["episode_uuid"],
                "collection_uuid": source_row["source_dir"],
                "group_id": group_id,
                "rows": n,
                "meta_length": int(meta_row["length"]),
                "episode_success": bool(meta_row["episode_success"]),
                "termination_reason": meta_row["termination_reason"],
                "episode_valid": bool(meta_row["episode_valid"]),
                "timestamp_monotonic": bool(np.all(timestamp_diff > 0)),
                "timestamp_nonpositive_steps": int(np.sum(timestamp_diff <= 0)),
                "timestamp_long_gaps_gt_50ms": int(np.sum(timestamp_diff > 0.05)),
                "timestamp_max_gap_s": float(timestamp_diff.max(initial=0)),
                "duplicate_frame_index": int(n - np.unique(frame_index).size),
                "frame_index_contiguous": bool(np.array_equal(frame_index, np.arange(n))),
                "episode_index_uniform": bool(np.all(episode_index_values == episode_index)),
                "transition_valid": int(transition_valid.sum()),
                "transition_invalid": int((~transition_valid).sum()),
                "transition_invalid_reasons_derived": dict(episode_invalid_reasons),
                "valid_cross_segment": valid_cross_segment,
                "segment_terminal_marked_valid": segment_terminal_valid,
                "terminal_valid": bool(transition_valid[-1]),
                "segments": int(np.unique(segments).size),
                "intervention_count": int(meta_row["intervention_count"]),
                "intervention_frames": int(intervention.sum()),
                "intervention_duration_frames_meta": int(meta_row["intervention_duration_frames"]),
                "intervention_duration_s": float(meta_row["intervention_duration_frames"] / EXPECTED_FPS),
                "intervention_fraction": float(intervention.mean()),
                "control_state_counts": {
                    CONTROL_STATE.get(int(key), f"UNKNOWN_{key}"): int(value)
                    for key, value in zip(*np.unique(control_state, return_counts=True))
                },
                "safety_clipped": int(safety_clipped.sum()),
                "safety_reason_counts": {
                    str(int(key)): int(value) for key, value in zip(*np.unique(safety_reason, return_counts=True))
                },
                "deadline_missed": int(
                    scalar_array(table, "complementary_info.deadline_missed", bool).sum()
                ),
                "action_coverage": action_coverage,
                "camera_sequences": camera_sequences,
                "source_review_status": source_row.get("review_status"),
            }
        )
        if len(episode_records) % 10 == 0 or len(episode_records) == len(data_files):
            progress(f"parquet_audit_progress episodes={len(episode_records)}/{len(data_files)}")

    episode_records.sort(key=lambda item: item["episode_index"])
    vector_statistics = {field: vector_stats(values) for field, values in arrays.items()}

    video_jobs = []
    for episode in episode_records:
        episode_index = episode["episode_index"]
        for camera_key in CAMERA_KEYS:
            video_jobs.append(
                (
                    episode_index,
                    camera_key,
                    root / "videos" / camera_key / "chunk-000" / f"file-{episode_index:03d}.mp4",
                )
            )
    video_results: list[dict[str, Any]] = []
    if not args.skip_video_decode:
        progress(f"video_decode_start files={len(video_jobs)} workers={args.video_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.video_workers) as pool:
            futures = {pool.submit(decode_video, path): (episode, camera) for episode, camera, path in video_jobs}
            for future in concurrent.futures.as_completed(futures):
                episode, camera = futures[future]
                record = future.result()
                record.update({"episode_index": episode, "camera_key": camera})
                expected_rows = episode_records[episode]["rows"]
                record["expected_frames"] = expected_rows
                record["decoded_minus_parquet"] = (
                    record["decoded_frames"] - expected_rows if record.get("decoded_frames") is not None else None
                )
                video_results.append(record)
                if len(video_results) % 10 == 0 or len(video_results) == len(video_jobs):
                    progress(f"video_decode_progress files={len(video_results)}/{len(video_jobs)}")
        video_results.sort(key=lambda item: (item["episode_index"], item["camera_key"]))
        progress("video_decode_done")

    groups_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episode_records:
        groups_map[episode["group_id"]].append(episode)
    groups = [
        Group(
            name=name,
            episodes=[item["episode_index"] for item in items],
            frames=sum(item["rows"] for item in items),
            failures=sum(not item["episode_success"] for item in items),
            autonomous_successes=sum(item["episode_success"] and item["intervention_count"] == 0 for item in items),
            autonomous_failures=sum((not item["episode_success"]) and item["intervention_count"] == 0 for item in items),
        )
        for name, items in groups_map.items()
    ]
    split_groups = choose_group_split(groups, args.seed, len(episode_records), total_rows)
    split_episode_ids = {
        split: sorted(ep for group_name in group_names for ep in next(group.episodes for group in groups if group.name == group_name))
        for split, group_names in split_groups.items()
    }

    bookkeeping = {"download_manifest.json", "DOWNLOAD_COMPLETE"}
    primary_files = [path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts and path.name not in bookkeeping]
    primary_bytes = sum(path.stat().st_size for path in primary_files)
    cache_files = [path for path in (root / ".cache").rglob("*") if path.is_file()]
    cache_summary = {
        "files": len(cache_files),
        "bytes": sum(path.stat().st_size for path in cache_files),
        "locks": sum(path.name.endswith(".lock") for path in cache_files),
        "incomplete": sum(path.name.endswith(".incomplete") for path in cache_files),
        "nonzero_incomplete": sum(path.name.endswith(".incomplete") and path.stat().st_size > 0 for path in cache_files),
    }
    sha_manifest_paths = {row["path"] for row in sha_rows}
    primary_not_in_sha_manifest = []
    for path in primary_files:
        relative = path.relative_to(root).as_posix()
        if relative not in sha_manifest_paths:
            primary_not_in_sha_manifest.append(
                {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    primary_not_in_sha_manifest.sort(key=lambda item: item["path"])

    video_bad = [
        item for item in video_results
        if item.get("decode_returncode") != 0 or item.get("decoded_minus_parquet") != 0
    ]
    gates = {
        "revision_matches": download_manifest.get("revision") == EXPECTED_REVISION,
        "primary_file_count_matches": len(primary_files) == EXPECTED_FILES,
        "primary_bytes_match": primary_bytes == EXPECTED_BYTES,
        "episode_count_matches": len(episode_records) == EXPECTED_EPISODES,
        "frame_count_matches": total_rows == EXPECTED_FRAMES,
        "sha256_manifest_all_ok": len(sha_results) == len(sha_rows) and all(item["ok"] for item in sha_results),
        "all_episode_meta_lengths_match": all(item["rows"] == item["meta_length"] for item in episode_records),
        "timestamps_strictly_monotonic": all(item["timestamp_monotonic"] for item in episode_records),
        "frame_indices_unique_contiguous": all(item["frame_index_contiguous"] for item in episode_records),
        "no_cross_segment_valid_transition": all(item["valid_cross_segment"] == 0 for item in episode_records),
        "all_segment_terminals_invalid": all(item["segment_terminal_marked_valid"] == 0 for item in episode_records),
        "state_action_dimension_14": str(schema_map.get("observation.state", "")).endswith("[14]") and str(schema_map.get("action", "")).endswith("[14]"),
        "all_videos_decode_exactly": (not args.skip_video_decode) and not video_bad and len(video_results) == EXPECTED_EPISODES * len(CAMERA_KEYS),
    }

    inventory = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_started_at_utc": started.isoformat(),
        "dataset_root": str(root),
        "repo_id": download_manifest.get("repo_id"),
        "revision": download_manifest.get("revision"),
        "info": info,
        "primary_content": {"files": len(primary_files), "bytes": primary_bytes},
        "local_bookkeeping": [
            {"path": name, "size": (root / name).stat().st_size, "mtime_ns": (root / name).stat().st_mtime_ns}
            for name in sorted(bookkeeping) if (root / name).is_file()
        ],
        "cache_summary": cache_summary,
        "sha256_manifest_entries": len(sha_rows),
        "primary_files_not_in_sha256_manifest": primary_not_in_sha_manifest,
        "parquet_schema": schema_map,
        "field_contract": {
            key: {"actual": actual, "validity_field": validity, "present": actual is not None and (actual.startswith("meta/") or actual in schema_map)}
            for key, (actual, validity) in FIELD_CONTRACT.items()
        },
        "totals": {
            "episodes": len(episode_records),
            "frames": total_rows,
            "duration_s": total_rows / EXPECTED_FPS,
            "fps": info.get("fps"),
            "success": sum(item["episode_success"] for item in episode_records),
            "failure": sum(not item["episode_success"] for item in episode_records),
            "autonomous_success": sum(item["episode_success"] and item["intervention_count"] == 0 for item in episode_records),
            "autonomous_failure": sum((not item["episode_success"]) and item["intervention_count"] == 0 for item in episode_records),
            "hil_recovered_success": sum(item["episode_success"] and item["intervention_count"] > 0 for item in episode_records),
            "hil_unrecovered_failure": sum((not item["episode_success"]) and item["intervention_count"] > 0 for item in episode_records),
            "failure_frames": sum(item["rows"] for item in episode_records if not item["episode_success"]),
            "failure_frame_fraction": sum(item["rows"] for item in episode_records if not item["episode_success"]) / total_rows,
            "intervention_events": sum(item["intervention_count"] for item in episode_records),
            "intervention_frames": sum(item["intervention_frames"] for item in episode_records),
            "intervention_duration_frames_meta": sum(item["intervention_duration_frames_meta"] for item in episode_records),
            "safety_clipped": sum(item["safety_clipped"] for item in episode_records),
            "transition_invalid_reasons_derived": dict(transition_reason_counts),
        },
        "categorical_counts": {key: dict(value) for key, value in global_counts.items()},
        "vector_statistics": vector_statistics,
        "episodes": episode_records,
        "videos": video_results,
        "video_failures": video_bad,
        "sha256_files": sha_results,
        "gates": gates,
        "split": {"seed": args.seed, "group_key": "policy_hil.session_id", "groups": split_groups, "episodes": split_episode_ids},
    }
    json_dump(output_root / "manifests/dataset_inventory.json", inventory)

    manifest_common = {
        "repo_id": download_manifest.get("repo_id"),
        "revision": download_manifest.get("revision"),
        "seed": args.seed,
        "split_method": "group-disjoint randomized search; group=policy_hil.session_id; failure coverage required",
    }
    for split, episode_ids in split_episode_ids.items():
        payload = {
            **manifest_common,
            "split": split,
            "groups": split_groups[split],
            "episode_indices": episode_ids,
            "episodes": [item for item in episode_records if item["episode_index"] in episode_ids],
        }
        payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        json_dump(output_root / f"manifests/{split}_episodes.json", payload)
    json_dump(output_root / "manifests/excluded_episodes.json", {**manifest_common, "episodes": [], "reason": "none; all 76 source episodes retained pending video gate"})

    field_rows = []
    for semantic, (actual, validity) in FIELD_CONTRACT.items():
        present = actual is not None and (actual.startswith("meta/") or actual in schema_map)
        note = ""
        if semantic == "executed_action":
            note = "Dataset contract defines `action` as the action actually executed by the robot."
        elif semantic == "transition_invalid_reason":
            note = "No explicit column; audit derives segment_boundary/episode_terminal from segment_id and frame order."
        field_rows.append([semantic, actual or "absent", schema_map.get(actual, "meta/episodes" if present else "—"), validity or "—", present, note])
    coverage_md = "# Field coverage\n\n" + md_table(
        ["Semantic field", "Actual field", "Type", "Validity mask", "Present", "Notes"], field_rows
    ) + "\n\n## Enum mappings\n\n" + md_table(
        ["Field", "Code mapping"],
        [["control_state", json.dumps(CONTROL_STATE, sort_keys=True)], ["action_source", json.dumps(ACTION_SOURCE, sort_keys=True)], ["policy_action_role", json.dumps(POLICY_ACTION_ROLE, sort_keys=True)], ["policy_action_invalid_reason", json.dumps(POLICY_INVALID_REASON, sort_keys=True)], ["intervention_cause", json.dumps(INTERVENTION_CAUSE, sort_keys=True)]],
    ) + "\n"
    (output_root / "reports/field_coverage.md").write_text(coverage_md, encoding="utf-8")

    gate_rows = [[key, "PASS" if value else "FAIL"] for key, value in gates.items()]
    split_rows = []
    for split, ids in split_episode_ids.items():
        items = [item for item in episode_records if item["episode_index"] in ids]
        split_rows.append([split, len(ids), sum(item["rows"] for item in items), sum(item["episode_success"] for item in items), sum(not item["episode_success"] for item in items), sum(item["episode_success"] and item["intervention_count"] == 0 for item in items), sum((not item["episode_success"]) and item["intervention_count"] == 0 for item in items), len(split_groups[split])])
    episode_rows = [
        [item["episode_index"], item["rows"], "success" if item["episode_success"] else "failure", item["termination_reason"], item["intervention_count"], item["intervention_frames"], item["segments"], item["timestamp_long_gaps_gt_50ms"], item["transition_invalid"], item["safety_clipped"]]
        for item in episode_records
    ]
    audit_md = f"""# Dataset audit

- Dataset: `{download_manifest.get('repo_id')}`
- Revision: `{download_manifest.get('revision')}`
- Primary content: **{len(primary_files)} files / {primary_bytes} bytes**
- Episodes / frames / FPS: **{len(episode_records)} / {total_rows} / {info.get('fps')}**
- Duration: **{total_rows / EXPECTED_FPS:.3f} seconds ({total_rows / EXPECTED_FPS / 3600:.3f} hours)**
- Outcomes: **{sum(item['episode_success'] for item in episode_records)} success / {sum(not item['episode_success'] for item in episode_records)} failure**
- Autonomous rollouts: **{sum(item['episode_success'] and item['intervention_count'] == 0 for item in episode_records)} success / {sum((not item['episode_success']) and item['intervention_count'] == 0 for item in episode_records)} failure**
- HIL rollouts: **{sum(item['episode_success'] and item['intervention_count'] > 0 for item in episode_records)} recovered success / {sum((not item['episode_success']) and item['intervention_count'] > 0 for item in episode_records)} unrecovered failure**
- Failure-frame support: **{sum(item['rows'] for item in episode_records if not item['episode_success'])} / {total_rows} ({100 * sum(item['rows'] for item in episode_records if not item['episode_success']) / total_rows:.2f}%)**
- Intervention: **{sum(item['intervention_count'] for item in episode_records)} events / {sum(item['intervention_frames'] for item in episode_records)} frames**
- Cache evidence retained: {cache_summary['locks']} lock files, {cache_summary['incomplete']} incomplete files ({cache_summary['nonzero_incomplete']} non-zero).
- SHA256 manifest: **{len(sha_rows)} declared files**, all checked individually; undeclared primary files are recorded separately in `dataset_inventory.json` with independently computed hashes.

## Gates

{md_table(['Gate', 'Result'], gate_rows)}

## Group-disjoint split

{md_table(['Split', 'Episodes', 'Frames', 'Success', 'Failure', 'Autonomous success', 'Autonomous failure', 'Groups'], split_rows)}

The split groups by capture `policy_hil.session_id`, not by frame. No session appears in more than one split. Only three episodes are autonomous 5/5 successes and seven are autonomous failures; the other 66 successes required intervention. This is adequate for an official HIL/ACP recovery-training smoke run, but not for interpreting the value model as a well-estimated autonomous-completion probability. Official value training also samples frames uniformly, so the short early-failure trajectories contribute only a small fraction of updates.

## Control-state frame counts

{md_table(['State', 'Frames'], [[key, value] for key, value in sorted(global_counts['control_state'].items())])}

## Derived invalid-transition reasons

{md_table(['Reason', 'Frames'], [[key, value] for key, value in sorted(transition_reason_counts.items())])}

There is no explicit `transition_invalid_reason` column. Reasons above are deterministic audit derivations. `action` is treated as the executed action; policy/operator/requested actions remain separate evidence streams.

## Episode summary

{md_table(['Episode', 'Frames', 'Outcome', 'Termination', 'Interventions', 'Intervention frames', 'Segments', '>50ms gaps', 'Invalid TD', 'Safety clipped'], episode_rows)}
"""
    (output_root / "reports/dataset_audit.md").write_text(audit_md, encoding="utf-8")

    print(json.dumps({"gates": gates, "totals": inventory["totals"], "split": split_rows, "video_failures": len(video_bad)}, indent=2, sort_keys=True))
    progress("audit_done")
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
