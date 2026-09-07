#!/usr/bin/env python3
"""Build the attempt-aware v2 policy replay view without copying video payloads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROMPT = "Insert the copper screw into the black sleeve."
STATE = "observation.state"
ACTION = "action"
ATTEMPT_ID = "logical_attempt_id"
VALID_CHUNK = "logical_action_chunk_valid_50"
INTERVENTION = "complementary_info.is_intervention"
BASE_VIDEO_KEYS = (
    "observation.images.camera_top",
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
)
ATTEMPT_TO_BASE_VIDEO_KEY = {
    "observation.images.top": "observation.images.camera_top",
    "observation.images.left_wrist": "observation.images.camera_wrist_left",
    "observation.images.right_wrist": "observation.images.camera_wrist_right",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scalar_numpy(table: pa.Table, key: str, dtype) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=dtype).reshape(-1)


def vector_numpy(table: pa.Table, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, width):
        raise ValueError(f"{key} expected {(table.num_rows, width)}, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value in {key}.")
    return values


def fixed_float_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), width
    )


def source_data_path(root: Path, info: dict, episode_index: int) -> Path:
    return root / info["data_path"].format(chunk_index=0, file_index=episode_index)


def source_video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
    return root / info["video_path"].format(
        video_key=video_key, chunk_index=0, file_index=episode_index
    )


def add_video_links(
    *,
    output_root: Path,
    output_episode: int,
    source_root: Path,
    source_info: dict,
    source_episode: int,
    key_map: dict[str, str],
) -> dict[str, dict[str, int]]:
    metadata = {}
    for source_key, output_key in key_map.items():
        source = source_video_path(source_root, source_info, source_key, source_episode)
        resolved = source.resolve(strict=True)
        destination = (
            output_root
            / "videos"
            / output_key
            / "chunk-000"
            / f"file-{output_episode:03d}.mp4"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(resolved, destination)
        metadata[output_key] = {"chunk_index": 0, "file_index": output_episode}
    return metadata


def policy_columns(
    table: pa.Table,
    *,
    source_kind: str,
    source_episode: int,
    acp_field: str,
    apply_mask_field: str,
    replay_source_field: str,
) -> dict[str, pa.Array]:
    rows = table.num_rows
    frame_index = scalar_numpy(table, "frame_index", np.int64)
    if source_kind == "attempt":
        indicator = scalar_numpy(table, acp_field, np.int64)
        apply_mask = scalar_numpy(table, apply_mask_field, np.int64)
        attempt_ids = np.asarray(table[ATTEMPT_ID].combine_chunks().to_pylist(), dtype=str)
        valid_chunk = scalar_numpy(table, VALID_CHUNK, np.bool_)
        intervention = scalar_numpy(table, INTERVENTION, np.bool_)
        if not np.isin(indicator, [0, 1]).all() or not np.isin(apply_mask, [0, 1]).all():
            raise ValueError("Attempt ACP indicator/apply mask must be binary.")
        if np.any(indicator[apply_mask == 0] != 0):
            raise ValueError("ACP indicator must be zero when apply mask is zero.")
        if len(set(attempt_ids.tolist())) != 1 or not attempt_ids[0]:
            raise ValueError(f"Episode {source_episode} must contain one non-empty logical attempt id.")
        source_values = np.ones(rows, dtype=np.int64)
    elif source_kind == "base":
        indicator = np.zeros(rows, dtype=np.int64)
        apply_mask = np.zeros(rows, dtype=np.int64)
        attempt_ids = np.full(rows, f"base:{source_episode}", dtype=object)
        valid_chunk = frame_index <= rows - 50
        intervention = np.zeros(rows, dtype=np.bool_)
        source_values = np.zeros(rows, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported source_kind={source_kind!r}.")
    return {
        acp_field: pa.array(indicator, type=pa.int64()),
        apply_mask_field: pa.array(apply_mask, type=pa.int64()),
        replay_source_field: pa.array(source_values, type=pa.int64()),
        ATTEMPT_ID: pa.array(attempt_ids.tolist(), type=pa.string()),
        VALID_CHUNK: pa.array(valid_chunk, type=pa.bool_()),
        INTERVENTION: pa.array(intervention, type=pa.bool_()),
    }


def write_episode(
    *,
    source_root: Path,
    source_info: dict,
    source_episode: int,
    output_root: Path,
    output_episode: int,
    global_index: int,
    source_kind: str,
    acp_field: str,
    apply_mask_field: str,
    replay_source_field: str,
) -> tuple[dict, dict]:
    input_path = source_data_path(source_root, source_info, source_episode)
    required = [STATE, ACTION, "timestamp", "frame_index", "episode_index", "index", "task_index"]
    if source_kind == "attempt":
        required.extend([acp_field, apply_mask_field, ATTEMPT_ID, VALID_CHUNK, INTERVENTION])
    table = pq.read_table(input_path, columns=required)
    rows = table.num_rows
    if rows <= 0:
        raise ValueError(f"Empty episode parquet: {input_path}.")
    original_episode = scalar_numpy(table, "episode_index", np.int64)
    if np.any(original_episode != source_episode):
        raise ValueError(f"Episode index drift in {input_path}.")
    frame_index = scalar_numpy(table, "frame_index", np.int64)
    if not np.array_equal(frame_index, np.arange(rows, dtype=np.int64)):
        raise ValueError(f"Non-contiguous frame index in {input_path}.")
    timestamp = scalar_numpy(table, "timestamp", np.float32)
    if rows > 1 and np.any(np.diff(timestamp.astype(np.float64)) <= 0):
        raise ValueError(f"Non-monotonic timestamps in {input_path}.")

    output_columns = {
        STATE: fixed_float_list(vector_numpy(table, STATE, 14), 14),
        ACTION: fixed_float_list(vector_numpy(table, ACTION, 14), 14),
        "timestamp": pa.array(timestamp, type=pa.float32()),
        "frame_index": pa.array(frame_index, type=pa.int64()),
        "episode_index": pa.array(np.full(rows, output_episode, dtype=np.int64)),
        "index": pa.array(np.arange(global_index, global_index + rows, dtype=np.int64)),
        "task_index": pa.array(np.zeros(rows, dtype=np.int64)),
    }
    output_columns.update(
        policy_columns(
            table,
            source_kind=source_kind,
            source_episode=source_episode,
            acp_field=acp_field,
            apply_mask_field=apply_mask_field,
            replay_source_field=replay_source_field,
        )
    )
    output_table = pa.table(output_columns)
    output_path = output_root / "data" / "chunk-000" / f"file-{output_episode:03d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, output_path, compression="zstd", use_dictionary=True)

    key_map = (
        ATTEMPT_TO_BASE_VIDEO_KEY
        if source_kind == "attempt"
        else {key: key for key in BASE_VIDEO_KEYS}
    )
    video_links = add_video_links(
        output_root=output_root,
        output_episode=output_episode,
        source_root=source_root,
        source_info=source_info,
        source_episode=source_episode,
        key_map=key_map,
    )
    episode_meta = {
        "episode_index": output_episode,
        "tasks": [PROMPT],
        "length": rows,
        "data/chunk_index": 0,
        "data/file_index": output_episode,
        "dataset_from_index": global_index,
        "dataset_to_index": global_index + rows,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }
    for key, location in video_links.items():
        episode_meta[f"videos/{key}/chunk_index"] = location["chunk_index"]
        episode_meta[f"videos/{key}/file_index"] = location["file_index"]
        episode_meta[f"videos/{key}/from_timestamp"] = 0.0
        episode_meta[f"videos/{key}/to_timestamp"] = float(timestamp[-1])
    file_record = {
        "output_episode_index": output_episode,
        "source_kind": source_kind,
        "source_episode_index": source_episode,
        "rows": rows,
        "output_parquet_sha256": sha256_file(output_path),
    }
    return episode_meta, file_record


def parse_train_attempts(manifest: dict) -> list[int]:
    if "splits" in manifest:
        values = manifest["splits"]["train"]["episode_indices"]
    else:
        values = manifest["episode_indices"]
    attempts = [int(value) for value in values]
    if len(attempts) != 386 or len(set(attempts)) != 386:
        raise ValueError(f"Expected 386 unique train attempts, got {len(attempts)}.")
    return attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--acp-field", default="acp_indicator_attempt_v2")
    parser.add_argument("--apply-mask-field", default="acp_apply_mask")
    parser.add_argument("--replay-source-field", default="replay_source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker_path = args.output_root / "meta" / "MIXED_REPLAY_COMPLETE.json"
    if args.output_root.exists():
        if marker_path.is_file() and read_json(marker_path).get("status") == "complete":
            print("MIXED_REPLAY_ALREADY_COMPLETE", args.output_root)
            return
        raise FileExistsError(f"Refusing to overwrite incomplete view: {args.output_root}")

    base_info_path = args.base_root / "meta" / "info.json"
    attempt_info_path = args.attempt_root / "meta" / "info.json"
    base_info = read_json(base_info_path)
    attempt_info = read_json(attempt_info_path)
    attempts = parse_train_attempts(read_json(args.split_manifest))
    if int(base_info["total_episodes"]) != 558:
        raise ValueError("Base view must contain 558 episodes.")
    if int(base_info["fps"]) != 30 or int(attempt_info["fps"]) != 30:
        raise ValueError("Both sources must be 30 FPS.")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.building-", dir=args.output_root.parent)
    )
    episodes_meta = []
    file_records = []
    global_index = 0
    try:
        sources = [
            ("base", args.base_root, base_info, list(range(558))),
            ("attempt", args.attempt_root, attempt_info, attempts),
        ]
        for source_kind, source_root, source_info, episodes in sources:
            for offset, source_episode in enumerate(episodes):
                output_episode = len(episodes_meta)
                metadata, record = write_episode(
                    source_root=source_root,
                    source_info=source_info,
                    source_episode=source_episode,
                    output_root=temp_root,
                    output_episode=output_episode,
                    global_index=global_index,
                    source_kind=source_kind,
                    acp_field=args.acp_field,
                    apply_mask_field=args.apply_mask_field,
                    replay_source_field=args.replay_source_field,
                )
                episodes_meta.append(metadata)
                file_records.append(record)
                global_index += int(metadata["length"])
                if (offset + 1) % 50 == 0 or offset + 1 == len(episodes):
                    print(f"{source_kind.upper()}_PROGRESS {offset + 1}/{len(episodes)}", flush=True)

        episodes_path = temp_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(episodes_meta), episodes_path, compression="zstd")
        shutil.copy2(args.base_root / "meta" / "tasks.parquet", temp_root / "meta" / "tasks.parquet")
        shutil.copy2(args.base_root / "meta" / "stats.json", temp_root / "meta" / "stats.json")

        base_frames = int(base_info["total_frames"])
        attempt_frames = sum(record["rows"] for record in file_records if record["source_kind"] == "attempt")
        info = copy.deepcopy(base_info)
        info["total_episodes"] = len(episodes_meta)
        info["total_frames"] = global_index
        info["duration"] = global_index / 30.0
        info["splits"] = {"train": f"0:{len(episodes_meta)}"}
        feature_specs = {
            args.acp_field: {"dtype": "int64", "shape": [1], "names": None},
            args.apply_mask_field: {"dtype": "int64", "shape": [1], "names": None},
            args.replay_source_field: {"dtype": "int64", "shape": [1], "names": None},
            ATTEMPT_ID: {"dtype": "string", "shape": [1], "names": None},
            VALID_CHUNK: {"dtype": "bool", "shape": [1], "names": None},
            INTERVENTION: {"dtype": "bool", "shape": [1], "names": None},
        }
        info["features"].update(feature_specs)
        info["topic_list"] = list(info.get("topic_list", info["features"]))
        for field in feature_specs:
            if field not in info["topic_list"]:
                info["topic_list"].append(field)
        info["data_files_size_in_mb"] = round(
            sum(path.stat().st_size for path in (temp_root / "data").rglob("*.parquet")) / 1e6, 6
        )
        info["video_files_size_in_mb"] = round(
            sum(path.resolve().stat().st_size for path in (temp_root / "videos").rglob("*.mp4")) / 1e6,
            6,
        )
        info["replay_contract"] = {
            "schema": "attempt-aware-policy-replay/v2",
            "base_episodes": 558,
            "attempt_train_episodes": 386,
            "base_frames": base_frames,
            "attempt_train_frames": attempt_frames,
            "acp_indicator_field": args.acp_field,
            "acp_apply_mask_field": args.apply_mask_field,
            "replay_source_field": args.replay_source_field,
            "attempt_id_field": ATTEMPT_ID,
            "valid_chunk_field": VALID_CHUNK,
            "normalization_stats": "copied byte-for-byte from full558 baseline view",
            "video_payloads": "absolute symlinks to immutable SSD sources",
        }
        write_json(temp_root / "meta" / "info.json", info)

        manifest = {
            "schema": "attempt-aware-policy-replay-manifest/v2",
            "base_root": str(args.base_root),
            "attempt_root": str(args.attempt_root),
            "output_root": str(args.output_root),
            "base_info_sha256": sha256_file(base_info_path),
            "attempt_info_sha256": sha256_file(attempt_info_path),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "base_episodes": 558,
            "attempt_episode_indices": attempts,
            "attempt_train_episodes": 386,
            "total_episodes": len(episodes_meta),
            "base_frames": base_frames,
            "attempt_train_frames": attempt_frames,
            "total_frames": global_index,
            "natural_attempt_fraction": attempt_frames / global_index,
            "files": file_records,
        }
        write_json(temp_root / "meta" / "replay_manifest.json", manifest)
        marker = {
            "status": "complete",
            "total_episodes": manifest["total_episodes"],
            "total_frames": manifest["total_frames"],
            "replay_manifest_sha256": sha256_file(temp_root / "meta" / "replay_manifest.json"),
        }
        write_json(temp_root / "meta" / "MIXED_REPLAY_COMPLETE.json", marker)
        os.replace(temp_root, args.output_root)
        write_json(args.manifest_out, manifest)
        print(
            "ATTEMPT_MIXED_REPLAY_PASS",
            json.dumps(
                {
                    "episodes": len(episodes_meta),
                    "frames": global_index,
                    "attempt_fraction": manifest["natural_attempt_fraction"],
                },
                sort_keys=True,
            ),
        )
    except BaseException:
        if temp_root.exists() and temp_root.parent == args.output_root.parent:
            shutil.rmtree(temp_root)
        raise


if __name__ == "__main__":
    main()
