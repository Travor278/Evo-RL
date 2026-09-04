#!/usr/bin/env python3
"""Build a deterministic LeRobot v3 replay view for paired ACP/BC training.

The view contains every base episode followed by the HIL training split. Video
payloads are symlinked to their immutable SSD sources; only compact parquet and
metadata files are materialized in the experiment workspace.
"""

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
BASE_VIDEO_KEYS = (
    "observation.images.camera_top",
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
)
HIL_TO_BASE_VIDEO_KEY = {
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


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def fixed_float_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"Expected [N,{width}] float array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite state/action value found")
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), width)


def scalar_numpy(table: pa.Table, key: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=dtype).reshape(-1)


def vector_numpy(table: pa.Table, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
    if values.shape != (table.num_rows, width):
        raise ValueError(f"{key} expected {(table.num_rows, width)}, got {values.shape}")
    return values


def source_data_path(root: Path, info: dict, episode_index: int) -> Path:
    return root / info["data_path"].format(chunk_index=0, file_index=episode_index)


def source_video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
    return root / info["video_path"].format(
        video_key=video_key, chunk_index=0, file_index=episode_index
    )


def add_video_links(
    tmp_root: Path,
    output_episode_index: int,
    source_root: Path,
    source_info: dict,
    source_episode_index: int,
    source_to_output_keys: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    metadata: dict[str, dict[str, float | int]] = {}
    for source_key, output_key in source_to_output_keys.items():
        source = source_video_path(source_root, source_info, source_key, source_episode_index)
        resolved = source.resolve(strict=True)
        destination = (
            tmp_root
            / "videos"
            / output_key
            / "chunk-000"
            / f"file-{output_episode_index:03d}.mp4"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(resolved, destination)
        metadata[output_key] = {
            "chunk_index": 0,
            "file_index": output_episode_index,
        }
    return metadata


def write_episode(
    *,
    source_root: Path,
    source_info: dict,
    source_episode_index: int,
    output_root: Path,
    output_episode_index: int,
    global_from_index: int,
    source_kind: str,
    acp_field: str,
    apply_mask_field: str,
    source_field: str,
) -> tuple[dict, dict]:
    input_path = source_data_path(source_root, source_info, source_episode_index)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    required = [STATE, ACTION, "timestamp", "frame_index", "episode_index", "index", "task_index"]
    if source_kind == "hil":
        required.append(acp_field)
    table = pq.read_table(input_path, columns=required)
    rows = table.num_rows
    if rows <= 0:
        raise ValueError(f"Empty episode parquet: {input_path}")

    original_episode = scalar_numpy(table, "episode_index", np.int64)
    if np.any(original_episode != source_episode_index):
        raise ValueError(f"Episode index drift in {input_path}")
    frame_index = scalar_numpy(table, "frame_index", np.int64)
    if not np.array_equal(frame_index, np.arange(rows, dtype=np.int64)):
        raise ValueError(f"Non-contiguous frame index in {input_path}")
    timestamp = scalar_numpy(table, "timestamp", np.float32)
    if np.any(np.diff(timestamp.astype(np.float64)) <= 0):
        raise ValueError(f"Non-monotonic timestamp in {input_path}")

    state = vector_numpy(table, STATE, 14)
    action = vector_numpy(table, ACTION, 14)
    if source_kind == "hil":
        indicator = scalar_numpy(table, acp_field, np.int64)
        if not np.isin(indicator, [0, 1]).all():
            raise ValueError(f"Non-binary ACP indicator in {input_path}")
        apply_mask = np.ones(rows, dtype=np.int64)
        source_value = np.ones(rows, dtype=np.int64)
        video_key_map = HIL_TO_BASE_VIDEO_KEY
    else:
        indicator = np.zeros(rows, dtype=np.int64)
        apply_mask = np.zeros(rows, dtype=np.int64)
        source_value = np.zeros(rows, dtype=np.int64)
        video_key_map = {key: key for key in BASE_VIDEO_KEYS}

    output_table = pa.table(
        {
            STATE: fixed_float_list(state, 14),
            ACTION: fixed_float_list(action, 14),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(
                np.full(rows, output_episode_index, dtype=np.int64), type=pa.int64()
            ),
            "index": pa.array(
                np.arange(global_from_index, global_from_index + rows, dtype=np.int64),
                type=pa.int64(),
            ),
            "task_index": pa.array(np.zeros(rows, dtype=np.int64), type=pa.int64()),
            acp_field: pa.array(indicator, type=pa.int64()),
            apply_mask_field: pa.array(apply_mask, type=pa.int64()),
            source_field: pa.array(source_value, type=pa.int64()),
        }
    )
    output_path = output_root / "data" / "chunk-000" / f"file-{output_episode_index:03d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table, output_path, compression="zstd", use_dictionary=True)

    video_links = add_video_links(
        output_root,
        output_episode_index,
        source_root,
        source_info,
        source_episode_index,
        video_key_map,
    )
    to_timestamp = float(timestamp[-1])
    episode_meta: dict[str, object] = {
        "episode_index": output_episode_index,
        "tasks": [PROMPT],
        "length": rows,
        "data/chunk_index": 0,
        "data/file_index": output_episode_index,
        "dataset_from_index": global_from_index,
        "dataset_to_index": global_from_index + rows,
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index": 0,
    }
    for key, location in video_links.items():
        episode_meta[f"videos/{key}/chunk_index"] = location["chunk_index"]
        episode_meta[f"videos/{key}/file_index"] = location["file_index"]
        episode_meta[f"videos/{key}/from_timestamp"] = 0.0
        episode_meta[f"videos/{key}/to_timestamp"] = to_timestamp

    file_record = {
        "output_episode_index": output_episode_index,
        "source_kind": source_kind,
        "source_episode_index": source_episode_index,
        "rows": rows,
        "source_parquet": str(input_path),
        "output_parquet": str(output_path),
        "output_parquet_sha256": sha256_file(output_path),
    }
    return episode_meta, file_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--hil-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--acp-field", default="complementary_info.acp_indicator_evorl_official_v1"
    )
    parser.add_argument(
        "--apply-mask-field", default="complementary_info.acp_apply_mask_evorl_official_v1"
    )
    parser.add_argument(
        "--source-field", default="complementary_info.replay_source_evorl_official_v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    marker_path = args.output_root / "meta" / "MIXED_REPLAY_COMPLETE.json"
    if args.output_root.exists():
        if marker_path.is_file():
            marker = read_json(marker_path)
            if marker.get("status") == "complete":
                print("MIXED_REPLAY_ALREADY_COMPLETE", args.output_root)
                return
        raise FileExistsError(f"Refusing to overwrite incomplete view: {args.output_root}")

    base_info_path = args.base_root / "meta" / "info.json"
    hil_info_path = args.hil_root / "meta" / "info.json"
    base_info = read_json(base_info_path)
    hil_info = read_json(hil_info_path)
    split = read_json(args.train_manifest)
    hil_episodes = [int(value) for value in split["episode_indices"]]
    if len(hil_episodes) != 60 or len(set(hil_episodes)) != len(hil_episodes):
        raise ValueError(f"Expected 60 unique HIL train episodes, got {len(hil_episodes)}")
    base_count = int(base_info["total_episodes"])
    if base_count != 558:
        raise ValueError(f"Expected 558 base episodes, got {base_count}")
    if int(base_info["fps"]) != 30 or int(hil_info["fps"]) != 30:
        raise ValueError("Both sources must be 30 FPS")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.building-", dir=args.output_root.parent)
    )
    episodes_meta: list[dict] = []
    file_records: list[dict] = []
    global_index = 0
    try:
        for base_ep in range(base_count):
            meta, record = write_episode(
                source_root=args.base_root,
                source_info=base_info,
                source_episode_index=base_ep,
                output_root=temp_root,
                output_episode_index=base_ep,
                global_from_index=global_index,
                source_kind="base",
                acp_field=args.acp_field,
                apply_mask_field=args.apply_mask_field,
                source_field=args.source_field,
            )
            episodes_meta.append(meta)
            file_records.append(record)
            global_index += int(meta["length"])
            if (base_ep + 1) % 100 == 0:
                print(f"BASE_PROGRESS {base_ep + 1}/{base_count}", flush=True)

        for offset, hil_ep in enumerate(hil_episodes):
            output_ep = base_count + offset
            meta, record = write_episode(
                source_root=args.hil_root,
                source_info=hil_info,
                source_episode_index=hil_ep,
                output_root=temp_root,
                output_episode_index=output_ep,
                global_from_index=global_index,
                source_kind="hil",
                acp_field=args.acp_field,
                apply_mask_field=args.apply_mask_field,
                source_field=args.source_field,
            )
            episodes_meta.append(meta)
            file_records.append(record)
            global_index += int(meta["length"])
            print(f"HIL_PROGRESS {offset + 1}/{len(hil_episodes)}", flush=True)

        expected_frames = int(base_info["total_frames"]) + sum(
            record["rows"] for record in file_records if record["source_kind"] == "hil"
        )
        if global_index != expected_frames:
            raise ValueError(f"Frame count mismatch: {global_index} != {expected_frames}")

        episodes_table = pa.Table.from_pylist(episodes_meta)
        episodes_path = temp_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episodes_table, episodes_path, compression="zstd", use_dictionary=True)

        shutil.copy2(args.base_root / "meta" / "tasks.parquet", temp_root / "meta" / "tasks.parquet")
        shutil.copy2(args.base_root / "meta" / "stats.json", temp_root / "meta" / "stats.json")

        output_info = copy.deepcopy(base_info)
        output_info["total_episodes"] = base_count + len(hil_episodes)
        output_info["total_frames"] = global_index
        output_info["duration"] = global_index / 30.0
        output_info["splits"] = {"train": f"0:{output_info['total_episodes']}"}
        for field in (args.acp_field, args.apply_mask_field, args.source_field):
            output_info["features"][field] = {"dtype": "int64", "shape": [1], "names": None}
        output_info["topic_list"] = list(output_info.get("topic_list", output_info["features"]))
        for field in (args.acp_field, args.apply_mask_field, args.source_field):
            if field not in output_info["topic_list"]:
                output_info["topic_list"].append(field)
        output_info["data_files_size_in_mb"] = round(
            sum(path.stat().st_size for path in (temp_root / "data").rglob("*.parquet")) / 1e6, 6
        )
        output_info["video_files_size_in_mb"] = round(
            sum(path.resolve().stat().st_size for path in (temp_root / "videos").rglob("*.mp4")) / 1e6,
            6,
        )
        output_info["replay_contract"] = {
            "schema": "evorl_piperx_mixed_replay/v1",
            "base_episodes": base_count,
            "hil_train_episodes": len(hil_episodes),
            "base_frames": int(base_info["total_frames"]),
            "hil_train_frames": global_index - int(base_info["total_frames"]),
            "acp_field": args.acp_field,
            "acp_apply_mask_field": args.apply_mask_field,
            "source_field": args.source_field,
            "normalization_stats": "copied byte-for-byte from full558 base training view",
            "video_payloads": "absolute symlinks to immutable SSD source files",
        }
        write_json(temp_root / "meta" / "info.json", output_info)

        manifest = {
            "schema": "evorl_piperx_mixed_replay_manifest/v1",
            "base_root": str(args.base_root),
            "hil_root": str(args.hil_root),
            "output_root": str(args.output_root),
            "base_info_sha256": sha256_file(base_info_path),
            "hil_info_sha256": sha256_file(hil_info_path),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "base_episodes": base_count,
            "hil_train_episode_indices": hil_episodes,
            "hil_train_episodes": len(hil_episodes),
            "total_episodes": base_count + len(hil_episodes),
            "base_frames": int(base_info["total_frames"]),
            "hil_train_frames": global_index - int(base_info["total_frames"]),
            "total_frames": global_index,
            "natural_hil_fraction": (global_index - int(base_info["total_frames"])) / global_index,
            "fields": {
                "acp_indicator": args.acp_field,
                "acp_apply_mask": args.apply_mask_field,
                "replay_source": args.source_field,
            },
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
            "MIXED_REPLAY_PASS",
            json.dumps(
                {
                    "episodes": manifest["total_episodes"],
                    "frames": manifest["total_frames"],
                    "hil_fraction": manifest["natural_hil_fraction"],
                    "view": str(args.output_root),
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
