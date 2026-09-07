#!/usr/bin/env python3
"""Atomically remove nullable policy diagnostics from the dedicated Value view."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

VIEW = Path(os.environ["ATTEMPT_VALUE_VIEW"]).resolve()
OUTPUT = Path(os.environ["VALUE_MINIMAL_AUDIT"])

PARQUET_FIELDS = sorted(
    [
        "action",
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "complementary_info.is_intervention",
        "source_collection_id",
        "source_episode_uid",
        "physical_episode_uid",
        "dataset_split",
        "logical_attempt_index",
        "logical_attempt_id",
        "logical_attempt_frame_index",
        "logical_attempt_timestamp",
        "logical_attempt_terminal",
        "logical_transition_valid",
        "logical_action_chunk_valid_50",
        "source_episode_success",
        "logical_attempt_success",
        "logical_attempt_outcome_known",
        "logical_attempt_terminal_reason",
        "logical_attempt_intervened",
        "logical_attempt_autonomous_success",
        "logical_attempt_failure_type",
        "logical_attempt_outcome",
    ]
)
VIDEO_FIELDS = sorted(
    [
        "observation.images.top",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if not (VIEW / "build_artifacts" / "attempt_outcomes_v2" / "attempt_outcomes_v2.json").is_file():
        raise RuntimeError("Refusing to modify a view without attempt outcome provenance")
    data_files = sorted((VIEW / "data").glob("chunk-*/file-*.parquet"))
    if len(data_files) != 482:
        raise ValueError(f"Expected 482 data files, found {len(data_files)}")
    records = []
    for path in data_files:
        table = pq.read_table(path)
        missing = sorted(set(PARQUET_FIELDS) - set(table.column_names))
        if missing:
            raise ValueError(f"{path.name} missing required fields: {missing}")
        minimal = table.select(PARQUET_FIELDS)
        nulls = {field: minimal[field].null_count for field in minimal.column_names if minimal[field].null_count}
        if nulls:
            raise ValueError(f"{path.name} has null required fields: {nulls}")
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            pq.write_table(minimal, temporary, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        records.append(
            {
                "path": path.relative_to(VIEW).as_posix(),
                "rows": minimal.num_rows,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    info_path = VIEW / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_features = info["features"]
    keep = sorted(PARQUET_FIELDS + VIDEO_FIELDS)
    missing_features = sorted(set(keep) - set(source_features))
    if missing_features:
        raise ValueError(f"info.json missing required features: {missing_features}")
    info["features"] = {field: source_features[field] for field in keep}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=info_path.parent, delete=False) as stream:
        json.dump(info, stream, indent=2)
        stream.write("\n")
        temporary_info = Path(stream.name)
    temporary_info.replace(info_path)

    result = {
        "status": "pass",
        "view": str(VIEW),
        "data_files": len(records),
        "frames": sum(item["rows"] for item in records),
        "parquet_columns": PARQUET_FIELDS,
        "video_fields": VIDEO_FIELDS,
        "null_count": 0,
        "info_sha256": sha256(info_path),
        "files": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
