#!/usr/bin/env python3
"""Create a null-free minimal dataset view for official Pi*0.6 value training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


BASE_PARQUET_FIELDS = sorted(
    [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
)
VIDEO_FIELDS = [
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-derived", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-intervention",
        action="store_true",
        help="Keep the non-null intervention flag needed by ACP inference and policy training.",
    )
    args = parser.parse_args()
    parquet_fields = list(BASE_PARQUET_FIELDS)
    if args.include_intervention:
        parquet_fields.append("complementary_info.is_intervention")
        parquet_fields.sort()
    source = args.source_derived.resolve()
    destination = args.destination.resolve()
    if not (source / "DERIVED_VIEW_PROVENANCE.json").is_file():
        raise RuntimeError("Source is not a recorded derived view")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite value view: {destination}")
    destination.mkdir(parents=True)
    for name in (".gitattributes", "README.md"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "meta", destination / "meta", copy_function=shutil.copy2)
    os.symlink(source / "videos", destination / "videos", target_is_directory=True)
    (destination / "data/chunk-000").mkdir(parents=True)

    records = []
    for input_path in sorted((source / "data/chunk-000").glob("*.parquet")):
        table = pq.read_table(input_path)
        missing = [field for field in parquet_fields if field not in table.column_names]
        if missing:
            raise RuntimeError(f"Missing required fields in {input_path}: {missing}")
        minimal = table.select(parquet_fields)
        if any(minimal[field].null_count for field in minimal.column_names):
            raise RuntimeError(f"Null found in required value-training field: {input_path}")
        output_path = destination / "data/chunk-000" / input_path.name
        pq.write_table(minimal, output_path, compression="zstd")
        records.append(
            {
                "path": str(output_path.relative_to(destination)),
                "rows": minimal.num_rows,
                "columns": minimal.column_names,
                "sha256": sha256_file(output_path),
                "size": output_path.stat().st_size,
            }
        )

    info_path = destination / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_features = info["features"]
    keep_features = parquet_fields + VIDEO_FIELDS
    missing_features = [field for field in keep_features if field not in source_features]
    if missing_features:
        raise RuntimeError(f"Required fields missing from info.json: {missing_features}")
    info["features"] = {field: source_features[field] for field in sorted(keep_features)}
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_derived": str(source),
        "destination": str(destination),
        "purpose": (
            "Null-free official Pi*0.6 ACP inference/policy view; full HIL evidence remains in source derived view."
            if args.include_intervention
            else "Null-free minimal official Pi*0.6 value-training view; full HIL evidence remains in source derived view."
        ),
        "parquet_fields": parquet_fields,
        "video_fields": VIDEO_FIELDS,
        "files": records,
        "info_sha256": sha256_file(info_path),
        "stats_sha256": sha256_file(destination / "meta/stats.json"),
        "tasks_sha256": sha256_file(destination / "meta/tasks.parquet"),
    }
    (destination / "VALUE_VIEW_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"VALUE_MINIMAL_VIEW_OK files={len(records)} rows={sum(item['rows'] for item in records)} "
        f"columns={len(parquet_fields)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
