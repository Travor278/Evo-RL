#!/usr/bin/env python3
"""Hard gate for the derived attempt-level Value training view."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path(os.environ["ATTEMPT_SNAPSHOT"]).resolve()
VIEW = Path(os.environ["ATTEMPT_VALUE_VIEW"]).resolve()
OUT = Path(os.environ["ATTEMPT_VALUE_AUDIT_OUTPUT"])
HDD_LINK = Path(os.environ["ATTEMPT_VALUE_VIEW_LINK"])

REQUIRED = {
    "source_episode_success",
    "logical_attempt_success",
    "logical_attempt_outcome_known",
    "logical_attempt_terminal_reason",
    "logical_attempt_intervened",
    "logical_attempt_autonomous_success",
    "logical_attempt_failure_type",
    "logical_attempt_outcome",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    errors: list[str] = []
    data_files = sorted((VIEW / "data").glob("chunk-*/file-*.parquet"))
    episode_files = sorted((VIEW / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if len(data_files) != 482:
        errors.append(f"data files={len(data_files)} expected=482")
    if len(episode_files) != 482:
        errors.append(f"episode files={len(episode_files)} expected=482")

    frame_count = 0
    success = Counter()
    outcomes = Counter()
    split_outcomes: dict[str, Counter] = {}
    different_data_inodes = 0
    for episode_index, data_path in enumerate(data_files):
        table = pq.read_table(data_path)
        frame_count += table.num_rows
        missing = REQUIRED - set(table.column_names)
        if missing:
            errors.append(f"{data_path.name} missing={sorted(missing)}")
            continue
        indices = set(table["episode_index"].to_pylist())
        if indices != {episode_index}:
            errors.append(f"{data_path.name} episode_index={indices}")
        for field in REQUIRED:
            if len(set(table[field].to_pylist())) != 1:
                errors.append(f"{data_path.name} non-constant {field}")
        logical_success = bool(table["logical_attempt_success"][0].as_py())
        known = bool(table["logical_attempt_outcome_known"][0].as_py())
        outcome = str(table["logical_attempt_outcome"][0].as_py())
        split = str(table["dataset_split"][0].as_py())
        success["known" if known else "unknown"] += 1
        success["success" if logical_success else "failure"] += 1
        outcomes[outcome] += 1
        split_outcomes.setdefault(split, Counter())["success" if logical_success else "failure"] += 1
        if data_path.stat().st_ino != (SOURCE / data_path.relative_to(VIEW)).stat().st_ino:
            different_data_inodes += 1

        chunk_valid = table["logical_action_chunk_valid_50"].to_pylist()
        attempt_indices = table["logical_attempt_frame_index"].to_pylist()
        attempt_ids = table["logical_attempt_id"].to_pylist()
        if len(set(attempt_ids)) != 1:
            errors.append(f"{data_path.name} contains multiple logical_attempt_id values")
        length = table.num_rows
        for offset, is_valid in enumerate(chunk_valid):
            if is_valid and (offset + 49 >= length or attempt_indices[offset + 49] != attempt_indices[offset] + 49):
                errors.append(f"{data_path.name} invalid chunk mask at row={offset}")
                break

        ep_table = pq.read_table(episode_files[episode_index])
        if ep_table.num_rows != 1:
            errors.append(f"{episode_files[episode_index].name} rows={ep_table.num_rows}")
        elif bool(ep_table["episode_success"][0].as_py()) != logical_success:
            errors.append(f"{episode_files[episode_index].name} episode_success mismatch")

    for split in ("train", "validation", "test"):
        counts = split_outcomes.get(split, Counter())
        if counts["success"] == 0 or counts["failure"] == 0:
            errors.append(f"split {split} lacks success/failure: {dict(counts)}")
    if frame_count != 384701:
        errors.append(f"frames={frame_count} expected=384701")
    if success != Counter({"known": 482, "success": 475, "failure": 7}):
        errors.append(f"outcome totals mismatch: {dict(success)}")
    if different_data_inodes != 482:
        errors.append(f"modified data inode separation={different_data_inodes} expected=482")
    if not HDD_LINK.is_symlink() or HDD_LINK.resolve() != VIEW:
        errors.append(f"HDD link mismatch: {HDD_LINK}")

    source_info = SOURCE / "meta" / "info.json"
    view_info = VIEW / "meta" / "info.json"
    info = json.loads(view_info.read_text(encoding="utf-8"))
    missing_info = REQUIRED - set(info["features"])
    if missing_info:
        errors.append(f"info.json missing features={sorted(missing_info)}")

    result = {
        "status": "pass" if not errors else "fail",
        "source": str(SOURCE),
        "view": str(VIEW),
        "data_files": len(data_files),
        "episode_files": len(episode_files),
        "frames": frame_count,
        "attempts": sum(outcomes.values()),
        "success": success["success"],
        "failure": success["failure"],
        "unknown": success["unknown"],
        "outcomes": dict(sorted(outcomes.items())),
        "split_outcomes": {key: dict(value) for key, value in sorted(split_outcomes.items())},
        "modified_data_inode_separation": different_data_inodes,
        "source_info_sha256": digest(source_info),
        "view_info_sha256": digest(view_info),
        "hdd_link_target": str(HDD_LINK.resolve()) if HDD_LINK.exists() else None,
        "errors": errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
