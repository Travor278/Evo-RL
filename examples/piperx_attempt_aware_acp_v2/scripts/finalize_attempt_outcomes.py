#!/usr/bin/env python3
"""Adjudicate attempt outcomes and build a non-destructive attempt-level Value view."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE = Path(os.environ["ATTEMPT_SNAPSHOT"]).resolve()
AUDIT_CSV = Path(os.environ["ATTEMPT_AUDIT_CSV"]).resolve()
DESTINATION = Path(os.environ["ATTEMPT_VALUE_VIEW"]).resolve()
REPORT_DIR = Path(os.environ["OUTCOME_REPORT_DIR"]).resolve()
HDD_LINK = Path(os.environ["ATTEMPT_VALUE_VIEW_LINK"])

FAILED_SUCCESS_EPISODES = {450, 451, 452, 453, 455, 456, 457, 458, 470, 473}
FAILED_FAILURES = {
    454: ("operator_abort", "operator_abort"),
    459: ("autonomous_failure", "task_failure"),
    471: ("autonomous_failure", "task_failure"),
    472: ("autonomous_failure", "task_failure"),
    474: ("autonomous_failure", "task_failure"),
    475: ("autonomous_failure", "task_failure"),
    481: ("timeout_failure", "timeout"),
}

OUTCOME_ENUM = {
    "autonomous_success",
    "intervention_recovered_success",
    "autonomous_failure",
    "timeout_failure",
    "safety_abort",
    "operator_abort",
    "unknown",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_parquet(path: Path, table: pa.Table) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def set_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def label_record(row: dict[str, str]) -> dict:
    record: dict = dict(row)
    episode_index = int(row["attempt_episode_index"])
    source_success = parse_bool(row["source_episode_success"])
    intervened = parse_bool(row["logical_attempt_intervened_evidence"])
    review_group = row["review_group"]

    if review_group in {"new_declared_complete_pool", "new_required_spotcheck", "old_source_success_review"} or review_group == "failed_source_full_review" and episode_index in FAILED_SUCCESS_EPISODES:
        success = True
        failure_type = "none"
        terminal_reason = "task_success"
    elif review_group == "failed_source_full_review" and episode_index in FAILED_FAILURES:
        success = False
        failure_type, terminal_reason = FAILED_FAILURES[episode_index]
    else:
        raise ValueError(f"No adjudication rule for episode={episode_index} group={review_group!r}")

    if success:
        outcome = "intervention_recovered_success" if intervened else "autonomous_success"
    else:
        outcome = failure_type
    if outcome not in OUTCOME_ENUM:
        raise ValueError(f"Invalid outcome enum: {outcome}")

    if review_group == "failed_source_full_review":
        review_status = "full_video_evidence_reviewed"
        evidence = f"three-camera contact sheet attempt-{episode_index:03d}.jpg; four temporal samples per camera"
    elif review_group == "old_source_success_review":
        review_status = "full_video_evidence_reviewed"
        evidence = f"source success plus three-camera contact sheet attempt-{episode_index:03d}.jpg"
    elif review_group == "new_required_spotcheck":
        review_status = "stratified_video_spotcheck_reviewed"
        evidence = f"declared 5x complete plus three-camera contact sheet attempt-{episode_index:03d}.jpg"
    else:
        review_status = "collection_claim_accepted_after_stratified_spotcheck"
        evidence = "declared 5x complete; one physical episode/collection (50 attempts total) visually passed"

    record.update(
        {
            "attempt_episode_index": episode_index,
            "logical_attempt_index": int(row["logical_attempt_index"]),
            "parent_physical_episode_index": int(row["parent_physical_episode_index"]),
            "attempt_length_frames": int(row["attempt_length_frames"]),
            "source_episode_success": source_success,
            "logical_attempt_intervened": intervened,
            "logical_attempt_outcome_known": True,
            "logical_attempt_success": success,
            "logical_attempt_outcome": outcome,
            "logical_attempt_terminal_reason": terminal_reason,
            "logical_attempt_failure_type": failure_type,
            "logical_attempt_autonomous_success": success and not intervened,
            "review_status": review_status,
            "review_evidence": evidence,
            "physical_attempt_outcome_conflict": source_success != success,
        }
    )
    return record


def load_records() -> list[dict]:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    records = [label_record(row) for row in rows]
    if len(records) != 482:
        raise ValueError(f"Expected 482 attempts, found {len(records)}")
    episode_indices = [record["attempt_episode_index"] for record in records]
    if sorted(episode_indices) != list(range(482)):
        raise ValueError("Attempt episode indices must be exactly 0..481")
    return records


def audit_splits(records: list[dict]) -> dict:
    physical_split: dict[str, str] = {}
    collection_split: dict[str, str] = {}
    split_counts: dict[str, Counter] = defaultdict(Counter)
    split_attempt_index: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        split = record["dataset_split"]
        physical = record["physical_episode_uid"]
        collection = record["source_collection_id"]
        if physical in physical_split and physical_split[physical] != split:
            raise ValueError(f"physical leakage: {physical}")
        if collection in collection_split and collection_split[collection] != split:
            raise ValueError(f"collection leakage: {collection}")
        physical_split[physical] = split
        collection_split[collection] = split
        split_counts[split]["attempts"] += 1
        split_counts[split]["success" if record["logical_attempt_success"] else "failure"] += 1
        split_counts[split]["known"] += int(record["logical_attempt_outcome_known"])
        split_counts[split]["intervened" if record["logical_attempt_intervened"] else "autonomous"] += 1
        split_counts[split]["new" if parse_bool(str(record["is_new_collection"])) else "old"] += 1
        split_attempt_index[split][str(record["logical_attempt_index"])] += 1

    for split in ("train", "validation", "test"):
        counts = split_counts[split]
        if counts["success"] == 0 or counts["failure"] == 0:
            raise ValueError(f"split {split} lacks success/failure coverage: {dict(counts)}")
        if counts["autonomous"] == 0 or counts["intervened"] == 0:
            raise ValueError(f"split {split} lacks autonomous/intervention coverage: {dict(counts)}")
    return {
        "counts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
        "attempt_index": {split: dict(counts) for split, counts in sorted(split_attempt_index.items())},
        "physical_episode_leakage": 0,
        "collection_leakage": 0,
    }


def update_info(root: Path) -> None:
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    specs = {
        "source_episode_success": "bool",
        "logical_attempt_success": "bool",
        "logical_attempt_outcome_known": "bool",
        "logical_attempt_terminal_reason": "string",
        "logical_attempt_intervened": "bool",
        "logical_attempt_autonomous_success": "bool",
        "logical_attempt_failure_type": "string",
        "logical_attempt_outcome": "string",
    }
    for name, dtype in specs.items():
        info["features"][name] = {"dtype": dtype, "shape": [1], "names": None}
    atomic_json(path, info)


def normalize_data_column_order(root: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    feature_order = list(info["features"])
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        desired = [name for name in feature_order if name in table.column_names]
        if set(desired) != set(table.column_names):
            raise ValueError(f"{path} contains columns absent from info.json")
        atomic_parquet(path, table.select(desired))


def annotate_parquets(root: Path, records: list[dict]) -> None:
    fields = (
        ("source_episode_success", pa.bool_()),
        ("logical_attempt_success", pa.bool_()),
        ("logical_attempt_outcome_known", pa.bool_()),
        ("logical_attempt_terminal_reason", pa.string()),
        ("logical_attempt_intervened", pa.bool_()),
        ("logical_attempt_autonomous_success", pa.bool_()),
        ("logical_attempt_failure_type", pa.string()),
        ("logical_attempt_outcome", pa.string()),
    )
    for record in records:
        episode_index = record["attempt_episode_index"]
        chunk = episode_index // 1000
        relative_paths = (
            Path("data") / f"chunk-{chunk:03d}" / f"file-{episode_index:03d}.parquet",
            Path("meta") / "episodes" / f"chunk-{chunk:03d}" / f"file-{episode_index:03d}.parquet",
        )
        for relative in relative_paths:
            path = root / relative
            table = pq.read_table(path)
            row_count = table.num_rows
            if relative.parts[0] == "data":
                episode_values = set(table["episode_index"].to_pylist())
                if episode_values != {episode_index}:
                    raise ValueError(f"Episode index mismatch in {relative}: {episode_values}")
            for field, field_type in fields:
                table = set_column(table, field, pa.array([record[field]] * row_count, type=field_type))
            if relative.parts[0] == "meta":
                table = set_column(
                    table,
                    "episode_success",
                    pa.array([record["logical_attempt_success"]] * row_count, type=pa.bool_()),
                )
            atomic_parquet(path, table)


def build_parquet_manifest(root: Path) -> list[dict]:
    manifest = []
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    paths += sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for path in paths:
        manifest.append(
            {
                "path": path.relative_to(root).as_posix(),
                "rows": pq.read_metadata(path).num_rows,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return manifest


def main() -> None:
    if SOURCE == DESTINATION or SOURCE in DESTINATION.parents:
        raise ValueError("Destination must be a sibling view, never the source snapshot or its child")
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {DESTINATION}")
    if HDD_LINK.exists() or HDD_LINK.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing HDD link: {HDD_LINK}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    split_audit = audit_splits(records)
    staging = DESTINATION.with_name(f".{DESTINATION.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    shutil.copytree(SOURCE, staging, copy_function=os.link)
    try:
        annotate_parquets(staging, records)
        update_info(staging)
        normalize_data_column_order(staging)
        parquet_manifest = build_parquet_manifest(staging)
        artifact_dir = staging / "build_artifacts" / "attempt_outcomes_v2"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(artifact_dir / "attempt_outcomes_v2.json", records)
        atomic_json(artifact_dir / "split_outcome_audit.json", split_audit)
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(DESTINATION)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    HDD_LINK.parent.mkdir(parents=True, exist_ok=True)
    HDD_LINK.symlink_to(DESTINATION, target_is_directory=True)

    outcome_counts = Counter(record["logical_attempt_outcome"] for record in records)
    success_count = sum(record["logical_attempt_success"] for record in records)
    conflict_count = sum(record["physical_attempt_outcome_conflict"] for record in records)
    reviewed_count = sum("reviewed" in record["review_status"] for record in records)
    summary = {
        "schema_version": "attempt-aware-outcome-contract/v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": str(SOURCE),
        "source_revision": "d8c2c0f0115ce3456dbfa38f5529276c0e4bbe16",
        "derived_view": str(DESTINATION),
        "attempts": len(records),
        "known_success": int(success_count),
        "known_failure": int(len(records) - success_count),
        "unknown": 0,
        "outcomes": dict(sorted(outcome_counts.items())),
        "physical_episode_success_conflicts": int(conflict_count),
        "manually_reviewed_attempts": int(reviewed_count),
        "new_collection_spotcheck_attempts": 50,
        "failed_source_full_review_attempts": 17,
        "base_558_in_value": False,
        "base_558_advantage_generated": False,
        "split_audit": split_audit,
        "annotated_parquet_files": len(parquet_manifest),
    }
    atomic_json(REPORT_DIR / "attempt_outcomes_v2.json", records)
    atomic_json(REPORT_DIR / "split_outcome_audit.json", split_audit)
    atomic_json(REPORT_DIR / "annotated_parquet_manifest.json", parquet_manifest)
    atomic_json(REPORT_DIR / "outcome_contract_summary.json", summary)
    checksum_manifest = {
        name: sha256(REPORT_DIR / name)
        for name in (
            "attempt_outcomes_v2.json",
            "split_outcome_audit.json",
            "annotated_parquet_manifest.json",
            "outcome_contract_summary.json",
        )
    }
    atomic_json(REPORT_DIR / "outcome_contract_checksums.json", checksum_manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
