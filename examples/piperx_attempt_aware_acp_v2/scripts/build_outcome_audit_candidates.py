from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SNAPSHOT = Path(os.environ["ATTEMPT_SNAPSHOT"])
OUTPUT_DIR = Path(os.environ["OUTCOME_AUDIT_OUTPUT"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_episode_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


attempts = read_episode_rows(SNAPSHOT / "meta/episodes")
physical = read_episode_rows(SNAPSHOT / "physical_master/meta/episodes")
physical_by_uid = {row["physical_episode_uid"]: row for row in physical}

records: list[dict] = []
for attempt in sorted(attempts, key=lambda row: row["episode_index"]):
    chunk = int(attempt["data/chunk_index"])
    file_index = int(attempt["data/file_index"])
    data_path = SNAPSHOT / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
    table = pq.read_table(
        data_path,
        columns=[
            "complementary_info.is_intervention",
            "complementary_info.action_source",
            "complementary_info.intervention_source",
            "logical_attempt_terminal",
            "logical_transition_valid",
            "logical_action_chunk_valid_50",
        ],
    )
    data = table.to_pydict()
    intervened_frames = sum(bool(value) for value in data["complementary_info.is_intervention"])
    action_sources = Counter(str(value) for value in data["complementary_info.action_source"])
    intervention_sources = Counter(
        str(value)
        for value in data["complementary_info.intervention_source"]
        if value not in (None, "", "None")
    )
    physical_row = physical_by_uid[attempt["physical_episode_uid"]]
    source_dataset = str(attempt["source_dataset"])
    is_new = source_dataset == "EvoStudio local collection snapshot"
    source_success = bool(attempt["episode_success"])
    if not source_success:
        review_group = "failed_source_full_review"
    elif not is_new:
        review_group = "old_source_success_review"
    else:
        review_group = "new_declared_complete_pool"
    records.append(
        {
            "attempt_episode_index": int(attempt["episode_index"]),
            "logical_attempt_id": attempt["logical_attempt_id"],
            "logical_attempt_index": int(attempt["logical_attempt_index"]),
            "physical_episode_uid": attempt["physical_episode_uid"],
            "parent_physical_episode_index": int(attempt["parent_physical_episode_index"]),
            "source_collection_id": attempt["source_collection_id"],
            "source_episode_index": int(attempt["source_episode_index"]),
            "source_episode_uid": attempt["source_episode_uid"],
            "source_dataset": source_dataset,
            "is_new_collection": is_new,
            "dataset_split": attempt["dataset_split"],
            "attempt_length_frames": int(attempt["length"]),
            "attempt_duration_seconds": float(attempt["length"]) / 30.0,
            "source_episode_success": source_success,
            "source_termination_reason": attempt["termination_reason"],
            "source_failure_note_present": bool(attempt["failure_note_present"]),
            "physical_attempt_count": int(physical_row["logical_attempt_count"]),
            "logical_attempt_intervened_evidence": intervened_frames > 0,
            "intervened_frame_count": intervened_frames,
            "action_source_counts": dict(sorted(action_sources.items())),
            "intervention_source_counts": dict(sorted(intervention_sources.items())),
            "terminal_frame_count": sum(bool(value) for value in data["logical_attempt_terminal"]),
            "transition_valid_count": sum(bool(value) for value in data["logical_transition_valid"]),
            "chunk_valid_50_count": sum(bool(value) for value in data["logical_action_chunk_valid_50"]),
            "review_group": review_group,
            "logical_attempt_outcome_known": False,
            "logical_attempt_success": None,
            "logical_attempt_terminal_reason": "unknown",
            "logical_attempt_failure_type": "unknown",
            "logical_attempt_autonomous_success": False,
            "review_status": "pending",
            "review_evidence": "",
        }
    )

# New-data spot check: choose one deterministic physical episode per collection,
# then inspect all five attempts. This covers every collection and attempt index.
new_by_collection: dict[str, list[str]] = defaultdict(list)
for record in records:
    if record["review_group"] == "new_declared_complete_pool":
        new_by_collection[record["source_collection_id"]].append(record["physical_episode_uid"])
selected_new_uids = {
    sorted(set(uids), key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())[0]
    for uids in new_by_collection.values()
}
for record in records:
    if (
        record["review_group"] == "new_declared_complete_pool"
        and record["physical_episode_uid"] in selected_new_uids
    ):
        record["review_group"] = "new_required_spotcheck"

fieldnames = list(records[0])
with (OUTPUT_DIR / "attempt_outcome_audit.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        row = dict(record)
        row["action_source_counts"] = json.dumps(row["action_source_counts"], sort_keys=True)
        row["intervention_source_counts"] = json.dumps(row["intervention_source_counts"], sort_keys=True)
        writer.writerow(row)

(OUTPUT_DIR / "attempt_outcome_audit.json").write_text(json.dumps(records, indent=2) + "\n")
review_records = [
    record
    for record in records
    if record["review_group"]
    in {
        "failed_source_full_review",
        "old_source_success_review",
        "new_required_spotcheck",
    }
]
(OUTPUT_DIR / "manual_review_manifest.json").write_text(
    json.dumps(review_records, indent=2) + "\n"
)

summary = {
    "attempts": len(records),
    "physical_episodes": len(physical),
    "review_groups": dict(Counter(record["review_group"] for record in records)),
    "source_success": dict(Counter(str(record["source_episode_success"]) for record in records)),
    "intervened_attempts": sum(record["logical_attempt_intervened_evidence"] for record in records),
    "manual_review_attempts": len(review_records),
    "new_spotcheck_physical_episodes": len(selected_new_uids),
    "all_outcomes_initially_unknown": all(not record["logical_attempt_outcome_known"] for record in records),
}
(OUTPUT_DIR / "outcome_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
