#!/usr/bin/env python3
"""Freeze episode indices for each attempt-level split after outcome adjudication."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

OUTCOMES = Path(os.environ["ATTEMPT_OUTCOMES"])
OUTPUT = Path(os.environ["ATTEMPT_SPLIT_MANIFEST"])


def main() -> None:
    records = json.loads(OUTCOMES.read_text(encoding="utf-8"))
    splits = {}
    for split in ("train", "validation", "test"):
        selected = [record for record in records if record["dataset_split"] == split]
        indices = sorted(int(record["attempt_episode_index"]) for record in selected)
        outcomes = Counter("success" if record["logical_attempt_success"] else "failure" for record in selected)
        splits[split] = {
            "episode_indices": indices,
            "attempts": len(indices),
            "success": outcomes["success"],
            "failure": outcomes["failure"],
        }
    expected = {"train": 386, "validation": 50, "test": 46}
    if {split: item["attempts"] for split, item in splits.items()} != expected:
        raise ValueError(f"Unexpected split counts: {splits}")
    payload = {
        "schema_version": "attempt-aware-splits/v2",
        "seed": 20260906,
        "grouping": ["source_collection_id", "physical_episode_uid"],
        "held_out_test_policy": "single final evaluation after mixture, steps, and checkpoint are frozen",
        "splits": splits,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "episode_indices"} for key, value in splits.items()}, indent=2))


if __name__ == "__main__":
    main()
