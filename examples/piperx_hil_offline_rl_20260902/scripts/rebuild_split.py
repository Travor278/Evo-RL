#!/usr/bin/env python3
"""Rebuild group-disjoint split while reserving autonomous-success validation support."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from audit_dataset import Group, choose_group_split, json_dump, md_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    root = args.output_root.resolve()
    inventory_path = root / "manifests/dataset_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    episodes = inventory["episodes"]

    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["group_id"]].append(episode)
    groups = [
        Group(
            name=name,
            episodes=[item["episode_index"] for item in items],
            frames=sum(item["rows"] for item in items),
            failures=sum(not item["episode_success"] for item in items),
            autonomous_successes=sum(item["episode_success"] and item["intervention_count"] == 0 for item in items),
            autonomous_failures=sum((not item["episode_success"]) and item["intervention_count"] == 0 for item in items),
        )
        for name, items in grouped.items()
    ]
    split_groups = choose_group_split(groups, args.seed, len(episodes), inventory["totals"]["frames"])
    group_lookup = {group.name: group for group in groups}
    split_ids = {
        split: sorted(ep for group_name in names for ep in group_lookup[group_name].episodes)
        for split, names in split_groups.items()
    }
    all_groups = [group for names in split_groups.values() for group in names]
    if len(all_groups) != len(set(all_groups)):
        raise RuntimeError("Group leakage detected")

    inventory["split"] = {
        "seed": args.seed,
        "group_key": "policy_hil.session_id",
        "constraints": {
            "train_failure_min": 2,
            "val_failure_min": 1,
            "test_failure_min": 1,
            "train_autonomous_success_min": 1,
            "val_autonomous_success_min": 1,
            "test_autonomous_success_min": 0,
            "note": "Only two independent sessions contain autonomous successes, so all three splits cannot contain one without session leakage.",
        },
        "groups": split_groups,
        "episodes": split_ids,
    }
    json_dump(inventory_path, inventory)

    common = {
        "repo_id": inventory["repo_id"],
        "revision": inventory["revision"],
        "seed": args.seed,
        "split_method": "group-disjoint randomized search; group=policy_hil.session_id; failure coverage in all splits; autonomous success required in train/val",
    }
    rows = []
    for split, ids in split_ids.items():
        selected = [item for item in episodes if item["episode_index"] in ids]
        payload = {
            **common,
            "split": split,
            "groups": split_groups[split],
            "episode_indices": ids,
            "episodes": selected,
        }
        payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        json_dump(root / f"manifests/{split}_episodes.json", payload)
        rows.append(
            [
                split,
                len(ids),
                sum(item["rows"] for item in selected),
                sum(item["episode_success"] for item in selected),
                sum(not item["episode_success"] for item in selected),
                sum(item["episode_success"] and item["intervention_count"] == 0 for item in selected),
                sum((not item["episode_success"]) and item["intervention_count"] == 0 for item in selected),
                len(split_groups[split]),
            ]
        )

    report_path = root / "reports/dataset_audit.md"
    report = report_path.read_text(encoding="utf-8")
    new_table = md_table(
        ["Split", "Episodes", "Frames", "Success", "Failure", "Autonomous success", "Autonomous failure", "Groups"],
        rows,
    )
    pattern = r"\| Split \| Episodes \| Frames \| Success \| Failure \| Autonomous success \| Autonomous failure \| Groups \|\n\|.*?\n(?:\|.*?\n){3}"
    report, replacements = re.subn(pattern, new_table + "\n", report, count=1)
    if replacements != 1:
        raise RuntimeError("Could not replace split table in dataset_audit.md")
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"groups": split_groups, "episodes": split_ids, "summary": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
