#!/usr/bin/env python3
"""Compute leakage-free train-only state/action normalization statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_array(table, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64).reshape(len(array), 14)


def stats(array: np.ndarray) -> dict:
    result = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }
    for quantile in QUANTILES:
        result[f"q{int(quantile * 100):02d}"] = np.quantile(array, quantile, axis=0).tolist()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.derived_root.resolve()
    manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    train_ids = set(map(int, manifest["episode_indices"]))
    stats_path = root / "meta/stats.json"
    before_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    before_sha = sha256_file(stats_path)
    state_rows = []
    action_rows = []
    for episode in sorted(train_ids):
        path = root / "data/chunk-000" / f"file-{episode:03d}.parquet"
        table = pq.read_table(path)
        transition_valid = np.asarray(
            table["complementary_info.transition_valid"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        action_valid = np.asarray(
            table["complementary_info.action_valid"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        state = list_array(table, "observation.state")
        action = list_array(table, "action")
        state_rows.append(state[transition_valid])
        action_rows.append(action[transition_valid & action_valid])
    state = np.concatenate(state_rows)
    action = np.concatenate(action_rows)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise RuntimeError("Non-finite value found while computing train-only stats")
    train_stats = {"observation.state": stats(state), "action": stats(action)}
    stats_path.write_text(json.dumps(train_stats, indent=4) + "\n", encoding="utf-8")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "derived_root": str(root),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest.resolve()),
        "train_episode_indices": sorted(train_ids),
        "filter": "transition_valid for state; transition_valid & action_valid for action",
        "before_stats_sha256": before_sha,
        "before_stats": before_stats,
        "after_stats_sha256": sha256_file(stats_path),
        "after_stats": train_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"TRAIN_ONLY_STATS_OK state_count={state.shape[0]} action_count={action.shape[0]} "
        f"sha256={payload['after_stats_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
