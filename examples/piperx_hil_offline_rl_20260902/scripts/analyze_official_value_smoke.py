#!/usr/bin/env python3
"""Analyze value/advantage/ACP outputs with explicit autonomous/HIL semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_VALUE = "complementary_info.value_evorl_official_smoke_r1"
DEFAULT_ADVANTAGE = "complementary_info.advantage_evorl_official_smoke_r1"
DEFAULT_INDICATOR = "complementary_info.acp_indicator_evorl_official_smoke_r1"
INTERVENTION = "complementary_info.is_intervention"


def scalar(table, key, dtype=float):
    return np.asarray(table[key].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-field", default=DEFAULT_VALUE)
    parser.add_argument("--advantage-field", default=DEFAULT_ADVANTAGE)
    parser.add_argument("--indicator-field", default=DEFAULT_INDICATOR)
    args = parser.parse_args()
    value_field = args.value_field
    advantage_field = args.advantage_field
    indicator_field = args.indicator_field
    split_for_episode = {}
    for split in ("train", "val", "test"):
        manifest = json.loads((args.manifest_root / f"{split}_episodes.json").read_text())
        for episode in manifest["episode_indices"]:
            split_for_episode[int(episode)] = split

    meta_files = sorted((args.derived_root / "meta/episodes/chunk-000").glob("*.parquet"))
    meta = pq.read_table(meta_files).to_pydict()
    meta_by_episode = {
        int(meta["episode_index"][i]): {key: meta[key][i] for key in meta}
        for i in range(len(meta["episode_index"]))
    }
    episodes = []
    grouped_values = {}
    grouped_advantages = {}
    grouped_indicators = {}
    for path in sorted((args.derived_root / "data/chunk-000").glob("*.parquet")):
        table = pq.read_table(path)
        missing = [key for key in (value_field, advantage_field, indicator_field) if key not in table.column_names]
        if missing:
            raise RuntimeError(f"Missing inferred fields in {path}: {missing}")
        episode_index = int(scalar(table, "episode_index", int)[0])
        value = scalar(table, value_field)
        advantage = scalar(table, advantage_field)
        indicator = scalar(table, indicator_field, int)
        intervention = scalar(table, INTERVENTION, bool)
        row = meta_by_episode[episode_index]
        autonomous = int(row["intervention_count"]) == 0
        success_label = row["episode_success"]
        success = success_label == "success" if isinstance(success_label, str) else bool(success_label)
        category = (
            "autonomous_success" if autonomous and success else
            "autonomous_failure" if autonomous else
            "hil_recovered_success" if success else
            "hil_unrecovered_failure"
        )
        split = split_for_episode[episode_index]
        key = f"{split}/{category}"
        grouped_values.setdefault(key, []).append(value)
        grouped_advantages.setdefault(key, []).append(advantage)
        grouped_indicators.setdefault(key, []).append(indicator)
        onset_deltas = []
        onset = np.flatnonzero(intervention & np.r_[True, ~intervention[:-1]])
        for index in onset:
            before = value[max(0, index - 50):index]
            after = value[index:min(len(value), index + 50)]
            if before.size and after.size:
                onset_deltas.append(float(after.mean() - before.mean()))
        episodes.append(
            {
                "episode_index": episode_index,
                "split": split,
                "category": category,
                "success": success,
                "interventions": int(row["intervention_count"]),
                "value": summarize(value),
                "advantage": summarize(advantage),
                "indicator_fraction": float(indicator.mean()),
                "intervention_onset_value_delta_mean": float(np.mean(onset_deltas)) if onset_deltas else None,
            }
        )

    payload = {
        "fields": {"value": value_field, "advantage": advantage_field, "indicator": indicator_field},
        "episodes": sorted(episodes, key=lambda item: item["episode_index"]),
        "groups": {},
    }
    for key in sorted(grouped_values):
        values = np.concatenate(grouped_values[key])
        advantages = np.concatenate(grouped_advantages[key])
        indicators = np.concatenate(grouped_indicators[key])
        payload["groups"][key] = {
            "value": summarize(values),
            "advantage": summarize(advantages),
            "indicator_fraction": float(indicators.mean()),
        }

    val_success = payload["groups"].get("val/autonomous_success")
    val_failure = payload["groups"].get("val/autonomous_failure")
    separation = None
    if val_success and val_failure:
        separation = val_success["value"]["mean"] - val_failure["value"]["mean"]
    all_values = np.concatenate([array for arrays in grouped_values.values() for array in arrays])
    all_advantages = np.concatenate([array for arrays in grouped_advantages.values() for array in arrays])
    all_indicators = np.concatenate([array for arrays in grouped_indicators.values() for array in arrays])
    payload["global"] = {
        "value": summarize(all_values),
        "advantage": summarize(all_advantages),
        "indicator_fraction": float(all_indicators.mean()),
        "val_autonomous_success_minus_failure_mean_value": separation,
    }
    payload["gates"] = {
        "finite_value": bool(np.isfinite(all_values).all()),
        "finite_advantage": bool(np.isfinite(all_advantages).all()),
        "value_not_collapsed": float(all_values.std()) > 1e-3,
        "advantage_not_collapsed": float(all_advantages.std()) > 1e-4,
        "indicator_not_all_same": 0.05 < float(all_indicators.mean()) < 0.95,
        "val_autonomous_success_above_failure": separation is not None and separation > 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"global": payload["global"], "gates": payload["gates"]}, indent=2, sort_keys=True))
    return 0 if all(payload["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
