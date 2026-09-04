#!/usr/bin/env python3
"""Build paired episode-bootstrap comparisons from completed held-out MAE runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(
    "/inspire/hdd/project/luojianlan/zhubingwen-253108120125/"
    "codex_remote_ops/evorl_hil_rl_piperx_20260902"
)
MAIN = ROOT / "reports/pi05_base_rlaware10k_rlaware20k_heldout_mae.json"
SAME = ROOT / "reports/pi05_rlaware20k_run_step10k_heldout_mae.json"
DATA_ONLY = ROOT / "reports/pi05_data_only20k_heldout_mae.json"
OUT = ROOT / "reports/pi05_heldout_mae_paired_analysis.json"


def identity(record: dict) -> tuple[int, int, int]:
    return int(record["episode_index"]), int(record["frame_index"]), int(record["global_index"])


def paired_bootstrap(
    left: list[dict], right: list[dict], field: str, seed: int
) -> dict[str, float | list[float]]:
    left_map = {identity(row): float(row[field]) for row in left}
    right_map = {identity(row): float(row[field]) for row in right}
    if left_map.keys() != right_map.keys():
        raise RuntimeError("paired anchor identity mismatch")
    by_episode: dict[int, list[float]] = defaultdict(list)
    for key in sorted(left_map):
        by_episode[key[0]].append(right_map[key] - left_map[key])
    episode_diffs = np.asarray(
        [np.mean(values) for _, values in sorted(by_episode.items())], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(20_000, dtype=np.float64)
    for index in range(draws.size):
        draws[index] = rng.choice(episode_diffs, size=len(episode_diffs), replace=True).mean()
    left_mean = float(np.mean(list(left_map.values())))
    right_mean = float(np.mean(list(right_map.values())))
    return {
        "left_mean": left_mean,
        "right_mean": right_mean,
        "right_minus_left": right_mean - left_mean,
        "relative_change_percent": (right_mean / left_mean - 1.0) * 100.0,
        "paired_episode_bootstrap_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_probability_right_better": float(np.mean(draws < 0.0)),
        "episode_mean_differences": episode_diffs.tolist(),
    }


def main() -> None:
    main = json.loads(MAIN.read_text(encoding="utf-8"))
    same = json.loads(SAME.read_text(encoding="utf-8"))
    data_only = json.loads(DATA_ONLY.read_text(encoding="utf-8"))
    records = {name: main["models"][name]["records"] for name in main["model_order"]}
    records["rlaware_20k_run_step10k"] = same["models"]["rlaware_20k_run_step10k"]["records"]
    records["data_only_20k"] = data_only["models"]["data_only_20k"]["records"]
    comparisons = {
        "base_to_rlaware_10k": {
            "physical_chunk_mae": paired_bootstrap(records["base"], records["rlaware_10k"], "physical_chunk_mae", 11),
            "normalized_chunk_mae": paired_bootstrap(records["base"], records["rlaware_10k"], "normalized_chunk_mae", 12),
        },
        "base_to_rlaware_20k": {
            "physical_chunk_mae": paired_bootstrap(records["base"], records["rlaware_20k"], "physical_chunk_mae", 21),
            "normalized_chunk_mae": paired_bootstrap(records["base"], records["rlaware_20k"], "normalized_chunk_mae", 22),
        },
        "published_4gpu_10k_to_8gpu_20k": {
            "strict_step_ablation": False,
            "physical_chunk_mae": paired_bootstrap(records["rlaware_10k"], records["rlaware_20k"], "physical_chunk_mae", 31),
            "normalized_chunk_mae": paired_bootstrap(records["rlaware_10k"], records["rlaware_20k"], "normalized_chunk_mae", 32),
        },
        "same_8gpu_trajectory_10k_to_20k": {
            "strict_step_ablation": True,
            "physical_chunk_mae": paired_bootstrap(records["rlaware_20k_run_step10k"], records["rlaware_20k"], "physical_chunk_mae", 41),
            "normalized_chunk_mae": paired_bootstrap(records["rlaware_20k_run_step10k"], records["rlaware_20k"], "normalized_chunk_mae", 42),
        },
        "base_to_data_only_20k": {
            "physical_chunk_mae": paired_bootstrap(records["base"], records["data_only_20k"], "physical_chunk_mae", 51),
            "normalized_chunk_mae": paired_bootstrap(records["base"], records["data_only_20k"], "normalized_chunk_mae", 52),
        },
        "data_only_20k_to_rlaware_20k": {
            "strict_algorithm_ablation": True,
            "physical_chunk_mae": paired_bootstrap(records["data_only_20k"], records["rlaware_20k"], "physical_chunk_mae", 61),
            "normalized_chunk_mae": paired_bootstrap(records["data_only_20k"], records["rlaware_20k"], "normalized_chunk_mae", 62),
        },
    }
    payload = {
        "contract": {
            "paired_by": ["episode_index", "frame_index", "global_index"],
            "bootstrap_unit": "episode",
            "bootstrap_draws": 20_000,
            "lower_mae_is_better": True,
            "offline_metric_not_success_rate": True,
        },
        "comparisons": comparisons,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Paired held-out MAE analysis",
        "",
        "All differences are paired on identical anchors and bootstrapped by Episode. Negative change means lower MAE.",
        "",
        "| Comparison | Physical MAE change | Relative | Paired 95% CI | P(improvement) | Strict matched comparison |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, comparison in comparisons.items():
        metric = comparison["physical_chunk_mae"]
        ci = metric["paired_episode_bootstrap_95ci"]
        strict = bool(
            comparison.get("strict_step_ablation")
            or comparison.get("strict_algorithm_ablation")
            or name.startswith("base_to")
        )
        lines.append(
            f"| {name} | {metric['right_minus_left']:+.6f} | "
            f"{metric['relative_change_percent']:+.2f}% | [{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{metric['bootstrap_probability_right_better']:.4f} | {strict} |"
        )
    lines += [
        "",
        "The 4-GPU 10k and 8-GPU 20k checkpoints are directly comparable as evaluated artifacts, but their difference cannot be attributed solely to training duration. The same-8-GPU-trajectory row is the strict 10k→20k step comparison. The Data-only 20k→RL-aware 20k row is the matched algorithm/label-use ablation.",
    ]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("MAE_PAIRED_ANALYSIS_PASS", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
