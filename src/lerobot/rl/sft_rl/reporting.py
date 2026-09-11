"""Offline plots and protocol-bound aggregation of operator-supplied robot trials."""

import json
import math
from pathlib import Path

import numpy as np

from .protocol import canonical_sha256, n_step_advantage


def plot_reports(metrics_path, predictions_path, selection_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [
        json.loads(line)["evaluation"]
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record is not None]
    if not records:
        raise ValueError("No measured Value evaluation records to plot")
    prediction = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if (
        prediction["value_checkpoint_sha256"] != selection["value_checkpoint_sha256"]
        or prediction["contract_sha256"] != selection["contract_sha256"]
    ):
        raise ValueError("Plots must use the same pinned Value/selection/data artifacts")
    steps = [row["step"] for row in records]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for metric, axis in zip(["CE", "MAE"], axes, strict=True):
        for split in ["train", "holdout"]:
            axis.plot(steps, [row[split][metric] for row in records], label=split)
        axis.set(xlabel="Optimizer update", ylabel=metric)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "value_ce_mae.png", dpi=180)
    plt.close(figure)
    scores = np.concatenate(
        [n_step_advantage(np.asarray(v), selection["Z"]) for v in prediction["predictions"].values()]
    )
    lengths = [row["source_to"] - row["source_from"] for row in selection["segments"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(scores, bins=80)
    if selection["global_top_threshold"] is not None:
        axes[0].axvline(selection["global_top_threshold"], color="crimson", label="Global top10% cutoff")
    axes[0].axvline(0, color="black", linestyle="--", label="A > 0 boundary")
    axes[0].set(xlabel="50-action advantage", ylabel="Nonterminal origins")
    axes[0].legend()
    axes[1].hist(lengths, bins=min(50, max(1, len(set(lengths)))))
    axes[1].set(xlabel="Retained segment length (actions)", ylabel="Segments")
    figure.tight_layout()
    figure.savefig(output / "advantage_and_segments.png", dpi=180)
    plt.close(figure)
    return {
        "value_plot": str(output / "value_ce_mae.png"),
        "selection_plot": str(output / "advantage_and_segments.png"),
    }


def aggregate_robot_trials(protocol, trials, *, checkpoint_sha256, policy_run_protocol_sha256):
    """Never infer SR/TP from model loss; all inputs must be measured trials."""
    required = (
        "task_setup",
        "initialization",
        "time_limit_seconds",
        "camera_mapping",
        "RTC",
        "scoring_rule",
        "stage_names",
        "throughput",
    )
    for key in required:
        if key not in protocol or protocol[key] is None:
            raise ValueError(f"Missing paper-matched real-robot protocol: {key}")
    if not trials:
        raise ValueError("No real-robot trial records; cannot fill the paper SFT+RL row")
    if not protocol["stage_names"] or len(set(protocol["stage_names"])) != len(protocol["stage_names"]):
        raise ValueError("Explicit, unique evaluation stages are required")
    if (
        not isinstance(protocol["time_limit_seconds"], (int, float))
        or not math.isfinite(protocol["time_limit_seconds"])
        or protocol["time_limit_seconds"] <= 0
    ):
        raise ValueError("Invalid fixed evaluation time limit")
    expected = canonical_sha256(protocol)
    seen = set()
    stages = {name: {"successes": 0, "attempts": 0} for name in protocol["stage_names"]}
    successes, elapsed, completed_units = 0, 0.0, 0
    for trial in trials:
        if not trial.get("trial_id") or trial["trial_id"] in seen or trial.get("protocol_sha256") != expected:
            raise ValueError("Duplicate trial or mismatched robot evaluation protocol")
        seen.add(trial["trial_id"])
        if trial.get("group") != "SFT+RL" or trial.get("evidence_kind") != "real_robot":
            raise ValueError(
                "Only actual SFT+RL robot trials are eligible; offline imitation metrics are not SR"
            )
        if (
            trial.get("checkpoint_sha256") != checkpoint_sha256
            or trial.get("policy_run_protocol_sha256") != policy_run_protocol_sha256
        ):
            raise ValueError("Trial checkpoint/provenance differs from the evaluated SFT+RL policy")
        if type(trial.get("success")) is not bool or not isinstance(
            trial.get("elapsed_seconds"), (int, float)
        ):
            raise ValueError("Explicit observed success and elapsed time are required")
        if not math.isfinite(trial["elapsed_seconds"]) or trial["elapsed_seconds"] <= 0:
            raise ValueError("Invalid measured trial duration")
        if set(trial["stages"]) != set(stages):
            raise ValueError("Missing stage attempts/successes")
        for name, row in trial["stages"].items():
            if (
                type(row.get("attempted")) is not bool
                or type(row.get("success")) is not bool
                or row["success"]
                and not row["attempted"]
            ):
                raise ValueError("Invalid observed stage record")
            stages[name]["attempts"] += int(row["attempted"])
            stages[name]["successes"] += int(row["success"])
        if protocol["scoring_rule"] == "all_stages" and trial["success"] != all(
            row["success"] for row in trial["stages"].values()
        ):
            raise ValueError("Overall success differs from the declared all-stages scoring rule")
        successes += trial["success"]
        elapsed += trial["elapsed_seconds"]
        if protocol["throughput"]["numerator"] == "completed_units":
            units = trial.get("completed_units")
            if type(units) is not int or units < 0:
                raise ValueError("Throughput requires observed completed-unit counts")
            completed_units += units
    throughput = protocol["throughput"]
    if throughput["numerator"] not in ("successes", "completed_units") or throughput["denominator"] not in (
        "elapsed_seconds",
        "scheduled_seconds",
    ):
        raise ValueError("Specify the paper TP definition explicitly; it is not guessed")
    unit = throughput.get("unit_seconds")
    if not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit <= 0:
        raise ValueError("Specify the paper throughput time unit")
    denominator = (
        elapsed
        if throughput["denominator"] == "elapsed_seconds"
        else len(trials) * protocol["time_limit_seconds"]
    )
    if denominator <= 0:
        raise ValueError("Invalid throughput denominator")
    numerator = successes if throughput["numerator"] == "successes" else completed_units
    for row in stages.values():
        row["SR"] = row["successes"] / row["attempts"] if row["attempts"] else None
    return {
        "group": "SFT+RL",
        "protocol_sha256": expected,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_run_protocol_sha256": policy_run_protocol_sha256,
        "successes": successes,
        "attempts": len(trials),
        "SR": successes / len(trials),
        "TP": numerator / denominator * unit,
        "TP_definition": throughput,
        "stages": stages,
    }
