"""Pure functions defining the SFT+RL data and selection contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class DemoEpisode:
    episode_index: int
    length: int
    source_identity: str
    task: str
    source_kind: str
    complete_trajectory: bool
    successful: bool

    def validate(self) -> None:
        if type(self.episode_index) is not int or type(self.length) is not int:
            raise ValueError("Episode index and length must be integers")
        if not isinstance(self.source_identity, str) or not isinstance(self.task, str):
            raise ValueError("Episode source identity and task must be strings")
        if self.episode_index < 0 or self.length < 2 or not self.source_identity or not self.task:
            raise ValueError(f"Invalid episode metadata: {self.episode_index}")
        if self.source_kind != "human_teleoperation" or self.complete_trajectory is not True:
            raise ValueError(f"Episode {self.episode_index} is not a complete pure human demonstration")
        if self.successful is not True:
            raise ValueError(f"Episode {self.episode_index} has no verified successful terminal")


def validate_demo_contract(contract: dict) -> list[DemoEpisode]:
    if contract.get("schema") != "sft-rl-demo/v1":
        raise ValueError("Expected a frozen sft-rl-demo/v1 manifest")
    for key in ("repo_id", "revision", "root", "provenance_evidence", "task"):
        if not contract.get(key):
            raise ValueError(f"Missing demo provenance field: {key}")
    episodes = [DemoEpisode(**row) for row in contract["episodes"]]
    if len(episodes) < 2:
        raise ValueError("At least two complete demonstrations are required")
    for ep in episodes:
        ep.validate()
        if ep.task != contract["task"]:
            raise ValueError("Z and global ranking are task-specific; do not mix tasks")
    ids = [ep.episode_index for ep in episodes]
    identities = [ep.source_identity for ep in episodes]
    if len(set(ids)) != len(ids) or len(set(identities)) != len(identities):
        raise ValueError("Duplicate source episode or source identity")
    # An explicit list, including [] when genuinely empty, is required.
    test = contract.get("final_policy_test_identities")
    if not isinstance(test, list) or not contract.get("final_policy_test_manifest_sha256"):
        raise ValueError("Final policy test exclusion manifest must be explicit and pinned")
    overlap = set(identities) & set(test)
    if overlap:
        raise ValueError(f"Final policy test leakage: {sorted(overlap)}")
    return episodes


def split_episodes(episodes: list[DemoEpisode], *, seed: int, holdout_count: int) -> dict:
    if not 0 < holdout_count < len(episodes):
        raise ValueError("Both training and holdout must contain complete episodes")
    ranked = sorted(
        episodes,
        key=lambda ep: (
            hashlib.sha256(f"sft-rl-value-v1:{seed}:{ep.source_identity}".encode()).digest(),
            ep.source_identity,
        ),
    )
    test, train = ranked[:holdout_count], ranked[holdout_count:]
    z = max(ep.length - 1 for ep in train)
    out_of_support = [
        {"episode_index": ep.episode_index, "cost": ep.length - 1, "Z": z} for ep in test if ep.length - 1 > z
    ]
    return {
        "seed": seed,
        "train_episode_indices": sorted(ep.episode_index for ep in train),
        "holdout_episode_indices": sorted(ep.episode_index for ep in test),
        "Z": z,
        "normalization": "max_train_cumulative_cost",
        "normalization_multiplier": 1,
        "out_of_support_rule": "fail",
        "out_of_support": out_of_support,
    }


def require_supported_split(split: dict) -> None:
    if split["out_of_support"]:
        raise ValueError(
            "Holdout returns outside [-1,0]; no clipping or holdout rescaling: "
            + json.dumps(split["out_of_support"])
        )


def normalized_returns(length: int, z: int) -> np.ndarray:
    if length < 2 or not isinstance(z, int) or z < 1:
        raise ValueError("Require a complete trajectory and positive integer training Z")
    result = -(length - 1 - np.arange(length, dtype=np.float64)) / z
    if result.min() < -1:
        raise ValueError(f"Out-of-support return: length={length}, Z={z}")
    return result


def n_step_advantage(
    values: np.ndarray, z: int, *, horizon: int = 50, true_success_terminal: bool = True
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite scalar predictions for a complete episode")
    if horizon != 50 or not true_success_terminal or z <= 0:
        raise ValueError(
            "This protocol requires n=50 and a true successful terminal; cropped clips are invalid"
        )
    if (values < -1).any() or (values > 0).any():
        raise ValueError("Value prediction outside model support")
    values = values.copy()
    values[-1] = 0.0
    t = np.arange(len(values) - 1)
    end = np.minimum(t + horizon, len(values) - 1)
    # There is no takeover penalty and no discount. Terminal is not an origin.
    return -(end - t) / z + values[end] - values[t]


def select_high_segments(
    episodes: list[DemoEpisode],
    predictions: dict[int, np.ndarray],
    *,
    z: int,
    value_checkpoint_sha256: str,
    contract_sha256: str,
) -> dict:
    if not value_checkpoint_sha256 or not contract_sha256:
        raise ValueError("Selection requires a pinned Value checkpoint and data contract")
    if set(predictions) != {ep.episode_index for ep in episodes}:
        raise ValueError("Predictions must cover exactly the allowed complete demonstrations")
    ordered = sorted(episodes, key=lambda ep: ep.episode_index)
    scores, origins = [], []
    for ep in ordered:
        ep.validate()
        if len(predictions[ep.episode_index]) != ep.length:
            raise ValueError(f"Incomplete predictions for episode {ep.episode_index}")
        advantage = n_step_advantage(predictions[ep.episode_index], z)
        scores.extend(advantage.tolist())
        origins.extend((ep.episode_index, t) for t in range(ep.length - 1))
    scores = np.asarray(scores, dtype=np.float64)
    requested = math.ceil(0.10 * len(scores))
    # Stable tie break: episode index, then frame index. Do NOT filter positive first.
    top = np.argsort(-scores, kind="stable")[:requested]
    chosen = top[scores[top] > 0]
    raw_threshold = float(scores[top[-1]]) if len(top) else None
    anchors = {ep.episode_index: [] for ep in ordered}
    for index in chosen:
        ep, frame = origins[int(index)]
        anchors[ep].append({"frame": frame, "advantage": float(scores[index])})
    segments = []
    for ep in ordered:
        rows = sorted(anchors[ep.episode_index], key=lambda row: row["frame"])
        for row in rows:
            start, end = row["frame"], min(row["frame"] + 20, ep.length)
            if (
                segments
                and segments[-1]["source_episode"] == ep.episode_index
                and start <= segments[-1]["source_to"]
            ):
                segments[-1]["source_to"] = max(segments[-1]["source_to"], end)
                segments[-1]["anchors"].append(row)
            else:
                segments.append(
                    {
                        "source_episode": ep.episode_index,
                        "source_identity": ep.source_identity,
                        "source_from": start,
                        "source_to": end,
                        "source_length": ep.length,
                        "anchors": [row],
                        "is_success_terminal": False,
                    }
                )
    return {
        "schema": "sft-rl-selection/v1",
        "contract_sha256": contract_sha256,
        "value_checkpoint_sha256": value_checkpoint_sha256,
        "Z": z,
        "gamma": 1,
        "advantage_horizon": 50,
        "retained_action_window": 20,
        "global_top_fraction": 0.10,
        "positive_required": True,
        "ranking": "global_top_then_positive",
        "nonterminal_origins": len(scores),
        "requested_origins": requested,
        "selected_origins": len(chosen),
        "global_top_threshold": raw_threshold,
        "selected_minimum": float(scores[chosen].min()) if len(chosen) else None,
        "score_quantiles": {str(q): float(np.quantile(scores, q)) for q in [0, 0.1, 0.5, 0.9, 0.99, 1]},
        "retained_frames": sum(row["source_to"] - row["source_from"] for row in segments),
        "segments": segments,
    }


def validate_selection(selection: dict, contract: dict) -> None:
    episodes = {ep.episode_index: ep for ep in validate_demo_contract(contract)}
    expected = {
        "schema": "sft-rl-selection/v1",
        "contract_sha256": canonical_sha256(contract),
        "gamma": 1,
        "advantage_horizon": 50,
        "retained_action_window": 20,
        "global_top_fraction": 0.1,
        "positive_required": True,
        "ranking": "global_top_then_positive",
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"Selection protocol mismatch: {key}")
    origins = sum(ep.length - 1 for ep in episodes.values())
    requested = math.ceil(0.1 * origins)
    if selection.get("nonterminal_origins") != origins or selection.get("requested_origins") != requested:
        raise ValueError("Global Top10% origin counts do not match the allowed complete demonstrations")
    seen_anchors = set()
    previous = {}
    for row in selection["segments"]:
        ep = episodes[row["source_episode"]]
        a, b = row["source_from"], row["source_to"]
        if (
            not 0 <= a < b <= ep.length
            or row["source_length"] != ep.length
            or row["source_identity"] != ep.source_identity
        ):
            raise ValueError("Invalid or cross-episode selected interval")
        if row["is_success_terminal"] is not False or not row["anchors"]:
            raise ValueError("A clipped interval cannot claim task success")
        if a <= previous.get(ep.episode_index, -1):
            raise ValueError("Intervals must be sorted and already merged, including adjacency")
        previous[ep.episode_index] = b
        cover = np.zeros(b - a, dtype=bool)
        for anchor in row["anchors"]:
            t = anchor["frame"]
            identity = (ep.episode_index, t)
            if identity in seen_anchors:
                raise ValueError("Duplicate selected anchor")
            seen_anchors.add(identity)
            if (
                not a <= t < min(b, ep.length - 1)
                or not math.isfinite(anchor["advantage"])
                or anchor["advantage"] <= 0
            ):
                raise ValueError("Invalid selected anchor")
            cover[t - a : min(t + 20, ep.length) - a] = True
        if not cover.all():
            raise ValueError("Segment contains frames not covered by selected 20-action windows")
        if a != min(x["frame"] for x in row["anchors"]) or b != max(
            min(x["frame"] + 20, ep.length) for x in row["anchors"]
        ):
            raise ValueError(
                "A selected segment must contain each full retained window up to the source endpoint"
            )
    if selection.get("selected_origins") != len(seen_anchors) or len(seen_anchors) > requested:
        raise ValueError("Selected origin count exceeds or misreports the global Top10% budget")
    retained = sum(row["source_to"] - row["source_from"] for row in selection["segments"])
    if selection.get("retained_frames") != retained:
        raise ValueError("Retained frame count differs from merged intervals")


def validate_value_provenance(provenance: dict, contract: dict) -> None:
    episodes = {ep.episode_index: ep for ep in validate_demo_contract(contract)}
    required = {
        "source_kind": "human_teleoperation",
        "contract_sha256": canonical_sha256(contract),
        "normalization": "max_train_cumulative_cost",
        "normalization_multiplier": 1,
        "takeover_penalty": 0,
        "gamma": 1,
        "architecture": "pure_vision_201_two_hot",
        "initialization": "pretrained_vision_random_head",
    }
    for key, value in required.items():
        if provenance.get(key) != value:
            raise ValueError(f"Value checkpoint is not eligible for this pure-demo experiment: {key}")
    train = provenance.get("train_episode_indices", [])
    test = provenance.get("holdout_episode_indices", [])
    if not train or not test or len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("Missing or duplicate Value split identities")
    if set(train) & set(test) or set(train) | set(test) != set(episodes):
        raise ValueError("Value checkpoint split differs from the allowed complete episodes")
    z = max(episodes[e].length - 1 for e in train)
    if provenance.get("Z") != z:
        raise ValueError("Value Z is not the maximum training cumulative cost")
    if any(episodes[e].length - 1 > z for e in test):
        raise ValueError("Out-of-support holdout: do not reuse a clipped or rescaled checkpoint")
