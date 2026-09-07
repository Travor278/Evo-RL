#!/usr/bin/env python

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class FailureLooFold:
    fold_id: str
    failure_episode_index: int
    failure_group: str
    success_control_group: str
    heldout_groups: tuple[str, str]
    train_episode_indices: tuple[int, ...]
    eval_failure_episode_indices: tuple[int, ...]
    eval_success_episode_indices: tuple[int, ...]


def _episode_index(record: dict[str, Any]) -> int:
    for field in ("attempt_episode_index", "episode_index"):
        if field in record:
            return int(record[field])
    raise KeyError("Outcome record is missing attempt_episode_index/episode_index.")


def _known(record: dict[str, Any]) -> bool:
    return bool(record.get("logical_attempt_outcome_known", False))


def _success(record: dict[str, Any]) -> bool:
    return bool(record["logical_attempt_success"])


def _tie_break(seed: int, failure_episode: int, group: str) -> str:
    payload = f"{seed}:{failure_episode}:{group}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_failure_loo_folds(
    records: list[dict[str, Any]],
    *,
    seed: int,
    group_field: str = "source_collection_id",
    expected_failures: int | None = None,
) -> list[FailureLooFold]:
    """Build deterministic group-safe failure LOO folds with unique success controls.

    Every fold excludes the complete failure collection and a metadata-matched,
    failure-free success collection. Matching never observes model predictions.
    """

    known_records = [record for record in records if _known(record)]
    if not known_records:
        raise ValueError("No known-outcome attempts were provided.")
    if any(group_field not in record or not str(record[group_field]) for record in known_records):
        raise KeyError(f"Every known record must define non-empty '{group_field}'.")

    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in known_records:
        by_group.setdefault(str(record[group_field]), []).append(record)

    failures = sorted(
        (record for record in known_records if not _success(record)), key=_episode_index
    )
    if expected_failures is not None and len(failures) != expected_failures:
        raise ValueError(f"Expected {expected_failures} failures, found {len(failures)}.")
    failure_groups = [str(record[group_field]) for record in failures]
    if len(failure_groups) != len(set(failure_groups)):
        raise ValueError("Each failure must belong to a distinct LOO group.")

    clean_success_groups = {
        group
        for group, group_records in by_group.items()
        if all(_success(record) for record in group_records)
    }
    control_use_count = {group: 0 for group in clean_success_groups}
    folds: list[FailureLooFold] = []

    for fold_number, failure in enumerate(failures):
        failure_episode = _episode_index(failure)
        failure_group = str(failure[group_field])
        failure_attempt_index = int(failure.get("logical_attempt_index", -1))
        failure_duration = float(failure.get("attempt_duration_seconds", 0.0))
        failure_source = str(failure.get("source_dataset", ""))
        failure_is_new = str(failure.get("is_new_collection", "")).lower() == "true"

        candidates: list[tuple[tuple[Any, ...], str]] = []
        for group in sorted(clean_success_groups):
            group_records = by_group[group]
            same_generation = [
                record
                for record in group_records
                if (str(record.get("is_new_collection", "")).lower() == "true") == failure_is_new
            ]
            if not same_generation:
                continue
            same_source = [
                record
                for record in same_generation
                if str(record.get("source_dataset", "")) == failure_source
            ]
            same_attempt = [
                record
                for record in same_generation
                if int(record.get("logical_attempt_index", -2)) == failure_attempt_index
            ]
            matching_records = same_attempt or same_generation
            duration_distance = min(
                abs(
                    math.log1p(float(record.get("attempt_duration_seconds", 0.0)))
                    - math.log1p(failure_duration)
                )
                for record in matching_records
            )
            score = (
                control_use_count[group],
                0 if same_source else 1,
                0 if same_attempt else 1,
                duration_distance,
                _tie_break(seed, failure_episode, group),
            )
            candidates.append((score, group))
        if not candidates:
            raise ValueError(f"No unused success control group is available for failure {failure_episode}.")

        control_group = min(candidates)[1]
        control_use_count[control_group] += 1
        heldout_groups = (failure_group, control_group)
        train_records = [
            record for record in known_records if str(record[group_field]) not in heldout_groups
        ]
        eval_records = [
            record for record in known_records if str(record[group_field]) in heldout_groups
        ]
        eval_failures = tuple(
            sorted(_episode_index(record) for record in eval_records if not _success(record))
        )
        if eval_failures != (failure_episode,):
            raise ValueError(
                f"Fold {fold_number} expected only failure {failure_episode}, found {eval_failures}."
            )
        eval_successes = tuple(sorted(_episode_index(record) for record in eval_records if _success(record)))
        if not eval_successes:
            raise ValueError(f"Fold {fold_number} has no out-of-fold success controls.")

        folds.append(
            FailureLooFold(
                fold_id=f"failure-loo-{fold_number:02d}-ep{failure_episode}",
                failure_episode_index=failure_episode,
                failure_group=failure_group,
                success_control_group=control_group,
                heldout_groups=heldout_groups,
                train_episode_indices=tuple(sorted(_episode_index(record) for record in train_records)),
                eval_failure_episode_indices=eval_failures,
                eval_success_episode_indices=eval_successes,
            )
        )
    return folds


def folds_to_dicts(folds: list[FailureLooFold]) -> list[dict[str, Any]]:
    return [asdict(fold) for fold in folds]


def _pooled_rank_auc(fold_results: list[dict[str, Any]]) -> float:
    successes = [
        float(value)
        for result in fold_results
        for value in result["success_end_values"]
    ]
    failures = [float(result["failure_end_value"]) for result in fold_results]
    wins = 0.0
    comparisons = 0
    for success in successes:
        for failure in failures:
            wins += float(success > failure) + 0.5 * float(success == failure)
            comparisons += 1
    if comparisons == 0:
        raise ValueError("LOO aggregation requires success and failure predictions.")
    return wins / comparisons


def aggregate_failure_loo(
    fold_results: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
    minimum_positive_folds: int = 5,
) -> dict[str, Any]:
    if len(fold_results) != 7:
        raise ValueError(f"The preregistered gate requires 7 folds, received {len(fold_results)}.")
    fold_ids = [str(result["fold_id"]) for result in fold_results]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("LOO fold ids must be unique.")

    separations: list[float] = []
    for result in fold_results:
        failure_value = float(result["failure_end_value"])
        success_values = [float(value) for value in result["success_end_values"]]
        if not success_values:
            raise ValueError(f"Fold {result['fold_id']} has no success predictions.")
        values = [failure_value, *success_values]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Fold {result['fold_id']} contains non-finite predictions.")
        separations.append(mean(success_values) - failure_value)

    rng = random.Random(seed)
    bootstrap_means = sorted(
        mean(rng.choice(separations) for _ in separations) for _ in range(bootstrap_samples)
    )
    low_index = max(0, int(0.025 * bootstrap_samples) - 1)
    high_index = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))
    positive_folds = sum(separation > 0.0 for separation in separations)
    median_separation = median(separations)
    mean_separation = mean(separations)
    pooled_auc = _pooled_rank_auc(fold_results)
    gates = {
        "all_finite": True,
        "at_least_5_of_7_positive": positive_folds >= minimum_positive_folds,
        "median_separation_positive": median_separation > 0.0,
        "mean_separation_positive": mean_separation > 0.0,
        "pooled_rank_auc_above_chance": pooled_auc > 0.5,
    }
    return {
        "fold_count": len(separations),
        "positive_fold_count": positive_folds,
        "fold_separations": separations,
        "median_separation": median_separation,
        "mean_separation": mean_separation,
        "mean_separation_bootstrap_95ci": [bootstrap_means[low_index], bootstrap_means[high_index]],
        "pooled_rank_auc": pooled_auc,
        "gates": gates,
        "pass": all(gates.values()),
    }
