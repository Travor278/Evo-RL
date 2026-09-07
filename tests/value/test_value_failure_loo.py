#!/usr/bin/env python

import pytest

from lerobot.rl.value_failure_loo import aggregate_failure_loo, build_failure_loo_folds


def _record(index, group, success, attempt_index=0, duration=1.0, source="old"):
    return {
        "attempt_episode_index": index,
        "source_collection_id": group,
        "logical_attempt_outcome_known": True,
        "logical_attempt_success": success,
        "logical_attempt_index": attempt_index,
        "attempt_duration_seconds": duration,
        "source_dataset": source,
        "is_new_collection": False,
    }


def test_build_failure_loo_folds_is_group_safe_and_deterministic():
    records = []
    for index in range(7):
        records.append(_record(index, f"failure-{index}", False, index % 2, index + 1))
        records.append(_record(100 + index, f"failure-{index}", True, (index + 1) % 2, index + 2))
    for index in range(10):
        records.append(_record(200 + index, f"success-{index}", True, index % 2, index + 1.5))

    folds = build_failure_loo_folds(records, seed=20260906, expected_failures=7)
    duplicate = build_failure_loo_folds(records, seed=20260906, expected_failures=7)
    assert folds == duplicate
    assert len(folds) == 7
    assert len({fold.success_control_group for fold in folds}) == 7
    for fold in folds:
        train_indices = set(fold.train_episode_indices)
        heldout_indices = set(fold.eval_failure_episode_indices + fold.eval_success_episode_indices)
        assert train_indices.isdisjoint(heldout_indices)
        assert fold.eval_failure_episode_indices == (fold.failure_episode_index,)
        assert fold.eval_success_episode_indices


def test_build_failure_loo_rejects_multiple_failures_in_one_group():
    records = [_record(0, "shared", False), _record(1, "shared", False)]
    with pytest.raises(ValueError, match="distinct LOO group"):
        build_failure_loo_folds(records, seed=1)


def test_build_failure_loo_allows_different_source_repo_with_same_collection_generation():
    records = []
    for index in range(7):
        records.append(_record(index, f"failure-{index}", False, source="failed-source"))
        control = _record(100 + index, f"success-{index}", True, source="success-source")
        records.append(control)
    folds = build_failure_loo_folds(records, seed=1, expected_failures=7)
    assert len(folds) == 7


def test_build_failure_loo_balances_reused_control_groups_when_only_three_exist():
    records = [_record(index, f"failure-{index}", False) for index in range(7)]
    records.extend(_record(100 + index, f"success-{index}", True) for index in range(3))
    folds = build_failure_loo_folds(records, seed=1, expected_failures=7)
    counts = {}
    for fold in folds:
        counts[fold.success_control_group] = counts.get(fold.success_control_group, 0) + 1
    assert sorted(counts.values()) == [2, 2, 3]


def test_aggregate_failure_loo_passes_preregistered_gate():
    results = [
        {"fold_id": f"fold-{index}", "failure_end_value": -0.5, "success_end_values": [-0.1, -0.2]}
        for index in range(7)
    ]
    report = aggregate_failure_loo(results, seed=20260906, bootstrap_samples=1000)
    assert report["pass"] is True
    assert report["positive_fold_count"] == 7
    assert report["pooled_rank_auc"] == 1.0


def test_aggregate_failure_loo_fails_when_only_four_folds_are_positive():
    results = []
    for index in range(7):
        failure = -0.5 if index < 4 else -0.05
        results.append(
            {"fold_id": f"fold-{index}", "failure_end_value": failure, "success_end_values": [-0.1]}
        )
    report = aggregate_failure_loo(results, seed=20260906, bootstrap_samples=1000)
    assert report["positive_fold_count"] == 4
    assert report["pass"] is False
