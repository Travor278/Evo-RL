#!/usr/bin/env python

import pytest

from lerobot.rl.attempt_sampler import build_attempt_uniform_sampler


class _FakeHFDataset:
    def __init__(self, columns: dict[str, list]):
        self.column_names = list(columns)
        self.columns = columns

    def with_format(self, _format):
        return self

    def __getitem__(self, field: str):
        return self.columns[field]


class _FakeDataset:
    def __init__(self, columns: dict[str, list]):
        self.hf_dataset = _FakeHFDataset(columns)

    def __len__(self):
        return len(next(iter(self.hf_dataset.columns.values())))


def test_attempt_uniform_sampler_equalizes_short_and_long_attempts():
    dataset = _FakeDataset(
        {
            "attempt": ["short", "long", "long", "long", "long", "unknown"],
            "valid": [1, 1, 1, 1, 1, 1],
            "known": [1, 1, 1, 1, 1, 0],
        }
    )
    sampler, stats = build_attempt_uniform_sampler(
        dataset,
        attempt_field="attempt",
        valid_field="valid",
        outcome_known_field="known",
        num_samples=5000,
        seed=20260906,
    )
    duplicate, duplicate_stats = build_attempt_uniform_sampler(
        dataset,
        attempt_field="attempt",
        valid_field="valid",
        outcome_known_field="known",
        num_samples=5000,
        seed=20260906,
    )
    samples = list(sampler)
    assert samples == list(duplicate)
    assert stats == duplicate_stats
    assert stats.distinct_attempts == 2
    assert stats.valid_frames == 5
    assert stats.excluded_frames == 1
    assert 5 not in samples
    short_fraction = sum(index == 0 for index in samples) / len(samples)
    assert 0.47 <= short_fraction <= 0.53


def test_attempt_uniform_sampler_rejects_missing_field():
    dataset = _FakeDataset({"attempt": ["a"], "valid": [1]})
    with pytest.raises(KeyError, match="known"):
        build_attempt_uniform_sampler(
            dataset,
            attempt_field="attempt",
            valid_field="valid",
            outcome_known_field="known",
            num_samples=1,
            seed=0,
        )


def test_outcome_balanced_sampler_caps_failure_fraction_and_reports_unique_failures():
    dataset = _FakeDataset(
        {
            "attempt": ["success-short", "success-long", "success-long", "failure", "failure"],
            "valid": [1, 1, 1, 1, 1],
            "known": [1, 1, 1, 1, 1],
            "success": [1, 1, 1, 0, 0],
        }
    )
    sampler, stats = build_attempt_uniform_sampler(
        dataset,
        attempt_field="attempt",
        valid_field="valid",
        outcome_known_field="known",
        outcome_success_field="success",
        failure_fraction=0.1,
        num_samples=20000,
        seed=20260906,
    )
    samples = list(sampler)
    observed_failure_fraction = sum(index in {3, 4} for index in samples) / len(samples)
    assert 0.085 <= observed_failure_fraction <= 0.115
    assert stats.success_attempts == 2
    assert stats.failure_attempts == 1
    assert stats.target_failure_fraction == 0.1
