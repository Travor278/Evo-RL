#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.train import ReplaySamplingConfig
from lerobot.rl.replay_sampler import build_replay_sampler


class _FakeHFDataset:
    def __init__(self, columns: dict[str, list]):
        self.column_names = list(columns)
        self._columns = columns

    def with_format(self, _format):
        return self

    def __getitem__(self, key: str):
        if key not in self._columns:
            raise KeyError(key)
        return self._columns[key]


class _FakeDataset:
    def __init__(self, source: list[int], **columns: list):
        self.hf_dataset = _FakeHFDataset({"replay_source": source, **columns})
        self._length = len(source)

    def __len__(self):
        return self._length


def _config(**overrides) -> ReplaySamplingConfig:
    values = {
        "enable": True,
        "source_field": "replay_source",
        "hil_value": 1,
        "target_hil_fraction": 0.25,
        "num_samples": 64,
    }
    values.update(overrides)
    return ReplaySamplingConfig(**values)


def test_weighted_replay_sampler_has_expected_source_mass_and_is_deterministic():
    dataset = _FakeDataset([0, 0, 0, 1])

    sampler, stats = build_replay_sampler(dataset, _config(), seed=20260902)
    duplicate, duplicate_stats = build_replay_sampler(dataset, _config(), seed=20260902)

    assert stats == duplicate_stats
    assert stats.base_count == 3
    assert stats.hil_count == 1
    assert stats.natural_hil_fraction == 0.25
    assert stats.target_hil_fraction == 0.25
    assert torch.allclose(
        sampler.weights,
        torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.double),
    )
    assert list(sampler) == list(duplicate)


def test_weighted_replay_sampler_matches_requested_imbalanced_mass():
    sampler, stats = build_replay_sampler(
        _FakeDataset([0, 0, 1, 1]),
        _config(target_hil_fraction=0.25),
        seed=7,
    )

    assert stats.base_count == stats.hil_count == 2
    assert torch.allclose(
        sampler.weights,
        torch.tensor([0.375, 0.375, 0.125, 0.125], dtype=torch.double),
    )


@pytest.mark.parametrize("source", [[0, 0], [1, 1]])
def test_weighted_replay_sampler_requires_both_sources(source):
    with pytest.raises(ValueError, match="needs both sources"):
        build_replay_sampler(_FakeDataset(source), _config(), seed=0)


def test_weighted_replay_sampler_rejects_unknown_source_value():
    with pytest.raises(ValueError, match="must contain only"):
        build_replay_sampler(_FakeDataset([0, 1, 2]), _config(), seed=0)


def test_attempt_balanced_replay_is_deterministic_and_never_samples_invalid_chunks():
    dataset = _FakeDataset(
        [0, 0, 0, 1, 1, 1, 1, 1],
        logical_attempt_id=["base", "base", "base", "short", "long", "long", "long", "long"],
        logical_action_chunk_valid_50=[1, 0, 1, 1, 1, 1, 0, 1],
    )
    cfg = _config(strategy="attempt_balanced", target_hil_fraction=0.5, num_samples=5000)
    sampler, stats = build_replay_sampler(dataset, cfg, seed=20260906)
    duplicate, duplicate_stats = build_replay_sampler(dataset, cfg, seed=20260906)

    samples = list(sampler)
    assert samples == list(duplicate)
    assert stats == duplicate_stats
    assert stats.strategy == "attempt_balanced"
    assert stats.base_valid_count == 2
    assert stats.hil_valid_count == 4
    assert stats.distinct_hil_attempts == 2
    assert set(samples) <= {0, 2, 3, 4, 5, 7}
    assert 1 not in samples and 6 not in samples

    hil_samples = [index for index in samples if index >= 3]
    short_fraction = sum(index == 3 for index in hil_samples) / len(hil_samples)
    assert 0.45 <= short_fraction <= 0.55
    source_fraction = len(hil_samples) / len(samples)
    assert 0.47 <= source_fraction <= 0.53


def test_attempt_balanced_replay_rejects_missing_attempt_field():
    dataset = _FakeDataset([0, 1], logical_action_chunk_valid_50=[1, 1])
    with pytest.raises(KeyError, match="logical_attempt_id"):
        build_replay_sampler(dataset, _config(strategy="attempt_balanced"), seed=0)
