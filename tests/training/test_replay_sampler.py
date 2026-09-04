#!/usr/bin/env python

import pytest
import torch

from lerobot.configs.train import ReplaySamplingConfig
from lerobot.rl.replay_sampler import build_replay_sampler


class _FakeHFDataset:
    def __init__(self, source: list[int]):
        self.column_names = ["replay_source"]
        self._source = source

    def with_format(self, _format):
        return self

    def __getitem__(self, key: str):
        if key != "replay_source":
            raise KeyError(key)
        return self._source


class _FakeDataset:
    def __init__(self, source: list[int]):
        self.hf_dataset = _FakeHFDataset(source)
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
