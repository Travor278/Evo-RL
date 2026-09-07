#!/usr/bin/env python

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class AttemptSamplingStats:
    attempt_field: str
    valid_field: str
    outcome_known_field: str | None
    distinct_attempts: int
    valid_frames: int
    excluded_frames: int
    num_samples: int


class AttemptUniformSampler(Sampler[int]):
    """Sample an attempt uniformly, then a legal frame uniformly within it."""

    def __init__(self, attempts: list[torch.Tensor], num_samples: int, seed: int) -> None:
        if not attempts or any(indices.numel() == 0 for indices in attempts):
            raise ValueError("Attempt-uniform sampling requires at least one non-empty attempt.")
        if num_samples <= 0:
            raise ValueError("Attempt-uniform sampling requires num_samples > 0.")
        self.attempts = [indices.to(dtype=torch.int64, device="cpu") for indices in attempts]
        self.num_samples = int(num_samples)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            attempt_index = int(torch.randint(len(self.attempts), (), generator=self.generator).item())
            candidates = self.attempts[attempt_index]
            frame_offset = int(torch.randint(candidates.numel(), (), generator=self.generator).item())
            yield int(candidates[frame_offset].item())


def _column(dataset: Any, field: str) -> list[Any]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise TypeError("Attempt sampling requires a LeRobotDataset with an hf_dataset.")
    if field not in hf_dataset.column_names:
        raise KeyError(f"Attempt sampling field '{field}' is missing from dataset.")
    values = list(hf_dataset.with_format(None)[field])
    if len(values) != len(dataset):
        raise ValueError(f"Attempt field '{field}' has {len(values)} values, dataset has {len(dataset)}.")
    return values


def build_attempt_uniform_sampler(
    dataset: Any,
    *,
    attempt_field: str,
    valid_field: str,
    outcome_known_field: str | None,
    num_samples: int | None,
    seed: int | None,
) -> tuple[AttemptUniformSampler, AttemptSamplingStats]:
    attempt_ids = _column(dataset, attempt_field)
    valid = torch.as_tensor(_column(dataset, valid_field), dtype=torch.bool).reshape(-1)
    if outcome_known_field is not None:
        known = torch.as_tensor(_column(dataset, outcome_known_field), dtype=torch.bool).reshape(-1)
        valid &= known

    grouped: dict[str, list[int]] = {}
    for index in torch.where(valid)[0].tolist():
        attempt_id = str(attempt_ids[index])
        if not attempt_id:
            raise ValueError(f"Empty attempt id at dataset index {index}.")
        grouped.setdefault(attempt_id, []).append(index)
    attempts = [torch.tensor(indices, dtype=torch.int64) for _, indices in sorted(grouped.items())]
    sample_count = int(num_samples) if num_samples is not None else len(dataset)
    sampler = AttemptUniformSampler(attempts, sample_count, int(seed) if seed is not None else 0)
    valid_frames = int(valid.sum().item())
    stats = AttemptSamplingStats(
        attempt_field=attempt_field,
        valid_field=valid_field,
        outcome_known_field=outcome_known_field,
        distinct_attempts=len(attempts),
        valid_frames=valid_frames,
        excluded_frames=len(dataset) - valid_frames,
        num_samples=sample_count,
    )
    return sampler, stats
