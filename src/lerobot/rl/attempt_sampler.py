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
    success_attempts: int
    failure_attempts: int
    target_failure_fraction: float | None
    valid_frames: int
    terminal_frames_included: int
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


class AttemptOutcomeBalancedSampler(Sampler[int]):
    """Choose outcome with a capped target fraction, then attempt and legal frame uniformly."""

    def __init__(
        self,
        *,
        success_attempts: list[torch.Tensor],
        failure_attempts: list[torch.Tensor],
        failure_fraction: float,
        num_samples: int,
        seed: int,
    ) -> None:
        if not success_attempts or not failure_attempts:
            raise ValueError("Outcome-balanced attempt sampling requires both success and failure attempts.")
        self.success_attempts = success_attempts
        self.failure_attempts = failure_attempts
        self.failure_fraction = float(failure_fraction)
        self.num_samples = int(num_samples)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            attempts = (
                self.failure_attempts
                if torch.rand((), generator=self.generator).item() < self.failure_fraction
                else self.success_attempts
            )
            attempt_index = int(torch.randint(len(attempts), (), generator=self.generator).item())
            candidates = attempts[attempt_index]
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
    terminal_field: str | None = None,
    outcome_known_field: str | None,
    outcome_success_field: str = "logical_attempt_success",
    failure_fraction: float | None = None,
    num_samples: int | None,
    seed: int | None,
) -> tuple[Sampler[int], AttemptSamplingStats]:
    attempt_ids = _column(dataset, attempt_field)
    valid = torch.as_tensor(_column(dataset, valid_field), dtype=torch.bool).reshape(-1)
    terminal_frames_included = 0
    if terminal_field is not None:
        terminal = torch.as_tensor(_column(dataset, terminal_field), dtype=torch.bool).reshape(-1)
        terminal_frames_included = int((terminal & ~valid).sum().item())
        valid |= terminal
    if outcome_known_field is not None:
        known = torch.as_tensor(_column(dataset, outcome_known_field), dtype=torch.bool).reshape(-1)
        valid &= known

    success_values = _column(dataset, outcome_success_field) if failure_fraction is not None else None
    grouped: dict[str, list[int]] = {}
    attempt_success: dict[str, bool] = {}
    for index in torch.where(valid)[0].tolist():
        attempt_id = str(attempt_ids[index])
        if not attempt_id:
            raise ValueError(f"Empty attempt id at dataset index {index}.")
        grouped.setdefault(attempt_id, []).append(index)
        if success_values is not None:
            success = bool(success_values[index])
            if attempt_id in attempt_success and attempt_success[attempt_id] != success:
                raise ValueError(f"Attempt '{attempt_id}' has inconsistent outcome labels.")
            attempt_success[attempt_id] = success
    attempts = [torch.tensor(indices, dtype=torch.int64) for _, indices in sorted(grouped.items())]
    sample_count = int(num_samples) if num_samples is not None else len(dataset)
    if failure_fraction is None:
        success_attempts = []
        failure_attempts = []
        sampler: Sampler[int] = AttemptUniformSampler(
            attempts, sample_count, int(seed) if seed is not None else 0
        )
    else:
        success_attempts = [
            torch.tensor(grouped[attempt_id], dtype=torch.int64)
            for attempt_id in sorted(grouped)
            if attempt_success[attempt_id]
        ]
        failure_attempts = [
            torch.tensor(grouped[attempt_id], dtype=torch.int64)
            for attempt_id in sorted(grouped)
            if not attempt_success[attempt_id]
        ]
        sampler = AttemptOutcomeBalancedSampler(
            success_attempts=success_attempts,
            failure_attempts=failure_attempts,
            failure_fraction=failure_fraction,
            num_samples=sample_count,
            seed=int(seed) if seed is not None else 0,
        )
    valid_frames = int(valid.sum().item())
    stats = AttemptSamplingStats(
        attempt_field=attempt_field,
        valid_field=valid_field,
        outcome_known_field=outcome_known_field,
        distinct_attempts=len(attempts),
        success_attempts=len(success_attempts) if failure_fraction is not None else 0,
        failure_attempts=len(failure_attempts) if failure_fraction is not None else 0,
        target_failure_fraction=failure_fraction,
        valid_frames=valid_frames,
        terminal_frames_included=terminal_frames_included,
        excluded_frames=len(dataset) - valid_frames,
        num_samples=sample_count,
    )
    return sampler, stats
