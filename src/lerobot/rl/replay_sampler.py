#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from lerobot.configs.train import ReplaySamplingConfig


@dataclass(frozen=True)
class ReplaySamplingStats:
    strategy: str
    source_field: str
    base_count: int
    hil_count: int
    base_valid_count: int
    hil_valid_count: int
    distinct_hil_attempts: int
    natural_hil_fraction: float
    target_hil_fraction: float
    num_samples: int


def _source_tensor(dataset: Any, source_field: str) -> torch.Tensor:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise TypeError("Replay sampling requires a LeRobotDataset with an hf_dataset.")
    if source_field not in hf_dataset.column_names:
        raise KeyError(f"Replay source field '{source_field}' is missing from dataset.")

    raw_values = hf_dataset.with_format(None)[source_field]
    values = torch.as_tensor(raw_values, dtype=torch.int64).reshape(-1)
    if values.numel() != len(dataset):
        raise ValueError(
            f"Replay source length mismatch: field has {values.numel()} values, dataset has {len(dataset)}."
        )
    return values


def _column(dataset: Any, field: str) -> list[Any]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise TypeError("Replay sampling requires a LeRobotDataset with an hf_dataset.")
    if field not in hf_dataset.column_names:
        raise KeyError(f"Replay sampling field '{field}' is missing from dataset.")
    values = list(hf_dataset.with_format(None)[field])
    if len(values) != len(dataset):
        raise ValueError(f"Replay field '{field}' has {len(values)} values, dataset has {len(dataset)}.")
    return values


class AttemptBalancedReplaySampler(Sampler[int]):
    """Two-level replay: choose source, then choose an attempt or legal base chunk.

    HIL sampling is attempt-uniform, followed by uniform sampling of a legal
    action-chunk start within that attempt. Base sampling remains uniform over
    legal baseline chunk starts. Sampling is with replacement and deterministic
    for a fixed seed.
    """

    def __init__(
        self,
        *,
        base_indices: torch.Tensor,
        hil_attempt_indices: list[torch.Tensor],
        target_hil_fraction: float,
        num_samples: int,
        seed: int,
    ) -> None:
        if base_indices.numel() == 0:
            raise ValueError("Attempt-balanced replay requires at least one legal Base chunk.")
        if not hil_attempt_indices or any(indices.numel() == 0 for indices in hil_attempt_indices):
            raise ValueError("Attempt-balanced replay requires at least one non-empty HIL attempt.")
        self.base_indices = base_indices.to(dtype=torch.int64, device="cpu")
        self.hil_attempt_indices = [indices.to(dtype=torch.int64, device="cpu") for indices in hil_attempt_indices]
        self.target_hil_fraction = float(target_hil_fraction)
        self.num_samples = int(num_samples)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            choose_hil = bool(torch.rand((), generator=self.generator).item() < self.target_hil_fraction)
            if choose_hil:
                attempt = int(
                    torch.randint(len(self.hil_attempt_indices), (), generator=self.generator).item()
                )
                candidates = self.hil_attempt_indices[attempt]
            else:
                candidates = self.base_indices
            offset = int(torch.randint(candidates.numel(), (), generator=self.generator).item())
            yield int(candidates[offset].item())


def _build_attempt_balanced_sampler(
    dataset: Any,
    cfg: ReplaySamplingConfig,
    seed: int | None,
) -> tuple[AttemptBalancedReplaySampler, ReplaySamplingStats]:
    source = _source_tensor(dataset, cfg.source_field)
    raw_valid = _column(dataset, cfg.valid_chunk_field)
    valid = torch.as_tensor(raw_valid, dtype=torch.bool).reshape(-1)
    attempt_ids = _column(dataset, cfg.attempt_id_field)

    is_hil = source == cfg.hil_value
    is_base = source == 0
    unexpected = ~(is_hil | is_base)
    if unexpected.any():
        bad = int(source[unexpected][0].item())
        raise ValueError(f"Replay source field must contain only base=0 or hil={cfg.hil_value}, got {bad}.")

    base_indices = torch.where(is_base & valid)[0]
    hil_valid_indices = torch.where(is_hil & valid)[0]
    grouped: dict[str, list[int]] = {}
    for index in hil_valid_indices.tolist():
        attempt_id = str(attempt_ids[index])
        if not attempt_id:
            raise ValueError(f"Empty attempt id at HIL dataset index {index}.")
        grouped.setdefault(attempt_id, []).append(index)
    hil_attempt_indices = [torch.tensor(indices, dtype=torch.int64) for _, indices in sorted(grouped.items())]
    num_samples = int(cfg.num_samples) if cfg.num_samples is not None else len(dataset)
    sampler = AttemptBalancedReplaySampler(
        base_indices=base_indices,
        hil_attempt_indices=hil_attempt_indices,
        target_hil_fraction=cfg.target_hil_fraction,
        num_samples=num_samples,
        seed=int(seed) if seed is not None else 0,
    )
    hil_count = int(is_hil.sum().item())
    base_count = int(is_base.sum().item())
    stats = ReplaySamplingStats(
        strategy="attempt_balanced",
        source_field=cfg.source_field,
        base_count=base_count,
        hil_count=hil_count,
        base_valid_count=int(base_indices.numel()),
        hil_valid_count=int(hil_valid_indices.numel()),
        distinct_hil_attempts=len(hil_attempt_indices),
        natural_hil_fraction=hil_count / (base_count + hil_count),
        target_hil_fraction=float(cfg.target_hil_fraction),
        num_samples=num_samples,
    )
    return sampler, stats


def build_replay_sampler(
    dataset: Any,
    cfg: ReplaySamplingConfig,
    seed: int | None,
) -> tuple[Sampler[int] | None, ReplaySamplingStats | None]:
    if not cfg.enable:
        return None, None

    if cfg.strategy == "attempt_balanced":
        return _build_attempt_balanced_sampler(dataset, cfg, seed)

    source = _source_tensor(dataset, cfg.source_field)
    is_hil = source == cfg.hil_value
    unexpected = ~(is_hil | (source == 0))
    if unexpected.any():
        bad = int(source[unexpected][0].item())
        raise ValueError(
            f"Replay source field must contain only base=0 or hil={cfg.hil_value}, got {bad}."
        )

    hil_count = int(is_hil.sum().item())
    base_count = int((~is_hil).sum().item())
    if hil_count == 0 or base_count == 0:
        raise ValueError(f"Replay sampling needs both sources, got base={base_count}, hil={hil_count}.")

    target = float(cfg.target_hil_fraction)
    weights = torch.empty(len(dataset), dtype=torch.double)
    weights[is_hil] = target / hil_count
    weights[~is_hil] = (1.0 - target) / base_count
    num_samples = int(cfg.num_samples) if cfg.num_samples is not None else len(dataset)
    generator = torch.Generator()
    generator.manual_seed(int(seed) if seed is not None else 0)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    stats = ReplaySamplingStats(
        strategy="source_weighted",
        source_field=cfg.source_field,
        base_count=base_count,
        hil_count=hil_count,
        base_valid_count=base_count,
        hil_valid_count=hil_count,
        distinct_hil_attempts=0,
        natural_hil_fraction=hil_count / (base_count + hil_count),
        target_hil_fraction=target,
        num_samples=num_samples,
    )
    return sampler, stats
