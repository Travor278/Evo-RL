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

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from lerobot.configs.train import ReplaySamplingConfig


@dataclass(frozen=True)
class ReplaySamplingStats:
    source_field: str
    base_count: int
    hil_count: int
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


def build_replay_sampler(
    dataset: Any,
    cfg: ReplaySamplingConfig,
    seed: int | None,
) -> tuple[WeightedRandomSampler | None, ReplaySamplingStats | None]:
    if not cfg.enable:
        return None, None

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
        source_field=cfg.source_field,
        base_count=base_count,
        hil_count=hil_count,
        natural_hil_fraction=hil_count / (base_count + hil_count),
        target_hil_fraction=target,
        num_samples=num_samples,
    )
    return sampler, stats
