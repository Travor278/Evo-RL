#!/usr/bin/env python

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa


SCRIPT = (
    Path(__file__).parents[2]
    / "examples"
    / "piperx_attempt_aware_acp_v2"
    / "scripts"
    / "prepare_attempt_mixed_replay_view.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_attempt_mixed_replay_view", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ATTEMPT_ID = MODULE.ATTEMPT_ID
INTERVENTION = MODULE.INTERVENTION
VALID_CHUNK = MODULE.VALID_CHUNK
policy_columns = MODULE.policy_columns


def _base_table(rows=55):
    return pa.table({"frame_index": pa.array(np.arange(rows), type=pa.int64())})


def test_base_policy_columns_are_untagged_and_use_legal_chunk_starts():
    columns = policy_columns(
        _base_table(),
        source_kind="base",
        source_episode=3,
        acp_field="acp",
        apply_mask_field="mask",
        replay_source_field="source",
    )
    assert columns["source"].to_pylist() == [0] * 55
    assert columns["acp"].to_pylist() == [0] * 55
    assert columns["mask"].to_pylist() == [0] * 55
    assert columns[ATTEMPT_ID].to_pylist() == ["base:3"] * 55
    assert columns[VALID_CHUNK].to_pylist() == [True] * 6 + [False] * 49
    assert columns[INTERVENTION].to_pylist() == [False] * 55


def test_attempt_policy_columns_preserve_labels_boundary_and_intervention():
    table = pa.table(
        {
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "acp": pa.array([1, 0, 0], type=pa.int64()),
            "mask": pa.array([1, 1, 0], type=pa.int64()),
            ATTEMPT_ID: pa.array(["attempt-a"] * 3),
            VALID_CHUNK: pa.array([True, False, False]),
            INTERVENTION: pa.array([False, True, False]),
        }
    )
    columns = policy_columns(
        table,
        source_kind="attempt",
        source_episode=9,
        acp_field="acp",
        apply_mask_field="mask",
        replay_source_field="source",
    )
    assert columns["source"].to_pylist() == [1, 1, 1]
    assert columns["acp"].to_pylist() == [1, 0, 0]
    assert columns["mask"].to_pylist() == [1, 1, 0]
    assert columns[ATTEMPT_ID].to_pylist() == ["attempt-a"] * 3
    assert columns[VALID_CHUNK].to_pylist() == [True, False, False]
    assert columns[INTERVENTION].to_pylist() == [False, True, False]
