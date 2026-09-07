#!/usr/bin/env python

import pandas as pd

from lerobot.datasets.utils import load_tasks


def test_load_tasks_normalizes_explicit_v3_task_column(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    task = "Insert the copper screw into the black sleeve."
    pd.DataFrame({"task_index": [0], "task": [task]}).to_parquet(meta / "tasks.parquet")

    loaded = load_tasks(tmp_path)

    assert loaded.index.tolist() == [task]
    assert loaded.task_index.tolist() == [0]
    assert "task" not in loaded.columns


def test_load_tasks_preserves_legacy_text_index(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    task = "legacy task"
    pd.DataFrame({"task_index": [0]}, index=[task]).to_parquet(meta / "tasks.parquet")

    loaded = load_tasks(tmp_path)

    assert loaded.index.tolist() == [task]
    assert loaded.task_index.tolist() == [0]
