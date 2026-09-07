#!/usr/bin/env python3
"""Atomically align derived parquet column order with the frozen LeRobot feature schema."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

VIEW = Path(os.environ["ATTEMPT_VALUE_VIEW"]).resolve()
OUTPUT = Path(os.environ["COLUMN_ORDER_AUDIT"])
def main() -> None:
    files = sorted((VIEW / "data").glob("chunk-*/file-*.parquet"))
    if len(files) != 482:
        raise ValueError(f"Expected 482 data files, found {len(files)}")
    info = json.loads((VIEW / "meta" / "info.json").read_text(encoding="utf-8"))
    feature_order = list(info["features"])
    changed = 0
    first_order = None
    for path in files:
        table = pq.read_table(path)
        desired = [name for name in feature_order if name in table.column_names]
        if set(desired) != set(table.column_names):
            missing_from_info = sorted(set(table.column_names) - set(desired))
            raise ValueError(f"{path.name} contains columns absent from info.json: {missing_from_info}")
        if first_order is None:
            first_order = desired
        if table.column_names == desired:
            continue
        table = table.select(desired)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        changed += 1
    result = {
        "status": "pass",
        "view": str(VIEW),
        "files": len(files),
        "files_reordered": changed,
        "canonical_column_count": len(first_order or []),
        "canonical_columns": first_order,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
