#!/usr/bin/env python3
"""Create an isolated writable Evo-RL view without copying the 8 GB video payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if not all(inventory["gates"].values()):
        failed = [key for key, value in inventory["gates"].items() if not value]
        raise RuntimeError(f"Dataset audit has failed gates: {failed}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived view: {destination}")

    destination.mkdir(parents=True)
    for name in (".gitattributes", "README.md"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "data", destination / "data", copy_function=shutil.copy2)
    shutil.copytree(source / "meta", destination / "meta", copy_function=shutil.copy2)
    os.symlink(source / "videos", destination / "videos", target_is_directory=True)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_revision": inventory["revision"],
        "dataset_inventory": str(args.inventory.resolve()),
        "dataset_inventory_sha256": sha256_file(args.inventory.resolve()),
        "data_mode": "copied writable parquet",
        "meta_mode": "copied writable metadata",
        "videos_mode": "absolute symlink to immutable SSD source",
        "purpose": "official Evo-RL value/advantage/ACP inference",
    }
    (destination / "DERIVED_VIEW_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
