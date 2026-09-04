#!/usr/bin/env python3
"""Create a weight-identical PI0.5 config view loadable by Evo-RL commit 6f2db449."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


REMOVED_CONFIG_FIELDS = (
    "pretrained_revision",
    "use_relative_actions",
    "relative_exclude_joints",
    "action_feature_names",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_model.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite compatibility view: {destination}")
    destination.mkdir(parents=True)

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    removed = {key: config.pop(key) for key in REMOVED_CONFIG_FIELDS if key in config}
    if set(removed) != set(REMOVED_CONFIG_FIELDS):
        raise RuntimeError(f"Unexpected source config fields; removed={sorted(removed)}")
    if config.get("type") != "pi05" or config.get("chunk_size") != 50 or config.get("n_action_steps") != 50:
        raise RuntimeError("Unexpected PI0.5 source configuration")
    if config["input_features"]["observation.state"]["shape"] != [14]:
        raise RuntimeError("Source state dimension is not 14")
    if config["output_features"]["action"]["shape"] != [14]:
        raise RuntimeError("Source action dimension is not 14")

    (destination / "config.json").write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    for name in (
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        shutil.copy2(source / name, destination / name)
    os.symlink(source / "model.safetensors", destination / "model.safetensors")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "load the exact step-50000 weights with Evo-RL commit 6f2db449; optimizer will be reinitialized",
        "source_model": str(source),
        "source_training_checkpoint": str(args.source_checkpoint.resolve()),
        "source_model_sha256": sha256_file(source / "model.safetensors"),
        "compat_config_sha256": sha256_file(destination / "config.json"),
        "removed_config_fields": removed,
        "weight_mode": "absolute symlink; bytes are unchanged",
        "evorl_commit": "6f2db449a21e1bac750b996f2e27cac6739aa63f",
    }
    (destination / "COMPAT_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
