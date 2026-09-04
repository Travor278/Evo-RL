#!/usr/bin/env python3
"""Create an Evo-RL policy-training view of the exact full558 step-50000 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remove_disabled_step(payload: dict, registry_name: str) -> dict:
    matches = [step for step in payload["steps"] if step.get("registry_name") == registry_name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {registry_name!r} step, found {len(matches)}")
    step = matches[0]
    if step.get("config", {}).get("enabled") is not False:
        raise RuntimeError(f"Refusing to remove enabled processor step: {step}")
    payload["steps"] = [item for item in payload["steps"] if item is not step]
    return step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite compatibility view: {destination}")

    parent_provenance = read_json(source / "COMPAT_PROVENANCE.json")
    expected_model_sha = "d85c7cd84060a924b6ef10d055491c500c5714a21ddbb49e1dbdc828b7a74147"
    if sha256_file(source / "model.safetensors") != expected_model_sha:
        raise RuntimeError("Source model hash mismatch")

    preprocessor = read_json(source / "policy_preprocessor.json")
    postprocessor = read_json(source / "policy_postprocessor.json")
    removed_pre = remove_disabled_step(preprocessor, "relative_actions_processor")
    removed_post = remove_disabled_step(postprocessor, "absolute_actions_processor")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        shutil.copy2(source / "config.json", temp / "config.json")
        shutil.copy2(
            source / "policy_preprocessor_step_3_normalizer_processor.safetensors",
            temp / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        )
        shutil.copy2(
            source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            temp / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        write_json(temp / "policy_preprocessor.json", preprocessor)
        write_json(temp / "policy_postprocessor.json", postprocessor)
        os.symlink((source / "model.safetensors").resolve(strict=True), temp / "model.safetensors")

        provenance = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "policy-training compatibility view for Evo-RL commit 6f2db449",
            "parent_compat_view": str(source),
            "parent_compat_provenance_sha256": sha256_file(source / "COMPAT_PROVENANCE.json"),
            "source_training_checkpoint": parent_provenance["source_training_checkpoint"],
            "source_model": parent_provenance["source_model"],
            "source_model_sha256": expected_model_sha,
            "weight_mode": "absolute symlink; bytes unchanged",
            "normalizer_state_sha256": sha256_file(
                temp / "policy_preprocessor_step_3_normalizer_processor.safetensors"
            ),
            "unnormalizer_state_sha256": sha256_file(
                temp / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
            ),
            "removed_disabled_preprocessor_step": removed_pre,
            "removed_disabled_postprocessor_step": removed_post,
            "evorl_commit": "6f2db449a21e1bac750b996f2e27cac6739aa63f",
        }
        write_json(temp / "COMPAT_PROVENANCE.json", provenance)
        os.replace(temp, destination)
        print("PI05_POLICY_COMPAT_V2_PASS")
        print(json.dumps(provenance, indent=2, sort_keys=True))
    except BaseException:
        if temp.exists() and temp.parent == destination.parent:
            shutil.rmtree(temp)
        raise


if __name__ == "__main__":
    main()
