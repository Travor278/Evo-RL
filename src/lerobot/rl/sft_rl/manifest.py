"""Expand a reviewed data scope into complete episode provenance, without guessing outcomes."""

import json
from pathlib import Path

from .protocol import validate_demo_contract
from .replay import file_sha256
from .value_training import write_json


def build_contract(scope_path, final_test_path, output):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.utils.recording_annotations import normalize_episode_success_label

    scope = json.loads(Path(scope_path).read_text(encoding="utf-8"))
    test = json.loads(Path(final_test_path).read_text(encoding="utf-8"))
    if scope.get("schema") != "sft-rl-scope/v1":
        raise ValueError("Expected a reviewed sft-rl-scope/v1 file")
    for key in ("pure_human_source_verified", "complete_source_trajectories_verified"):
        if scope.get(key) is not True:
            raise ValueError(f"Data provenance must explicitly verify {key}")
    for key in ("repo_id", "revision", "root", "task", "provenance_evidence"):
        if not scope.get(key):
            raise ValueError(f"Pin data scope field: {key}")
    if not isinstance(test.get("source_identities"), list):
        raise ValueError("Final policy test must declare source identities")
    if not test["source_identities"] and test.get("no_offline_policy_test") is not True:
        raise ValueError("An empty final policy test list must be explicitly explained")
    if not (Path(scope["root"]) / "meta/info.json").is_file():
        raise FileNotFoundError("Source metadata must already exist locally")
    metadata = LeRobotDatasetMetadata(scope["repo_id"], root=Path(scope["root"]), revision=scope["revision"])
    records = {int(row["episode_index"]): row for row in metadata.episodes}
    ids = scope.get("allowed_episode_indices")
    if ids is None and scope.get("include_all_episodes") is True:
        ids = sorted(records)
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Declare the allowed full-source episodes explicitly")
    confirmations = scope.get("success_confirmations", {})
    episodes = []
    for episode in ids:
        row = records[episode]
        if row.get("sft_rl_crop"):
            raise ValueError("Cropped high segments cannot serve as complete Value trajectories")
        outcome = row.get(scope.get("success_field", "episode_success"))
        success = normalize_episode_success_label(outcome) == "success"
        if not success:
            confirmation = confirmations.get(str(episode))
            if (
                not confirmation
                or confirmation.get("successful") is not True
                or not confirmation.get("evidence")
            ):
                raise ValueError(
                    f"Episode {episode} has no verified success; supply explicit reviewed evidence"
                )
        identity_field = scope.get("source_identity_field")
        identity = row.get(identity_field) if identity_field else None
        if not identity:
            identity = f"{scope.get('source_identity_namespace') or scope['repo_id']}:{episode}"
        episodes.append(
            {
                "episode_index": int(episode),
                "length": int(row["length"]),
                "source_identity": str(identity),
                "task": scope["task"],
                "source_kind": "human_teleoperation",
                "complete_trajectory": True,
                "successful": True,
            }
        )
    contract = {
        "schema": "sft-rl-demo/v1",
        "repo_id": scope["repo_id"],
        "revision": scope["revision"],
        "root": scope["root"],
        "task": scope["task"],
        "provenance_evidence": scope["provenance_evidence"],
        "scope_sha256": file_sha256(scope_path),
        "final_policy_test_manifest_sha256": file_sha256(final_test_path),
        "final_policy_test_identities": test["source_identities"],
        "camera_features": list(metadata.camera_keys),
        "fps": metadata.fps,
        "episodes": episodes,
    }
    validate_demo_contract(contract)
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, contract)
    return contract
