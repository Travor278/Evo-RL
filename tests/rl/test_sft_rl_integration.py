"""Synthetic local video/Value pipeline checks; these are not experiment results."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lerobot.rl.sft_rl.protocol import canonical_sha256, select_high_segments, validate_demo_contract
from lerobot.rl.sft_rl.replay import file_sha256
from lerobot.rl.sft_rl.value_training import infer_value, train_value
from lerobot.scripts.lerobot_sft_rl import prepare_value


class NativePipelineTests(unittest.TestCase):
    def make_fixture(self, directory):
        import av

        av.logging.set_level(av.logging.ERROR)
        from transformers import SiglipVisionConfig, SiglipVisionModel

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(directory)
        source = root / "source"
        camera = "observation.images.camera"
        features = {
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            camera: {"dtype": "video", "shape": (3, 16, 16), "names": ["channels", "height", "width"]},
        }
        dataset = LeRobotDataset.create(
            "local/synthetic-sft-rl",
            root=source,
            fps=10,
            features=features,
            robot_type="synthetic",
            video_backend="pyav",
            vcodec="h264",
        )
        for episode in range(2):
            for frame in range(6):
                image = np.full((3, 16, 16), 0.15 + 0.08 * frame + 0.1 * episode, dtype=np.float32)
                dataset.add_frame(
                    {
                        "action": np.array([episode, frame], dtype=np.float32),
                        "observation.state": np.array([frame, episode], dtype=np.float32),
                        camera: image,
                        "task": "insert",
                    }
                )
            dataset.save_episode(
                parallel_encoding=False, extra_episode_metadata={"episode_success": "success"}
            )
        dataset.finalize()
        contract = {
            "schema": "sft-rl-demo/v1",
            "repo_id": "local/synthetic-sft-rl",
            "revision": "a" * 40,
            "root": str(source),
            "task": "insert",
            "provenance_evidence": "synthetic integration test only",
            "final_policy_test_identities": ["separate-test"],
            "final_policy_test_manifest_sha256": "f" * 64,
            "episodes": [
                {
                    "episode_index": e,
                    "length": 6,
                    "source_identity": f"synthetic-{e}",
                    "task": "insert",
                    "source_kind": "human_teleoperation",
                    "complete_trajectory": True,
                    "successful": True,
                }
                for e in range(2)
            ],
        }
        (root / "contract.json").write_text(json.dumps(contract))
        backbone = root / "tiny-backbone"
        torch.manual_seed(17)
        SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=12,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=3,
                image_size=16,
                patch_size=4,
            )
        ).save_pretrained(backbone)
        training = {
            "steps": 4,
            "global_batch": 2,
            "eval_every": 2,
            "save_every": 2,
            "eval_frames_per_episode": 2,
            "lr": 0.001,
            "final_lr": 0.0001,
            "weight_decay": 0.00001,
            "grad_clip_norm": 1,
            "warmup_steps": 1,
            "checkpoint_selection": "final",
        }
        (root / "control.json").write_text(json.dumps({"seed": 1000, "training": training}))
        config = {
            "schema": "sft-rl-value-run/v1",
            "seed": 1000,
            "model": {
                "vision_repo_id": str(backbone),
                "vision_revision": "a" * 40,
                "camera_features": [camera],
                "image_size": 16,
                "projection_dim": 8,
                "dropout": 0.1,
                "gradient_checkpointing": False,
            },
            "training": training,
            "runtime": {"device": "cpu", "mixed_precision": "no", "num_workers": 0},
            "paired_control_config_path": str(root / "control.json"),
            "paired_control_config_sha256": file_sha256(root / "control.json"),
        }
        (root / "value.json").write_text(json.dumps(config))
        prepared = prepare_value(root / "contract.json", root / "value.json", 1, root / "prepared")
        return root, contract, prepared

    def test_reviewed_scope_expands_metadata_without_inventing_labels(self):
        from lerobot.rl.sft_rl.manifest import build_contract

        with tempfile.TemporaryDirectory() as directory:
            root, contract, _ = self.make_fixture(directory)
            scope = {
                "schema": "sft-rl-scope/v1",
                "repo_id": contract["repo_id"],
                "root": contract["root"],
                "revision": contract["revision"],
                "task": "insert",
                "pure_human_source_verified": True,
                "complete_source_trajectories_verified": True,
                "provenance_evidence": "synthetic test only",
                "allowed_episode_indices": [0, 1],
                "source_identity_namespace": "synthetic",
            }
            (root / "scope.json").write_text(json.dumps(scope))
            (root / "test.json").write_text(json.dumps({"source_identities": ["reserved"]}))
            result = build_contract(root / "scope.json", root / "test.json", root / "built.json")
            self.assertEqual([ep["length"] for ep in result["episodes"]], [6, 6])
            self.assertEqual(result["episodes"][0]["source_identity"], "synthetic:0")
            scope["pure_human_source_verified"] = False
            (root / "scope.json").write_text(json.dumps(scope))
            with self.assertRaises(ValueError):
                build_contract(root / "scope.json", root / "test.json", root / "invalid.json")
            self.assertFalse((root / "invalid.json").exists())

    def test_native_video_export_and_actual_chunk_batch(self):
        from lerobot.rl.sft_rl.export import export_segments

        with tempfile.TemporaryDirectory() as directory:
            root, contract, _ = self.make_fixture(directory)
            values = {0: np.array([0, 0, 0, 0, -0.9, 0]), 1: np.zeros(6)}
            selection = select_high_segments(
                validate_demo_contract(contract),
                values,
                z=5,
                value_checkpoint_sha256="c" * 64,
                contract_sha256=canonical_sha256(contract),
            )
            path = root / "selection.json"
            path.write_text(json.dumps(selection))
            report = export_segments(
                contract,
                selection,
                path,
                root / "export",
                repo_id="local/synthetic-export",
                output_horizon=50,
                preview_count=1,
            )
            self.assertEqual(report["frames_checked"], 2)
            self.assertTrue(report["action_state_exact"])
            self.assertEqual(report["real_policy_batch_shape"], [2, 50, 2])
            self.assertTrue(Path(report["previews"][0]["path"]).is_file())

    def test_value_checkpoint_resume_and_full_episode_inference(self):
        import lerobot.rl.sft_rl.value_training as training

        class StopAfterCheckpointError(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root, contract, prepared = self.make_fixture(directory)
            prepared_path = root / "prepared/prepared.json"
            with contextlib.redirect_stdout(io.StringIO()):
                train_value(prepared_path, root / "continuous")
            real_write = training.write_json

            def interrupt(path, value):
                real_write(path, value)
                if Path(path).name == "last.json" and value["step"] == 2:
                    raise StopAfterCheckpointError()

            with (
                self.assertRaises(StopAfterCheckpointError),
                patch.object(training, "write_json", interrupt),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                train_value(prepared_path, root / "resumed")
            with contextlib.redirect_stdout(io.StringIO()):
                train_value(prepared_path, root / "resumed", resume=root / "resumed/checkpoint-000002.pt")
            a = torch.load(root / "continuous/checkpoint-000004.pt", weights_only=True)
            b = torch.load(root / "resumed/checkpoint-000004.pt", weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            infer_value(
                prepared_path,
                root / "resumed/checkpoint-000004.pt",
                root / "prediction.json",
                device="cpu",
                batch_size=2,
            )
            result = json.loads((root / "prediction.json").read_text())
            self.assertEqual(set(result["predictions"]), {"0", "1"})
            self.assertTrue(all(len(values) == 6 for values in result["predictions"].values()))
            self.assertEqual(result["value_provenance"]["Z"], 5)
            metrics = [json.loads(line) for line in (root / "resumed/metrics.jsonl").read_text().splitlines()]
            self.assertIn("CE", metrics[-1]["evaluation"]["holdout"])
            self.assertIn("MAE", metrics[-1]["evaluation"]["train"])


if __name__ == "__main__":
    unittest.main()
