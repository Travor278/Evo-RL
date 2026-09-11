"""Configuration, processor, distributed sampler and evaluation boundaries."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.rl.sft_rl.config import SFTReplayConfig
from lerobot.rl.sft_rl.protocol import canonical_sha256
from lerobot.rl.sft_rl.replay import UpdateMixtureSampler, file_sha256, validate_control_alignment
from lerobot.rl.sft_rl.reporting import aggregate_robot_trials


class GuardTests(unittest.TestCase):
    def test_no_guessed_sampling_or_precision(self):
        with self.assertRaises(ValueError):
            SFTReplayConfig(enable=True).validate()

    def test_mask_survives_real_processor_roundtrip(self):
        from lerobot.processor.converters import batch_to_transition, transition_to_batch
        from lerobot.processor.device_processor import DeviceProcessorStep

        mask = torch.tensor([[False, False, True]])
        batch = {"action": torch.ones(1, 3, 2), "action_is_pad": mask, "task": ["insert"]}
        processed = transition_to_batch(
            DeviceProcessorStep(device="cpu", float_dtype="float32")(batch_to_transition(batch))
        )
        self.assertEqual(processed["action_is_pad"].dtype, torch.bool)
        torch.testing.assert_close(processed["action_is_pad"], mask)
        self.assertEqual(processed["task"], ["insert"])

    def test_accelerate_shards_global_stream_into_exact_updates(self):
        from accelerate.data_loader import prepare_data_loader

        dataset = torch.utils.data.TensorDataset(torch.arange(30))
        options = {"high_fraction": 0.6, "global_batch": 8, "seed": 18, "start_step": 0, "stop_step": 5}
        expected = list(UpdateMixtureSampler(20, 10, **options))
        ranks = []
        for rank in range(2):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, sampler=UpdateMixtureSampler(20, 10, **options), drop_last=True
            )
            prepared = prepare_data_loader(
                loader,
                device=torch.device("cpu"),
                num_processes=2,
                process_index=rank,
                split_batches=False,
                put_on_device=False,
                rng_types=[],
                even_batches=False,
            )
            ranks.append([batch[0].tolist() for batch in prepared])
        self.assertEqual([len(rows) for rows in ranks], [5, 5])
        assembled = [index for step in range(5) for rank in range(2) for index in ranks[rank][step]]
        self.assertEqual(assembled, expected)

    def test_actual_control_config_decodes_and_alignment_rejects_changes(self):
        import draccus

        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401

        path = (
            Path(__file__).parents[2] / "examples/pure_demo_sft_rl/control_evidence/bc20k_train_config.json"
        )
        control = json.loads(path.read_text())
        run_path = path.with_name("bc20k_run.json")
        run = json.loads(run_path.read_text())
        data = copy.deepcopy(control)
        data.pop("checkpoint_path", None)
        data["replay_sampling"] = {"enable": False}
        with tempfile.TemporaryDirectory() as directory:
            data["output_dir"] = str(Path(directory) / "new-run")
            data["use_policy_training_preset"] = False
            data["sft_rl"] = {
                "enable": True,
                "contract_path": "not-opened",
                "selection_path": "not-opened",
                "selection_sha256": "a" * 64,
                "export_validation_path": "not-opened",
                "high_fraction": 0.5,
                "expected_global_batch": 64,
                "expected_output_horizon": 50,
                "expected_mixed_precision": "bf16",
                "sft_checkpoint_sha256": run["initial_model_sha256"],
                "control_config_path": str(path),
                "control_config_sha256": file_sha256(path),
                "control_run_path": str(run_path),
                "control_run_sha256": file_sha256(run_path),
                "normalization_source": "checkpoint",
            }
            cfg = draccus.decode(TrainPipelineConfig, data)
            cfg.validate()
            accelerator = SimpleNamespace(
                num_processes=8, gradient_accumulation_steps=1, mixed_precision="bf16"
            )
            validate_control_alignment(cfg, accelerator)
            expected_sft = cfg.sft_rl.sft_checkpoint_sha256
            cfg.sft_rl.sft_checkpoint_sha256 = "b" * 64
            with self.assertRaises(ValueError):
                validate_control_alignment(cfg, accelerator)
            cfg.sft_rl.sft_checkpoint_sha256 = expected_sft
            cfg.steps += 1
            with self.assertRaises(ValueError):
                validate_control_alignment(cfg, accelerator)

    def test_robot_report_rejects_offline_results_and_wrong_checkpoint(self):
        protocol = {
            "task_setup": "synthetic unit-test fixture",
            "initialization": "fixed",
            "time_limit_seconds": 60,
            "camera_mapping": {"top": "top"},
            "RTC": {"enabled": False},
            "scoring_rule": "all_stages",
            "stage_names": ["insert"],
            "throughput": {"numerator": "successes", "denominator": "elapsed_seconds", "unit_seconds": 60},
        }
        kwargs = {"checkpoint_sha256": "c" * 64, "policy_run_protocol_sha256": "d" * 64}
        trial = {
            "trial_id": "synthetic-test-only",
            "protocol_sha256": canonical_sha256(protocol),
            "group": "SFT+RL",
            "evidence_kind": "offline",
            "checkpoint_sha256": "c" * 64,
            "policy_run_protocol_sha256": "d" * 64,
            "success": True,
            "elapsed_seconds": 30,
            "stages": {"insert": {"attempted": True, "success": True}},
        }
        with self.assertRaises(ValueError):
            aggregate_robot_trials(protocol, [trial], **kwargs)
        trial["evidence_kind"] = "real_robot"
        trial["checkpoint_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            aggregate_robot_trials(protocol, [trial], **kwargs)
        trial["checkpoint_sha256"] = "c" * 64
        result = aggregate_robot_trials(protocol, [trial], **kwargs)
        self.assertEqual(result["SR"], 1)
        self.assertEqual(result["TP"], 2)
        with self.assertRaises(ValueError):
            aggregate_robot_trials(protocol, [trial, trial], **kwargs)


if __name__ == "__main__":
    unittest.main()
