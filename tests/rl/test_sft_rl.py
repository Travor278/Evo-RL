"""CPU contract tests; no robot, network, pretrained weights or CUDA required."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.rl.sft_rl.loss import masked_action_losses
from lerobot.rl.sft_rl.protocol import (
    DemoEpisode,
    canonical_sha256,
    n_step_advantage,
    normalized_returns,
    require_supported_split,
    select_high_segments,
    split_episodes,
    validate_demo_contract,
    validate_selection,
)
from lerobot.rl.sft_rl.replay import DemoHighReplay, UpdateMixtureSampler
from lerobot.rl.sft_rl.value_model import (
    PureVisionValue,
    VisionValueConfig,
    two_hot_targets,
    value_expectation,
    value_loss,
)


def contract(length=51):
    return {
        "schema": "sft-rl-demo/v1",
        "repo_id": "test/demo",
        "revision": "a" * 40,
        "root": "/not-used",
        "provenance_evidence": "synthetic unit test, not robot evidence",
        "task": "insert",
        "final_policy_test_identities": ["test-held-out"],
        "final_policy_test_manifest_sha256": "f" * 64,
        "episodes": [
            {
                "episode_index": e,
                "length": length,
                "source_identity": f"demo-{e}",
                "task": "insert",
                "source_kind": "human_teleoperation",
                "complete_trajectory": True,
                "successful": True,
            }
            for e in [3, 8]
        ],
    }


def selection_for_replay(c):
    return {
        "schema": "sft-rl-selection/v1",
        "contract_sha256": canonical_sha256(c),
        "gamma": 1,
        "advantage_horizon": 50,
        "retained_action_window": 20,
        "global_top_fraction": 0.1,
        "positive_required": True,
        "ranking": "global_top_then_positive",
        "nonterminal_origins": sum(row["length"] - 1 for row in c["episodes"]),
        "requested_origins": int(np.ceil(0.1 * sum(row["length"] - 1 for row in c["episodes"]))),
        "selected_origins": 1,
        "retained_frames": 20,
        "segments": [
            {
                "source_episode": 3,
                "source_identity": "demo-3",
                "source_from": 0,
                "source_to": 20,
                "source_length": c["episodes"][0]["length"],
                "anchors": [{"frame": 0, "advantage": 0.1}],
                "is_success_terminal": False,
            }
        ],
    }


class ProtocolTests(unittest.TestCase):
    def test_pure_data_and_final_policy_test_exclusion(self):
        for key, value in [
            ("source_kind", "intervention"),
            ("complete_trajectory", False),
            ("successful", False),
        ]:
            c = contract()
            c["episodes"][0][key] = value
            with self.assertRaises(ValueError):
                validate_demo_contract(c)
        c = contract()
        c["final_policy_test_identities"].append("demo-3")
        with self.assertRaises(ValueError):
            validate_demo_contract(c)

    def test_train_only_z_no_factor_two_and_heldout_failure(self):
        eps = validate_demo_contract(contract())
        eps.append(DemoEpisode(9, 100, "long-demo", "insert", "human_teleoperation", True, True))
        for seed in range(30):
            split = split_episodes(eps, seed=seed, holdout_count=1)
            expected = max(ep.length - 1 for ep in eps if ep.episode_index in split["train_episode_indices"])
            self.assertEqual(split["Z"], expected)
            self.assertEqual(split, split_episodes(eps, seed=seed, holdout_count=1))
            if split["out_of_support"]:
                with self.assertRaises(ValueError):
                    require_supported_split(split)
                break
        else:
            self.fail("Longest trajectory never entered holdout")

    def test_returns_and_telescoping_advantage(self):
        values = normalized_returns(80, 79)
        self.assertEqual(values[0], -1)
        self.assertEqual(values[-1], 0)
        np.testing.assert_allclose(n_step_advantage(values, 79), 0, atol=1e-15)
        values[-1] = -0.8  # True terminal is forcibly zero at inference.
        np.testing.assert_allclose(n_step_advantage(values, 79), 0, atol=1e-15)
        with self.assertRaises(ValueError):
            normalized_returns(81, 79)
        with self.assertRaises(ValueError):
            n_step_advantage(values, 79, true_success_terminal=False)

    def test_global_top_then_positive_and_union(self):
        c = contract()
        eps = validate_demo_contract(c)
        baseline = -(50 - np.arange(51)) / 100
        scores = {
            3: np.r_[np.linspace(0.20, 0.29, 10), np.full(40, -0.001)],
            8: np.r_[np.linspace(0.05, 0.14, 10), np.full(40, -0.001)],
        }
        predictions = {ep: np.r_[baseline[:-1] - a, 0] for ep, a in scores.items()}
        result = select_high_segments(
            eps, predictions, z=100, value_checkpoint_sha256="c" * 64, contract_sha256=canonical_sha256(c)
        )
        self.assertEqual(result["selected_origins"], 10)  # Not top10% of the 20 positive origins.
        self.assertEqual(len(result["segments"]), 1)  # Not 5 origins per episode.
        self.assertEqual((result["segments"][0]["source_from"], result["segments"][0]["source_to"]), (0, 29))
        validate_selection(result, c)
        result["segments"][0]["source_to"] = 28
        with self.assertRaises(ValueError):
            validate_selection(result, c)

    def test_no_positive_and_truncated_terminal_window(self):
        c = contract()
        eps = validate_demo_contract(c)
        values = {ep.episode_index: np.zeros(ep.length) for ep in eps}
        empty = select_high_segments(
            eps, values, z=100, value_checkpoint_sha256="c" * 64, contract_sha256=canonical_sha256(c)
        )
        self.assertEqual(empty["selected_origins"], 0)
        values[3][-2] = -0.9
        selected = select_high_segments(
            eps, values, z=100, value_checkpoint_sha256="c" * 64, contract_sha256=canonical_sha256(c)
        )
        row = selected["segments"][0]
        self.assertEqual((row["source_from"], row["source_to"]), (49, 51))
        self.assertFalse(row["is_success_terminal"])


class FakeTable(dict):
    @property
    def column_names(self):
        return list(self)

    def with_format(self, _):
        return self


class FakeSource:
    def __init__(self, length=51, horizon=50):
        self.length, self.horizon = length, horizon
        self.hf_dataset = FakeTable(
            episode_index=np.repeat([3, 8], length), frame_index=np.tile(np.arange(length), 2)
        )
        self.meta = SimpleNamespace(stats={"unchanged": True})

    def __len__(self):
        return self.length * 2

    def __getitem__(self, row):
        episode, frame = int(self.hf_dataset["episode_index"][row]), int(self.hf_dataset["frame_index"][row])
        offsets = torch.arange(self.horizon)
        return {
            "action": (episode * 1000 + (frame + offsets).clamp_max(self.length - 1)).float()[:, None],
            "action_is_pad": frame + offsets >= self.length,
            "task": "insert",
        }


class ReplayTests(unittest.TestCase):
    def test_high_boundary_not_source_boundary(self):
        c = contract()
        source = FakeSource()
        replay = DemoHighReplay(source, c, selection_for_replay(c), 50)
        batch = replay[replay.base_count + 18]
        torch.testing.assert_close(batch["action"][:3, 0], torch.tensor([3018.0, 3019.0, 3019.0]))
        self.assertEqual(int((~batch["action_is_pad"]).sum()), 2)
        self.assertEqual(batch["action"].shape, (50, 1))  # Output horizon did not become 20.
        self.assertIs(replay.meta.stats, source.meta.stats)
        self.assertEqual(batch["task"], "insert")

    def test_source_guard(self):
        c = contract()
        source = FakeSource()
        source.hf_dataset["is_intervention"] = np.ones(len(source))
        with self.assertRaises(ValueError):
            DemoHighReplay(source, c, selection_for_replay(c), 50)
        source = FakeSource()
        source.meta.episodes = [{"episode_index": 3, "sft_rl_crop": True}]
        with self.assertRaises(ValueError):
            DemoHighReplay(source, c, selection_for_replay(c), 50)

    def test_sampler_probability_and_resume_suffix(self):
        kwargs = {"high_fraction": 0.7, "global_batch": 32, "seed": 1000, "stop_step": 1000}
        full = list(UpdateMixtureSampler(200, 10, start_step=0, **kwargs))
        resumed = list(UpdateMixtureSampler(200, 10, start_step=701, **kwargs))
        self.assertEqual(full[701 * 32 :], resumed)
        self.assertAlmostEqual(np.mean(np.asarray(full) >= 200), 0.7, delta=0.015)


class LossAndModelTests(unittest.TestCase):
    def test_padding_has_zero_gradient_and_per_sample_normalization(self):
        errors = torch.tensor([[[1.0], [900.0], [900.0]], [[4.0], [4.0], [4.0]]], requires_grad=True)
        mask = torch.tensor([[False, True, True], [False, False, False]])
        per_sample, _ = masked_action_losses(errors, mask)
        torch.testing.assert_close(per_sample, torch.tensor([1.0, 4.0]))
        per_sample.mean().backward()
        self.assertEqual(errors.grad[0, 1:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            masked_action_losses(errors, torch.ones_like(mask))

    def test_actual_pi05_forward_reads_mask(self):
        # Execute the actual method body with a tiny differentiable model; no VLM download.
        path = Path(__file__).parents[2] / "src/lerobot/policies/pi05/modeling_pi05.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI05Policy")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        namespace = {
            "Tensor": torch.Tensor,
            "ACTION": "action",
            "OBS_LANGUAGE_TOKENS": "tokens",
            "OBS_LANGUAGE_ATTENTION_MASK": "masks",
            "masked_action_losses": masked_action_losses,
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        prediction = torch.zeros((1, 50, 1), requires_grad=True)
        fake = SimpleNamespace(
            config=SimpleNamespace(output_features={"action": SimpleNamespace(shape=(1,))}),
            _preprocess_images=lambda batch: (None, None),
            prepare_action=lambda batch: batch["action"],
            model=SimpleNamespace(forward=lambda *args: (prediction - args[-1]).square()),
        )
        target = torch.ones_like(prediction)
        target[:, 20:] = 1000
        batch = {
            "action": target,
            "action_is_pad": torch.arange(50)[None, :] >= 20,
            "tokens": None,
            "masks": None,
        }
        loss, _ = namespace["forward"](fake, batch)
        self.assertEqual(loss.item(), 1)
        loss.backward()
        self.assertEqual(prediction.grad[:, 20:].abs().sum().item(), 0)

    def test_two_hot_expectation_and_tiny_value_update(self):
        y = torch.tensor([-1.0, -0.777, 0.0])
        distribution = two_hot_targets(y)
        torch.testing.assert_close(distribution.sum(-1), torch.ones(3))
        torch.testing.assert_close((distribution * torch.linspace(-1, 0, 201)).sum(-1), y)
        with self.assertRaises(ValueError):
            two_hot_targets(torch.tensor([-1.00001]))

        class Encoder(torch.nn.Module):
            config = SimpleNamespace(hidden_size=3)

            def forward(self, pixel_values, return_dict):
                return SimpleNamespace(pooler_output=pixel_values.mean((2, 3)))

        model = PureVisionValue(
            VisionValueConfig(
                "test",
                "a" * 40,
                ["camera"],
                image_size=4,
                projection_dim=8,
                dropout=0,
                gradient_checkpointing=False,
            ),
            Encoder(),
        )
        before = model.value_head[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        logits = model({"camera": torch.rand(3, 3, 4, 4)})
        value_loss(logits, y).mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.value_head[-1].weight))
        self.assertTrue(torch.isfinite(value_expectation(logits)).all())
        with self.assertRaises(ValueError):
            model({"camera": torch.rand(3, 3, 4, 4), "state": torch.zeros(3, 14)})


if __name__ == "__main__":
    unittest.main()
