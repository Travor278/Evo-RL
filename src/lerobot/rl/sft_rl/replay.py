"""Source-preserving D + D_high views and update-indexed explicit sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .protocol import canonical_sha256, validate_demo_contract, validate_selection, validate_value_provenance
from .provenance import runtime_versions, source_fingerprint


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frame_index(dataset, episodes):
    table = dataset.hf_dataset.with_format(None)
    ids = np.asarray(table["episode_index"], dtype=np.int64)
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    if len(ids) != len(dataset):
        raise ValueError("Source dataset index mapping does not match its frame table")
    wanted = {ep.episode_index: ep for ep in episodes}
    metadata = getattr(getattr(dataset, "meta", None), "episodes", None)
    if metadata is not None:
        for row in metadata:
            if int(row["episode_index"]) in wanted and row.get("sft_rl_crop"):
                raise ValueError("A cropped export cannot replace a complete source demonstration")
    selected = np.isin(ids, list(wanted))
    for key in ("complementary_info.is_intervention", "is_intervention", "intervention", "human_takeover"):
        if key in table.column_names:
            labels = np.asarray(table[key]).reshape(-1)
            if labels.shape != ids.shape or labels[selected].astype(bool).any():
                raise ValueError(f"Forbidden intervention frame detected: {key}")
    index = {}
    for ep in episodes:
        locations = np.flatnonzero(ids == ep.episode_index)
        if len(locations) != ep.length or not np.array_equal(
            np.sort(frames[locations]), np.arange(ep.length)
        ):
            raise ValueError(f"Incomplete/duplicate source frames for episode {ep.episode_index}")
        for row in locations:
            index[ep.episode_index, int(frames[row])] = int(row)
    return index


class DemoHighReplay(Dataset):
    """Read original videos; clip action queries at selected interval boundaries.

    All retained frames are eligible starts, not only the Top10% anchors.
    Dataset statistics remain those of D, never recomputed from duplicated D_high.
    """

    def __init__(self, source, contract, selection, output_horizon):
        episodes = validate_demo_contract(contract)
        validate_selection(selection, contract)
        if output_horizon < 1:
            raise ValueError("Invalid policy output horizon")
        self.source = source
        self.output_horizon = output_horizon
        index = source_frame_index(source, episodes)
        self.items = []
        for ep in sorted(episodes, key=lambda ep: ep.episode_index):
            self.items.extend(
                (index[ep.episode_index, t], ep.episode_index, t, ep.length, 0) for t in range(ep.length)
            )
        self.base_count = len(self.items)
        for row in selection["segments"]:
            ep, end = row["source_episode"], row["source_to"]
            self.items.extend((index[ep, t], ep, t, end, 1) for t in range(row["source_from"], end))
        self.high_count = len(self.items) - self.base_count
        if self.high_count == 0:
            raise ValueError(
                "No positive Top10% segments: report an empty selection; do not train a fake RL arm"
            )
        self.num_frames = len(self.items)
        self.num_episodes = len(episodes) + len(selection["segments"])

    def __getattr__(self, name):
        if name == "source":
            raise AttributeError(name)
        return getattr(self.source, name)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        source_index, ep, frame, end, pool = self.items[item]
        batch = dict(self.source[source_index])
        if any(tag in batch.get("task", "") for tag in ("Advantage: positive", "Advantage: negative")):
            raise ValueError("SFT+RL uses ordinary task instructions, never ACP-tagged prompts")
        actions = batch["action"]
        if actions.ndim != 2 or actions.shape[0] != self.output_horizon:
            raise ValueError("Source action query must use the unchanged policy output horizon")
        # The source loader may clamp at the true episode boundary; clamp earlier
        # for a high segment. Never fetch actions from a following segment.
        valid_length = min(end - frame, self.output_horizon)
        positions = torch.arange(self.output_horizon, device=actions.device)
        native = batch.get("action_is_pad")
        if native is None or native.dtype != torch.bool or native.shape != positions.shape:
            raise ValueError("Native dataset must supply a boolean action_is_pad for boundary verification")
        mask = native.to(actions.device) | (positions >= valid_length)
        if bool(mask[0]):
            raise ValueError("Source origin unexpectedly has no valid action")
        batch["action"] = actions[positions.clamp_max(valid_length - 1)].clone()
        batch["action_is_pad"] = mask
        batch["sft_rl_source"] = pool
        batch["sft_rl_source_episode"] = ep
        batch["sft_rl_source_frame"] = frame
        return batch


class UpdateMixtureSampler(Sampler):
    """Same global stream on all ranks; Accelerate shards per-rank batches.

    A fresh generator per optimizer update makes a resumed suffix identical
    without depending on worker prefetch or a separate generator checkpoint.
    """

    def __init__(self, base_count, high_count, *, high_fraction, global_batch, seed, start_step, stop_step):
        if min(base_count, high_count, global_batch) <= 0 or not 0 < high_fraction < 1:
            raise ValueError("Require nonempty D/D_high and an explicit source sampling probability")
        if not 0 <= start_step < stop_step:
            raise ValueError("Invalid remaining optimizer update budget")
        self.base_count, self.high_count = base_count, high_count
        self.high_fraction, self.global_batch = high_fraction, global_batch
        self.seed, self.start_step, self.stop_step = seed, start_step, stop_step

    def __len__(self):
        return (self.stop_step - self.start_step) * self.global_batch

    def __iter__(self):
        for step in range(self.start_step, self.stop_step):
            seed = int.from_bytes(
                hashlib.sha256(f"sft-rl-policy:{self.seed}:{step}".encode()).digest()[:8], "little"
            ) % (2**63)
            generator = torch.Generator().manual_seed(seed)
            high = torch.rand(self.global_batch, generator=generator) < self.high_fraction
            base_indices = torch.randint(self.base_count, (self.global_batch,), generator=generator)
            high_indices = self.base_count + torch.randint(
                self.high_count, (self.global_batch,), generator=generator
            )
            yield from torch.where(high, high_indices, base_indices).tolist()


def load_replay(source, cfg, train_cfg):
    cfg.validate()
    if file_sha256(cfg.control_config_path) != cfg.control_config_sha256:
        raise ValueError("Paired control configuration changed")
    contract = json.loads(Path(cfg.contract_path).read_text(encoding="utf-8"))
    selection = json.loads(Path(cfg.selection_path).read_text(encoding="utf-8"))
    if file_sha256(cfg.selection_path) != cfg.selection_sha256:
        raise ValueError("Selection changed after configuration review")
    export = json.loads(Path(cfg.export_validation_path).read_text(encoding="utf-8"))
    if (
        export.get("status") != "ok"
        or export.get("selection_sha256") != cfg.selection_sha256
        or export.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("Matching export/alignment/real-batch validation is required before policy training")
    for field in (
        "action_state_exact",
        "timestamps_checked",
        "episode_boundaries_checked",
        "all_encoded_frames_read",
    ):
        if export.get(field) is not True:
            raise ValueError(f"Incomplete export verification: {field}")
    if export.get("output_horizon") != cfg.expected_output_horizon:
        raise ValueError("Export validation used a different policy horizon")
    provenance = selection.get("value_provenance", {})
    validate_value_provenance(provenance, contract)
    if (
        provenance.get("contract_sha256") != canonical_sha256(contract)
        or provenance.get("source_kind") != "human_teleoperation"
    ):
        raise ValueError("Value provenance does not prove training on the allowed pure demonstrations")
    if (
        provenance.get("normalization") != "max_train_cumulative_cost"
        or provenance.get("normalization_multiplier") != 1
        or provenance.get("takeover_penalty") != 0
    ):
        raise ValueError("Mixed Value or incompatible return normalization cannot be reused")
    if provenance.get("Z") != selection["Z"] or provenance.get("architecture") != "pure_vision_201_two_hot":
        raise ValueError("Value architecture/normalization mismatch")
    if train_cfg.dataset.repo_id != contract["repo_id"] or train_cfg.dataset.revision != contract["revision"]:
        raise ValueError("Policy source dataset/revision differs from the frozen allowed demonstrations")
    if Path(train_cfg.dataset.root).resolve() != Path(contract["root"]).resolve():
        raise ValueError("Policy dataset root differs from the validated source")
    if cfg.normalization_source == "control_dataset":

        def plain(value):
            return (
                {k: plain(v) for k, v in value.items()}
                if isinstance(value, dict)
                else value.tolist()
                if hasattr(value, "tolist")
                else value
            )

        if canonical_sha256(plain(source.meta.stats)) != cfg.normalization_stats_sha256:
            raise ValueError("Source dataset statistics differ from the paired control normalization")
    if not train_cfg.resume:
        checkpoint = Path(train_cfg.policy.pretrained_path) / "model.safetensors"
        if file_sha256(checkpoint) != cfg.sft_checkpoint_sha256:
            raise ValueError("Policy must initialize from the shared SFT checkpoint, not base/BC/ACP weights")
    else:
        recorded = json.loads(
            (Path(train_cfg.output_dir) / "sft_rl_protocol.json").read_text(encoding="utf-8")
        )
        if recorded != replay_protocol(cfg, train_cfg):
            raise ValueError("Resume is only allowed for this same SFT+RL run, not an old BC/ACP checkpoint")
    return DemoHighReplay(source, contract, selection, cfg.expected_output_horizon)


def replay_protocol(cfg, train_cfg):
    selection = json.loads(Path(cfg.selection_path).read_text(encoding="utf-8"))
    contract = json.loads(Path(cfg.contract_path).read_text(encoding="utf-8"))
    return {
        "schema": "sft-rl-policy/v1",
        "contract_sha256": file_sha256(cfg.contract_path),
        "contract_canonical_sha256": canonical_sha256(contract),
        "value_checkpoint_sha256": selection["value_checkpoint_sha256"],
        "Z": selection["Z"],
        "selection_sha256": cfg.selection_sha256,
        "control_config_sha256": cfg.control_config_sha256,
        "control_run_sha256": cfg.control_run_sha256,
        "sft_checkpoint_sha256": cfg.sft_checkpoint_sha256,
        "high_fraction": cfg.high_fraction,
        "global_batch": cfg.expected_global_batch,
        "output_horizon": cfg.expected_output_horizon,
        "mixed_precision": cfg.expected_mixed_precision,
        "steps": train_cfg.steps,
        "seed": train_cfg.seed,
        "normalization_source": cfg.normalization_source,
        "normalization_stats_sha256": cfg.normalization_stats_sha256,
        "export_validation_sha256": file_sha256(cfg.export_validation_path),
        "source_code_sha256": source_fingerprint()["sha256"],
        "runtime_versions": runtime_versions(),
    }


def save_replay_protocol(cfg, train_cfg, destination=None):
    path = Path(destination or train_cfg.output_dir) / "sft_rl_protocol.json"
    value = replay_protocol(cfg, train_cfg)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Existing policy provenance differs; refusing to replace it")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    with path.with_name("source_code.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(source_fingerprint(), indent=2) + "\n")


def validate_control_alignment(cfg, accelerator):
    if file_sha256(cfg.sft_rl.control_config_path) != cfg.sft_rl.control_config_sha256:
        raise ValueError("Paired control configuration changed")
    control = json.loads(Path(cfg.sft_rl.control_config_path).read_text(encoding="utf-8"))
    if file_sha256(cfg.sft_rl.control_run_path) != cfg.sft_rl.control_run_sha256:
        raise ValueError("Counterpart run provenance changed")
    run = json.loads(Path(cfg.sft_rl.control_run_path).read_text(encoding="utf-8"))
    if run.get("initial_model_sha256") != cfg.sft_rl.sft_checkpoint_sha256:
        raise ValueError("Selected SFT checkpoint differs from the actual counterpart initialization")
    if (
        run.get("steps") != cfg.steps
        or run.get("seed") != cfg.seed
        or run.get("global_batch_size") != cfg.sft_rl.expected_global_batch
    ):
        raise ValueError("Update budget, seed or global batch differs from the actual counterpart run")
    actual = cfg.to_dict()
    for key in (
        "steps",
        "batch_size",
        "seed",
        "optimizer",
        "scheduler",
        "rename_map",
        "peft",
        "save_checkpoint",
        "save_freq",
        "save_steps",
        "log_freq",
    ):
        if key not in control or canonical_sha256(control[key]) != canonical_sha256(actual[key]):
            raise ValueError(f"Effective training configuration differs from the paired control: {key}")
    ignored = {"pretrained_path", "push_to_hub", "repo_id", "device"}
    expected_policy = {k: v for k, v in control["policy"].items() if k not in ignored}
    actual_policy = {k: v for k, v in actual["policy"].items() if k not in ignored}
    if canonical_sha256(expected_policy) != canonical_sha256(actual_policy):
        raise ValueError(
            "Policy architecture, horizon, precision or trainable scope differs from the paired control"
        )
    if cfg.batch_size * accelerator.num_processes != cfg.sft_rl.expected_global_batch:
        raise ValueError("Global batch differs from the reviewed paired control")
    if accelerator.gradient_accumulation_steps != 1:
        raise ValueError(
            "This native training loop counts one optimizer update per batch; accumulation is unsupported"
        )
    if accelerator.mixed_precision != cfg.sft_rl.expected_mixed_precision:
        raise ValueError("Accelerate precision differs from the reviewed paired control")
