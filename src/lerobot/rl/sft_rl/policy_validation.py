"""Explicit checkpoint reload and offline inference, never robot actuation."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .protocol import canonical_sha256, validate_demo_contract
from .replay import file_sha256
from .value_training import open_source, write_json


def verify_policy(checkpoint, contract_path, output, *, device, batch_size, seed):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint = Path(checkpoint)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    validate_demo_contract(contract)
    protocol = json.loads((checkpoint / "sft_rl_protocol.json").read_text(encoding="utf-8"))
    config_saved = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != "sft-rl-policy/v1" or protocol[
        "contract_canonical_sha256"
    ] != canonical_sha256(contract):
        raise ValueError("Checkpoint is not from this pure-demo SFT+RL run")
    if not config_saved.get("sft_rl", {}).get("enable") or config_saved.get("acp", {}).get("enable"):
        raise ValueError("Cannot validate a baseline/BC/ACP checkpoint as the SFT+RL result")
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.pretrained_path = checkpoint
    policy = PI05Policy.from_pretrained(str(checkpoint), config=config, strict=True, local_files_only=True)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        config, pretrained_path=checkpoint, preprocessor_overrides={"device_processor": {"device": device}}
    )
    source = open_source(contract, action_horizon=config.chunk_size)
    # These are explicit training-source sanity inputs, not a final test metric.
    indices = torch.linspace(0, len(source) - 1, min(batch_size, len(source))).long().tolist()
    batch = next(iter(DataLoader(Subset(source, indices), batch_size=len(indices))))
    if any("Advantage: positive" in task or "Advantage: negative" in task for task in batch["task"]):
        raise ValueError("Inference must preserve normal task instructions")
    processed = pre(batch)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        loss, _ = policy.forward(processed)
        actions = policy.predict_action_chunk(processed)
        physical_actions = post(actions)
    if (
        actions.shape[1] != config.chunk_size
        or not torch.isfinite(actions).all()
        or not torch.isfinite(physical_actions).all()
        or not torch.isfinite(loss)
    ):
        raise ValueError("Checkpoint reload/inference shape or finiteness verification failed")
    if Path(output).exists():
        raise FileExistsError(output)
    report = {
        "status": "ok",
        "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
        "policy_protocol_sha256": canonical_sha256(protocol),
        "contract_sha256": canonical_sha256(contract),
        "action_shape": list(actions.shape),
        "physical_action_shape": list(physical_actions.shape),
        "seed": seed,
        "device": device,
        "torch_version": torch.__version__,
        "flow_loss_sanity_check": float(loss),
        "interpretation": "Offline reload/finiteness check only, not robot success or policy improvement.",
    }
    write_json(output, report)
    return report
