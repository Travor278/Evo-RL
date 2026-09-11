"""Explicitly invoked Value training/inference; importing this module starts no jobs."""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .protocol import canonical_sha256, validate_demo_contract, validate_value_provenance
from .provenance import runtime_versions, source_fingerprint
from .replay import file_sha256
from .value_data import ValueFrames
from .value_model import PureVisionValue, VisionValueConfig, value_expectation, value_loss


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def validate_run_config(config):
    if config.get("schema") != "sft-rl-value-run/v1":
        raise ValueError("Expected sft-rl-value-run/v1")
    VisionValueConfig(**config["model"]).validate()
    training = config["training"]
    for key in ("steps", "global_batch", "eval_every", "save_every", "eval_frames_per_episode"):
        if not isinstance(training.get(key), int) or training[key] < 1:
            raise ValueError(f"Value training budget must explicitly specify {key}")
    for key in ("lr", "final_lr", "weight_decay", "grad_clip_norm"):
        if (
            not isinstance(training.get(key), (int, float))
            or not math.isfinite(training[key])
            or training[key] < 0
        ):
            raise ValueError(f"Invalid Value optimizer setting: {key}")
    if training["lr"] <= 0 or not 0 <= training["warmup_steps"] < training["steps"]:
        raise ValueError("Invalid learning-rate schedule")
    if training.get("checkpoint_selection") not in ("final", "minimum_holdout_mae"):
        raise ValueError("Freeze the Value checkpoint selection rule before training")
    if not isinstance(config.get("seed"), int):
        raise ValueError("Explicit Value seed required")
    runtime = config["runtime"]
    if (
        runtime["device"] not in ("cpu", "cuda")
        or runtime["mixed_precision"] not in ("no", "bf16")
        or runtime["num_workers"] < 0
    ):
        raise ValueError("Unsupported Value runtime")
    # The control is an explicit reviewed budget file, not a hard-coded default.
    control_path = config.get("paired_control_config_path")
    if not control_path or file_sha256(control_path) != config.get("paired_control_config_sha256"):
        raise ValueError("Pin the actual paired Value budget/configuration before launching")
    control = json.loads(Path(control_path).read_text(encoding="utf-8"))
    if control.get("training") != training or control.get("seed") != config["seed"]:
        raise ValueError("Value budget, optimizer or selection rule differs from the paired control")


def learning_rate(step, config):
    warmup, total = config["warmup_steps"], config["steps"]
    if warmup and step <= warmup:
        return config["lr"] * step / warmup
    fraction = (step - warmup) / (total - warmup)
    return config["final_lr"] + 0.5 * (config["lr"] - config["final_lr"]) * (1 + math.cos(math.pi * fraction))


def open_source(contract, *, action_horizon=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = validate_demo_contract(contract)
    if not (Path(contract["root"]) / "meta/info.json").is_file():
        raise FileNotFoundError(
            "The frozen source dataset must already exist locally; no implicit dataset download"
        )
    kwargs = {}
    if action_horizon is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        metadata = LeRobotDatasetMetadata(
            contract["repo_id"], root=Path(contract["root"]), revision=contract["revision"]
        )
        kwargs["delta_timestamps"] = {"action": [i / metadata.fps for i in range(action_horizon)]}
    return LeRobotDataset(
        contract["repo_id"],
        root=Path(contract["root"]),
        revision=contract["revision"],
        episodes=[ep.episode_index for ep in episodes],
        download_videos=False,
        video_backend="pyav",
        **kwargs,
    )


def autocast(device, precision):
    return torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def evaluate_value(model, dataset, *, device, batch_size, workers, precision, rank=0, world=1):
    model.eval()
    subset = Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            images = {key: image.to(device) for key, image in batch["images"].items()}
            y = batch["target"].to(device)
            with autocast(device, precision):
                logits = model(images)
            ce = value_loss(logits, y)
            error = (value_expectation(logits) - y).abs()
            sums += torch.stack(
                (
                    ce.double().sum(),
                    error.double().sum(),
                    torch.tensor(len(y), dtype=torch.float64, device=device),
                )
            )
    if world > 1:
        dist.all_reduce(sums)
    ce, mae, count = sums.cpu().tolist()
    if count == 0 or not all(math.isfinite(v) for v in (ce, mae)):
        raise ValueError("Empty or nonfinite Value evaluation")
    return {"CE": ce / count, "MAE": mae / count, "frames": int(count)}


def train_value(prepared_path, output, *, resume=None):
    prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    contract, config, provenance = (
        prepared["contract"],
        prepared["value_config"],
        prepared["value_provenance"],
    )
    validate_run_config(config)
    validate_value_provenance(provenance, contract)
    if provenance["value_config_sha256"] != canonical_sha256(config):
        raise ValueError("Prepared Value configuration changed")
    if provenance["source_code"]["sha256"] != source_fingerprint()["sha256"]:
        raise ValueError("Training source changed after preparation; review and prepare a new run")
    inventory = prepared["source_inventory"]
    if canonical_sha256(inventory) != provenance["source_inventory_sha256"]:
        raise ValueError("Prepared source inventory changed")
    for name, expected in inventory["metadata_and_parquet_sha256"].items():
        if file_sha256(Path(contract["root"]) / name) != expected:
            raise ValueError(f"Source metadata or action/state data changed: {name}")
    for row in inventory["video_files"]:
        if (Path(contract["root"]) / row["path"]).stat().st_size != row["bytes"]:
            raise ValueError(f"Source video size changed: {row['path']}")
    rank, world, local = [
        int(os.environ.get(k, default))
        for k, default in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]
    ]
    runtime, training = config["runtime"], config["training"]
    device = torch.device(runtime["device"], local) if runtime["device"] == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local)
    if training["global_batch"] % world:
        raise ValueError("Value global batch must divide the actual world size exactly")
    batch_size = training["global_batch"] // world
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    output = Path(output)
    if rank == 0:
        if resume is None:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "protocol.json", prepared)
        elif json.loads((output / "protocol.json").read_text(encoding="utf-8")) != prepared:
            raise ValueError("Resume protocol changed; do not overwrite the original experiment")
    if world > 1:
        dist.barrier()
    torch.manual_seed(config["seed"])
    model = PureVisionValue.from_pretrained_vision(VisionValueConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"]
    )
    start, best = 0, float("inf")
    if resume:
        last = json.loads((output / "last.json").read_text(encoding="utf-8"))
        if (
            Path(resume).resolve() != (output / last["file"]).resolve()
            or file_sha256(resume) != last["sha256"]
        ):
            raise ValueError(
                "Resume must use the last verified checkpoint, not rewind or overwrite later results"
            )
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["provenance"] != provenance or saved["world_size"] != world:
            raise ValueError("Resume checkpoint provenance/world size differs")
        if saved.get("runtime_versions") != runtime_versions():
            raise ValueError("Numerical stack changed during Value continuation")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["step"], saved["best_holdout_mae"]
        torch.set_rng_state(saved["rng"][rank]["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["rng"][rank]["cuda"], device)
    if not start < training["steps"]:
        raise ValueError("Value run already reached its fixed update budget")
    source = open_source(contract)
    cameras = config["model"]["camera_features"]
    train = ValueFrames(source, contract, provenance["train_episode_indices"], provenance["Z"], cameras)
    train_eval = ValueFrames(
        source,
        contract,
        provenance["train_episode_indices"],
        provenance["Z"],
        cameras,
        evaluation_frames=training["eval_frames_per_episode"],
    )
    holdout = ValueFrames(
        source,
        contract,
        provenance["holdout_episode_indices"],
        provenance["Z"],
        cameras,
        evaluation_frames=training["eval_frames_per_episode"],
    )

    class Batches:
        def __len__(self):
            return training["steps"] - start

        def __iter__(self):
            for step in range(start, training["steps"]):
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], step]))
                batch = rng.integers(len(train), size=training["global_batch"])
                yield batch[rank * batch_size : (rank + 1) * batch_size].tolist()

    loader = DataLoader(
        train,
        batch_sampler=Batches(),
        num_workers=runtime["num_workers"],
        generator=torch.Generator().manual_seed(config["seed"] + 1),
    )
    wrapped = (
        torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None
        )
        if world > 1
        else model
    )
    for step, batch in enumerate(loader, start + 1):
        model.train()
        rate = learning_rate(step, training)
        for group in optimizer.param_groups:
            group["lr"] = rate
        images = {key: image.to(device) for key, image in batch["images"].items()}
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, runtime["mixed_precision"]):
            logits = wrapped(images)
        loss = value_loss(logits, target).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Value loss")
        loss.backward()
        limit = training["grad_clip_norm"] if training["grad_clip_norm"] > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
        optimizer.step()
        metrics = None
        improved = False
        if step % training["eval_every"] == 0 or step == training["steps"]:
            kwargs = {
                "device": device,
                "batch_size": batch_size,
                "workers": runtime["num_workers"],
                "precision": runtime["mixed_precision"],
                "rank": rank,
                "world": world,
            }
            metrics = {
                "step": step,
                "train": evaluate_value(model, train_eval, **kwargs),
                "holdout": evaluate_value(model, holdout, **kwargs),
            }
            improved = metrics["holdout"]["MAE"] < best
            best = min(best, metrics["holdout"]["MAE"])
        record = {
            "step": step,
            "loss_rank0": float(loss.detach()),
            "lr": rate,
            "grad_norm": float(norm),
            "evaluation": metrics,
        }
        if rank == 0:
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
        should_save = (
            step % training["save_every"] == 0
            or step == training["steps"]
            or (improved and training["checkpoint_selection"] == "minimum_holdout_mae")
        )
        if should_save:
            rng = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            }
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states[0] = rng
            if rank == 0:
                name = f"checkpoint-{step:06d}.pt"
                temp = output / (name + ".tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "rng": states,
                        "world_size": world,
                        "best_holdout_mae": best,
                        "provenance": provenance,
                        "model_config": config["model"],
                        "runtime_versions": runtime_versions(),
                    },
                    temp,
                )
                os.replace(temp, output / name)
                info = {"step": step, "file": name, "sha256": file_sha256(output / name)}
                write_json(output / "last.json", info)
                if (
                    training["checkpoint_selection"] == "final"
                    and step == training["steps"]
                    or training["checkpoint_selection"] == "minimum_holdout_mae"
                    and improved
                ):
                    write_json(output / "selected.json", info)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def infer_value(prepared_path, checkpoint, output, *, device="cuda", batch_size=8, workers=0):
    prepared = json.loads(Path(prepared_path).read_text(encoding="utf-8"))
    contract = prepared["contract"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_value_provenance(saved["provenance"], contract)
    if saved["provenance"] != prepared["value_provenance"]:
        raise ValueError("Checkpoint does not match the frozen Value experiment")
    model = PureVisionValue.from_pretrained_vision(VisionValueConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    device = torch.device(device)
    model.to(device).eval()
    episodes = validate_demo_contract(contract)
    source = open_source(contract)
    data = ValueFrames(
        source,
        contract,
        [ep.episode_index for ep in episodes],
        saved["provenance"]["Z"],
        saved["model_config"]["camera_features"],
    )
    predictions = {ep.episode_index: np.full(ep.length, np.nan) for ep in episodes}
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=False):
            logits = model({key: image.to(device) for key, image in batch["images"].items()})
            values = value_expectation(logits).cpu().tolist()
            for ep, frame, value in zip(
                batch["episode"].tolist(), batch["frame"].tolist(), values, strict=True
            ):
                predictions[ep][frame] = value
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Incomplete full-source Value predictions")
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    write_json(
        target,
        {
            "schema": "sft-rl-predictions/v1",
            "contract_sha256": canonical_sha256(contract),
            "value_checkpoint_sha256": file_sha256(checkpoint),
            "value_provenance": saved["provenance"],
            "checkpoint_step": saved["step"],
            "predictions": {str(ep): values.tolist() for ep, values in predictions.items()},
        },
    )
