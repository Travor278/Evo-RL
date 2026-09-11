"""Prepare, train and audit pure-demonstration SFT+RL one explicit stage at a time."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from lerobot.rl.sft_rl.protocol import (
    canonical_sha256,
    require_supported_split,
    select_high_segments,
    split_episodes,
    validate_demo_contract,
    validate_value_provenance,
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_value(contract_path, config_path, holdout_count, output):
    from lerobot.rl.sft_rl.provenance import source_fingerprint
    from lerobot.rl.sft_rl.replay import file_sha256, source_frame_index
    from lerobot.rl.sft_rl.value_training import open_source, validate_run_config, write_json

    contract, config = load(contract_path), load(config_path)
    episodes = validate_demo_contract(contract)
    validate_run_config(config)
    split = split_episodes(episodes, seed=config["seed"], holdout_count=holdout_count)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "split_report.json", split)
    require_supported_split(split)  # Leaves an explicit offending-episode report on failure.
    source = open_source(contract)
    source_frame_index(source, episodes)
    if set(config["model"]["camera_features"]) != set(source.meta.camera_keys):
        raise ValueError("Value cameras do not match the frozen source camera mapping")
    source_root = Path(contract["root"])
    sealed = [path for path in (source_root / "meta").rglob("*") if path.is_file()]
    sealed += list((source_root / "data").rglob("*.parquet"))
    source_hashes = {path.relative_to(source_root).as_posix(): file_sha256(path) for path in sorted(sealed)}
    inventory = {
        "repo_id": contract["repo_id"],
        "revision": contract["revision"],
        "metadata_and_parquet_sha256": source_hashes,
        "video_files": [
            {"path": path.relative_to(source_root).as_posix(), "bytes": path.stat().st_size}
            for path in sorted((source_root / "videos").rglob("*.mp4"))
        ],
        "allowed_episode_indices": [ep.episode_index for ep in episodes],
        "note": "Storage files may contain multiple episodes; only the explicit allowed frames enter training or selection.",
    }
    write_json(output / "source_inventory.json", inventory)
    provenance = {
        key: value for key, value in split.items() if key not in ("out_of_support", "out_of_support_rule")
    }
    provenance.update(
        source_kind="human_teleoperation",
        contract_sha256=canonical_sha256(contract),
        takeover_penalty=0,
        gamma=1,
        architecture="pure_vision_201_two_hot",
        initialization="pretrained_vision_random_head",
        value_config_sha256=canonical_sha256(config),
        source_code=source_fingerprint(),
        source_inventory_sha256=canonical_sha256(inventory),
    )
    validate_value_provenance(provenance, contract)
    prepared = {
        "schema": "sft-rl-prepared/v1",
        "contract": contract,
        "value_config": config,
        "value_provenance": provenance,
        "source_contract_file_sha256": file_sha256(contract_path),
        "source_inventory": inventory,
    }
    write_json(output / "prepared.json", prepared)
    return prepared


def select(prepared_path, predictions_path, output):
    from lerobot.rl.sft_rl.replay import file_sha256
    from lerobot.rl.sft_rl.value_training import write_json

    prepared, prediction = load(prepared_path), load(predictions_path)
    contract = prepared["contract"]
    validate_value_provenance(prediction["value_provenance"], contract)
    if prediction["value_provenance"] != prepared["value_provenance"] or prediction[
        "contract_sha256"
    ] != canonical_sha256(contract):
        raise ValueError("Predictions are from a different Value experiment or allowed data set")
    values = {int(ep): np.asarray(row) for ep, row in prediction["predictions"].items()}
    result = select_high_segments(
        validate_demo_contract(contract),
        values,
        z=prediction["value_provenance"]["Z"],
        value_checkpoint_sha256=prediction["value_checkpoint_sha256"],
        contract_sha256=canonical_sha256(contract),
    )
    result["value_provenance"] = prediction["value_provenance"]
    result["checkpoint_step"] = prediction["checkpoint_step"]
    result["predictions_sha256"] = file_sha256(predictions_path)
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def prepare_policy(args):
    from lerobot.rl.sft_rl.protocol import validate_selection
    from lerobot.rl.sft_rl.replay import file_sha256
    from lerobot.rl.sft_rl.value_training import write_json

    control = load(args.control)
    control_run = load(args.control_run)
    if control_run.get("initial_model_sha256") != args.sft_sha256:
        raise ValueError("Starting weights must match the counterpart's actual SFT initialization")
    if (
        control_run.get("steps") != control.get("steps")
        or control_run.get("seed") != control.get("seed")
        or control_run.get("global_batch_size") != args.global_batch
    ):
        raise ValueError("Control config and actual run provenance do not agree")
    for key in ("steps", "batch_size", "seed", "optimizer", "scheduler", "policy", "rename_map", "peft"):
        if key not in control:
            raise ValueError(
                f"Need the actual resolved control train_config, missing {key}; summary YAML is insufficient"
            )
    contract, selection = load(args.contract), load(args.selection)
    validate_selection(selection, contract)
    validate_value_provenance(selection["value_provenance"], contract)
    if args.global_batch % control["batch_size"]:
        raise ValueError("Control global batch is incompatible with its per-rank batch size")
    if not 0 < args.high_fraction < 1:
        raise ValueError("An explicit reviewed D_high source fraction is required")
    checkpoint = Path(args.sft_checkpoint).resolve()
    for file in ("model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (checkpoint / file).is_file():
            raise ValueError("Shared SFT checkpoint is incomplete: " + file)
    if file_sha256(checkpoint / "model.safetensors") != args.sft_sha256:
        raise ValueError("SFT checkpoint does not match the reviewed control identity")
    validation = load(args.export_validation)
    selection_sha = file_sha256(args.selection)
    if validation.get("status") != "ok" or validation.get("selection_sha256") != selection_sha:
        raise ValueError("Export alignment/batch validation has not passed for this selection")
    config = copy.deepcopy(control)
    config.pop("checkpoint_path", None)
    config.update(
        resume=False,
        output_dir=str(Path(args.output_dir).resolve()),
        job_name=Path(args.output_dir).name,
        acp={"enable": False},
        replay_sampling={"enable": False},
        use_rabc=False,
        use_policy_training_preset=False,
    )
    config["policy"]["pretrained_path"] = str(checkpoint)
    config["policy"]["push_to_hub"] = False
    config["dataset"].update(
        repo_id=contract["repo_id"],
        root=contract["root"],
        revision=contract["revision"],
        episodes=[ep.episode_index for ep in validate_demo_contract(contract)],
        streaming=False,
    )
    config["sft_rl"] = {
        "enable": True,
        "contract_path": str(Path(args.contract).resolve()),
        "selection_path": str(Path(args.selection).resolve()),
        "selection_sha256": selection_sha,
        "export_validation_path": str(Path(args.export_validation).resolve()),
        "high_fraction": args.high_fraction,
        "expected_global_batch": args.global_batch,
        "expected_output_horizon": config["policy"]["chunk_size"],
        "expected_mixed_precision": args.mixed_precision,
        "sft_checkpoint_sha256": args.sft_sha256,
        "control_config_path": str(Path(args.control).resolve()),
        "control_config_sha256": file_sha256(args.control),
        "control_run_path": str(Path(args.control_run).resolve()),
        "control_run_sha256": file_sha256(args.control_run),
        "normalization_source": args.normalization_source,
        "normalization_stats_sha256": args.stats_sha256,
    }
    from lerobot.rl.sft_rl.config import SFTReplayConfig

    SFTReplayConfig(**config["sft_rl"]).validate()
    if Path(args.config_output).exists():
        raise FileExistsError(args.config_output)
    write_json(args.config_output, config)
    world = args.global_batch // control["batch_size"]
    review = {
        "status": "configuration_for_review_not_started",
        "SFT_checkpoint": str(checkpoint),
        "SFT_sha256": args.sft_sha256,
        "data_repo": contract["repo_id"],
        "data_revision": contract["revision"],
        "Value_checkpoint_sha256": selection["value_checkpoint_sha256"],
        "Value_provenance": selection["value_provenance"],
        "Z": selection["Z"],
        "advantage_horizon": 50,
        "selection": "global Top10% then A>0",
        "retained_window": 20,
        "original_fraction": 1 - args.high_fraction,
        "high_fraction": args.high_fraction,
        "policy_updates": config["steps"],
        "global_batch": args.global_batch,
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "policy_scope_and_horizon": config["policy"],
        "peft": config["peft"],
        "normalization_source": args.normalization_source,
        "output_dir": config["output_dir"],
        "log_freq": config.get("log_freq"),
        "save_freq": config.get("save_freq"),
        "save_steps": config.get("save_steps"),
        "command": [
            "python",
            "-m",
            "accelerate.commands.launch",
            *(["--multi_gpu"] if world > 1 else []),
            "--num_processes",
            str(world),
            "--num_machines",
            "1",
            "--mixed_precision",
            args.mixed_precision,
            "--module",
            "lerobot.scripts.lerobot_train",
            "--config_path",
            str(Path(args.config_output).resolve()),
        ],
    }
    write_json(Path(args.config_output).with_suffix(".review.json"), review)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    contract_builder = commands.add_parser("build-contract")
    for key in ("scope", "final-policy-test", "output"):
        contract_builder.add_argument("--" + key, required=True)
    prepare = commands.add_parser("prepare-value")
    for key in ("contract", "value-config", "output"):
        prepare.add_argument("--" + key, required=True)
    prepare.add_argument("--holdout-count", type=int, required=True)
    train = commands.add_parser("train-value")
    for key in ("prepared", "output"):
        train.add_argument("--" + key, required=True)
    train.add_argument("--resume")
    infer = commands.add_parser("infer-value")
    for key in ("prepared", "checkpoint", "output"):
        infer.add_argument("--" + key, required=True)
    infer.add_argument("--device", required=True, choices=["cpu", "cuda"])
    infer.add_argument("--batch-size", type=int, required=True)
    infer.add_argument("--workers", type=int, default=0)
    selection = commands.add_parser("select")
    for key in ("prepared", "predictions", "output"):
        selection.add_argument("--" + key, required=True)
    export = commands.add_parser("export")
    for key in ("contract", "selection", "output", "repo-id"):
        export.add_argument("--" + key, required=True)
    export.add_argument("--output-horizon", type=int, required=True)
    export.add_argument("--pixel-tolerance", type=float, default=0.08)
    export.add_argument("--preview-count", type=int, default=3)
    policy = commands.add_parser("prepare-policy")
    for key in (
        "control",
        "control-run",
        "contract",
        "selection",
        "export-validation",
        "sft-checkpoint",
        "sft-sha256",
        "output-dir",
        "config-output",
    ):
        policy.add_argument("--" + key, required=True)
    policy.add_argument("--high-fraction", type=float, required=True)
    policy.add_argument("--global-batch", type=int, required=True)
    policy.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], required=True)
    policy.add_argument("--normalization-source", choices=["checkpoint", "control_dataset"], required=True)
    policy.add_argument("--stats-sha256")
    reports = commands.add_parser("report")
    for key in ("metrics", "predictions", "selection", "output"):
        reports.add_argument("--" + key, required=True)
    robot = commands.add_parser("robot-report")
    for key in ("protocol", "trials", "policy-checkpoint", "output"):
        robot.add_argument("--" + key, required=True)
    verification = commands.add_parser("verify-policy")
    for key in ("checkpoint", "contract", "output", "device"):
        verification.add_argument("--" + key, required=True)
    verification.add_argument("--batch-size", type=int, required=True)
    verification.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "build-contract":
        from lerobot.rl.sft_rl.manifest import build_contract

        result = build_contract(args.scope, args.final_policy_test, args.output)
    elif args.command == "prepare-value":
        result = prepare_value(args.contract, args.value_config, args.holdout_count, args.output)
    elif args.command == "train-value":
        from lerobot.rl.sft_rl.value_training import train_value

        train_value(args.prepared, args.output, resume=args.resume)
        return
    elif args.command == "infer-value":
        from lerobot.rl.sft_rl.value_training import infer_value

        infer_value(
            args.prepared,
            args.checkpoint,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        return
    elif args.command == "select":
        result = select(args.prepared, args.predictions, args.output)
    elif args.command == "export":
        from lerobot.rl.sft_rl.export import export_segments

        result = export_segments(
            load(args.contract),
            load(args.selection),
            args.selection,
            args.output,
            repo_id=args.repo_id,
            output_horizon=args.output_horizon,
            pixel_tolerance=args.pixel_tolerance,
            preview_count=args.preview_count,
        )
    elif args.command == "report":
        from lerobot.rl.sft_rl.reporting import plot_reports

        result = plot_reports(args.metrics, args.predictions, args.selection, args.output)
    elif args.command == "robot-report":
        from lerobot.rl.sft_rl.replay import file_sha256
        from lerobot.rl.sft_rl.reporting import aggregate_robot_trials
        from lerobot.rl.sft_rl.value_training import write_json

        checkpoint = Path(args.policy_checkpoint)
        run = load(checkpoint / "sft_rl_protocol.json")
        if run.get("schema") != "sft-rl-policy/v1":
            raise ValueError("Not an SFT+RL policy checkpoint")
        result = aggregate_robot_trials(
            load(args.protocol),
            load(args.trials),
            checkpoint_sha256=file_sha256(checkpoint / "model.safetensors"),
            policy_run_protocol_sha256=canonical_sha256(run),
        )
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    elif args.command == "verify-policy":
        from lerobot.rl.sft_rl.policy_validation import verify_policy

        result = verify_policy(
            args.checkpoint,
            args.contract,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        result = prepare_policy(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
