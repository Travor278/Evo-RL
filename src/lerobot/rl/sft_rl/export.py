"""Materialize audited high intervals without relabeling crop ends as successes."""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .protocol import canonical_sha256, validate_demo_contract, validate_selection
from .replay import DemoHighReplay, file_sha256, source_frame_index
from .value_training import open_source, write_json


def export_segments(
    contract,
    selection,
    selection_path,
    output,
    *,
    repo_id,
    output_horizon,
    pixel_tolerance=0.08,
    preview_count=3,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = validate_demo_contract(contract)
    validate_selection(selection, contract)
    if not selection["segments"]:
        raise ValueError("No positive high segments; preserve the empty-selection report and do not train")
    if not 0 <= pixel_tolerance <= 1 or output_horizon < 1:
        raise ValueError("Invalid export verification settings")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Export must use a new directory: " + str(output))
    source = open_source(contract)
    index = source_frame_index(source, episodes)
    cameras = list(source.meta.camera_keys)
    required = ["action", "observation.state", *cameras]
    features = {key: copy.deepcopy(source.features[key]) for key in required}
    features.update(
        source_episode_index={"dtype": "int64", "shape": (1,), "names": None},
        source_frame_index={"dtype": "int64", "shape": (1,), "names": None},
    )
    exported = LeRobotDataset.create(
        repo_id,
        root=output,
        fps=source.fps,
        features=features,
        robot_type=source.meta.robot_type,
        use_videos=True,
        video_backend="pyav",
        vcodec="h264",
    )
    for row in selection["segments"]:
        for frame in range(row["source_from"], row["source_to"]):
            original = source[index[row["source_episode"], frame]]
            item = {
                key: original[key].detach().cpu().clone()
                if isinstance(original[key], torch.Tensor)
                else original[key]
                for key in required
            }
            item.update(
                task=original["task"],
                source_episode_index=np.array([row["source_episode"]], dtype=np.int64),
                source_frame_index=np.array([frame], dtype=np.int64),
            )
            exported.add_frame(item)
        exported.save_episode(
            parallel_encoding=False,
            extra_episode_metadata={
                "sft_rl_crop": True,
                "source_episode_index": row["source_episode"],
                "source_from": row["source_from"],
                "source_to": row["source_to"],
                "source_identity": row["source_identity"],
                "crop_endpoint_is_success": False,
            },
        )
    exported.finalize()
    write_json(output / "selection.json", selection)
    # Re-open real encoded videos and parquet, rather than trusting the write buffer.
    check = LeRobotDataset(repo_id, root=output, download_videos=False, video_backend="pyav")
    if list(check.meta.camera_keys) != cameras or len(check) != selection["retained_frames"]:
        raise ValueError("Export camera mapping or total frame count differs")
    cursor = 0
    max_pixel_error = 0.0
    for ordinal, row in enumerate(selection["segments"]):
        for local, frame in enumerate(range(row["source_from"], row["source_to"])):
            actual = check[cursor]
            original = source[index[row["source_episode"], frame]]
            if int(actual["episode_index"]) != ordinal or int(actual["frame_index"]) != local:
                raise ValueError("Export crossed an episode boundary")
            if (
                int(actual["source_episode_index"].item()) != row["source_episode"]
                or int(actual["source_frame_index"].item()) != frame
            ):
                raise ValueError("Export provenance is not aligned with its frames")
            if abs(float(actual["timestamp"]) - local / source.fps) > 1e-4:
                raise ValueError("Export timestamp mismatch")
            for key in ("action", "observation.state"):
                torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
            for key in cameras:
                if actual[key].shape != original[key].shape:
                    raise ValueError("Export image shape or camera mapping differs")
                error = float((actual[key].float() - original[key].float()).abs().mean())
                max_pixel_error = max(max_pixel_error, error)
                if error > pixel_tolerance:
                    raise ValueError(
                        f"Encoded video/source alignment check failed: {ordinal}/{local}/{key}, MAE={error}"
                    )
            cursor += 1
    chunk_source = open_source(contract, action_horizon=output_horizon)
    replay = DemoHighReplay(chunk_source, contract, selection, output_horizon)
    starts = [replay.base_count, len(replay) - 1]
    real_batch = next(iter(DataLoader(torch.utils.data.Subset(replay, starts), batch_size=len(starts))))
    if real_batch["action"].shape[1] != output_horizon or real_batch["action_is_pad"].dtype != torch.bool:
        raise ValueError("Actual policy batch horizon or padding mask is invalid")
    previews = render_previews(source, index, selection, output / "previews", cameras, preview_count)
    report = {
        "status": "ok",
        "contract_sha256": canonical_sha256(contract),
        "selection_sha256": file_sha256(selection_path),
        "frames_checked": cursor,
        "segments": len(selection["segments"]),
        "camera_keys": cameras,
        "action_state_exact": True,
        "timestamps_checked": True,
        "episode_boundaries_checked": True,
        "all_encoded_frames_read": True,
        "max_video_mean_absolute_error": max_pixel_error,
        "video_error_tolerance": pixel_tolerance,
        "real_policy_batch_shape": list(real_batch["action"].shape),
        "output_horizon": output_horizon,
        "previews": previews,
        "crop_success_labels_fabricated": False,
    }
    write_json(output / "export_validation.json", report)
    return report


def render_previews(source, index, selection, output, cameras, count):
    if count < 1:
        return []
    from fractions import Fraction

    import av
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=False)
    lengths = [row["source_to"] - row["source_from"] for row in selection["segments"]]
    order = np.argsort(lengths, kind="stable")
    chosen = np.unique(np.linspace(0, len(order) - 1, min(count, len(order)), dtype=int))
    results = []
    for position in chosen:
        ordinal = int(order[position])
        row = selection["segments"][ordinal]
        path = output / f"segment-{ordinal:05d}.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=int(source.fps))
            stream.width, stream.height, stream.pix_fmt = 320 * len(cameras), 280, "yuv420p"
            for local, frame in enumerate(range(row["source_from"], row["source_to"])):
                sample = source[index[row["source_episode"], frame]]
                image = Image.new("RGB", (stream.width, stream.height), "#101820")
                for column, key in enumerate(cameras):
                    rgb = (sample[key].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype("uint8")
                    image.paste(Image.fromarray(rgb).resize((320, 240)), (column * 320, 0))
                ImageDraw.Draw(image).text(
                    (8, 248),
                    f"source={row['source_episode']} frame={frame} interval=[{row['source_from']},{row['source_to']})",
                    fill="white",
                )
                video = av.VideoFrame.from_image(image)
                video.pts = local
                video.time_base = Fraction(1, int(source.fps))
                for packet in stream.encode(video):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        results.append(
            {
                "path": str(path),
                "source_episode": row["source_episode"],
                "source_from": row["source_from"],
                "source_to": row["source_to"],
            }
        )
    return results
