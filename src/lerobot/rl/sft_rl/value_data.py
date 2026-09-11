import numpy as np
import torch
from torch.utils.data import Dataset

from .protocol import normalized_returns, validate_demo_contract
from .replay import source_frame_index


class ValueFrames(Dataset):
    def __init__(self, source, contract, episode_ids, z, camera_features, *, evaluation_frames=None):
        episodes = validate_demo_contract(contract)
        by_id = {ep.episode_index: ep for ep in episodes}
        if len(set(episode_ids)) != len(episode_ids) or not set(episode_ids) <= set(by_id):
            raise ValueError("Invalid Value split")
        self.source, self.camera_features = source, camera_features
        if getattr(source, "delta_timestamps", None):
            raise ValueError(
                "Value reads current images of complete source episodes, not policy action chunks"
            )
        index = source_frame_index(source, episodes)
        self.frames = []
        for e in sorted(episode_ids):
            ep = by_id[e]
            targets = normalized_returns(ep.length, z)
            frames = (
                np.arange(ep.length)
                if evaluation_frames is None
                else np.unique(np.linspace(0, ep.length - 1, min(evaluation_frames, ep.length), dtype=int))
            )
            self.frames.extend((index[e, int(t)], e, int(t), float(targets[t])) for t in frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        row, ep, frame, target = self.frames[index]
        sample = self.source[row]
        return {
            "images": {key: sample[key] for key in self.camera_features},
            "target": torch.tensor(target, dtype=torch.float32),
            "episode": ep,
            "frame": frame,
        }
