"""Standalone version of the archived pure-vision SigLIP value architecture."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


@dataclass
class VisionValueConfig:
    vision_repo_id: str
    vision_revision: str
    camera_features: list[str]
    image_size: int = 384
    projection_dim: int = 512
    dropout: float = 0.1
    freeze_vision_encoder: bool = False
    gradient_checkpointing: bool = True

    def validate(self):
        if not self.vision_repo_id or not self.vision_revision or not self.camera_features:
            raise ValueError("Pin the pretrained vision backbone revision and camera mapping")
        if len(self.camera_features) != len(set(self.camera_features)):
            raise ValueError("Duplicate camera inputs")
        if self.image_size < 1 or self.projection_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid pure-vision value architecture")


def two_hot_targets(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or not torch.isfinite(values).all() or (values < -1).any() or (values > 0).any():
        raise ValueError("Two-hot labels must be finite scalars in [-1,0]; no silent clipping")
    scaled = (values.float() + 1) * 200
    lower = scaled.floor().long()
    upper = (lower + 1).clamp_max(200)
    weight = scaled - lower
    target = torch.zeros((len(values), 201), device=values.device, dtype=torch.float32)
    target.scatter_add_(1, lower[:, None], (1 - weight)[:, None])
    target.scatter_add_(1, upper[:, None], weight[:, None])
    return target


def value_expectation(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != 201:
        raise ValueError("Expected 201 equally spaced value bins")
    centers = torch.linspace(-1, 0, 201, device=logits.device)
    return (logits.float().softmax(-1) * centers).sum(-1)


def value_loss(logits, values):
    return -(two_hot_targets(values) * logits.float().log_softmax(-1)).sum(-1)


class PureVisionValue(nn.Module):
    def __init__(self, config: VisionValueConfig, vision_encoder):
        super().__init__()
        config.validate()
        self.config = config
        self.vision_encoder = vision_encoder
        if getattr(vision_encoder.config, "image_size", config.image_size) != config.image_size:
            raise ValueError("Value image resolution differs from the pinned vision backbone")
        hidden = vision_encoder.config.hidden_size
        dim = config.projection_dim
        self.image_projector = nn.Sequential(nn.Linear(hidden, dim), nn.GELU(), nn.Dropout(config.dropout))
        self.final_norm = nn.LayerNorm(dim)
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(dim, 201)
        )
        if config.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
        if config.gradient_checkpointing and not config.freeze_vision_encoder:
            self.vision_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @classmethod
    def from_pretrained_vision(cls, config):
        from transformers import SiglipVisionModel

        encoder = SiglipVisionModel.from_pretrained(
            config.vision_repo_id, revision=config.vision_revision, torch_dtype=torch.float32
        )
        return cls(config, encoder)

    def forward(self, images: dict[str, torch.Tensor]):
        if set(images) != set(self.config.camera_features):
            raise ValueError("Camera mapping differs from the fixed value inputs")
        views = []
        for key in self.config.camera_features:
            image = images[key]
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError("Expected batched RGB CHW images")
            image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Expected unnormalized RGB in [0,1] or uint8")
            image = F.interpolate(
                image, (self.config.image_size, self.config.image_size), mode="bilinear", align_corners=False
            )
            views.append((image - 0.5) / 0.5)
        tensor = torch.stack(views, dim=1)
        batch, cameras = tensor.shape[:2]
        features = self.vision_encoder(pixel_values=tensor.flatten(0, 1), return_dict=True).pooler_output
        projected = self.image_projector(features.float()).view(batch, cameras, -1).mean(dim=1)
        return self.value_head(self.final_norm(projected))
