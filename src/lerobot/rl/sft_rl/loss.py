"""Temporal action padding must not be interpreted as supervision."""

import torch


def masked_action_losses(losses: torch.Tensor, action_is_pad: torch.Tensor | None = None):
    if losses.ndim != 3:
        raise ValueError(f"Expected [batch,time,action_dim] elementwise losses, got {losses.shape}")
    if action_is_pad is None:
        return losses.mean(dim=(1, 2)), losses.mean(dim=(0, 1))
    if action_is_pad.dtype != torch.bool or action_is_pad.shape != losses.shape[:2]:
        raise ValueError("action_is_pad must be a boolean [batch,time] tensor")
    valid = ~action_is_pad.to(device=losses.device)
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("An all-padded action chunk cannot provide supervision")
    masked = torch.where(valid.unsqueeze(-1), losses, 0.0)
    per_sample = masked.sum(dim=(1, 2)) / (counts * losses.shape[-1])
    per_dimension = masked.sum(dim=(0, 1)) / counts.sum()
    return per_sample, per_dimension
