from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch


TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class AlgorithmConfig:
    """Base config for an RL algorithm."""

    name: str


class RlAlgorithm(Protocol):
    """Shared interface used by the training loop.

    Contract (current DDPO training loop):
    - Inputs include current-step diffusion transition tensors and advantage estimates.
    - Returns a scalar loss to backprop through the UNet.

    Required keys in `batch`:
    - "latents": (B, 4, 64, 64) latent x_t
    - "next_latents": (B, 4, 64, 64) latent x_{t-1}
    - "timesteps": (B,) timestep tensor for each sample
    - "old_log_probs": (B,) behavior-policy log prob of x_{t-1} given x_t
    - "advantages": (B,) advantage per sample (already normalized/clipped if desired)

    Required keys in `extras`:
    - "noise_pred": UNet output for x_t at timestep t (shape matches latents)
    - "log_prob": current-policy log prob of x_{t-1} given x_t

    The training loop is responsible for computing `noise_pred` and `log_prob`
    (via `ddim_step_with_logprob`) because that wiring depends on scheduler/model.
    """

    config: AlgorithmConfig

    def compute_loss(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        extras: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ...

    def get_metrics(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        extras: Dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ...


def detach_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in batch.items()}


def ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    if x.ndim != 1:
        raise ValueError(f"Expected {name} to be 1D (B,) but got shape {tuple(x.shape)}")
    return x
