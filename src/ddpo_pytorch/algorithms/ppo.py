from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .base import AlgorithmConfig, ensure_1d


@dataclass(frozen=True)
class PpoConfig(AlgorithmConfig):
    clip_range: float = 1e-4


class PpoAlgorithm:
    """PPO clipped surrogate objective.

    This matches the logic currently embedded in `src/scripts/train.py`.
    """

    def __init__(self, *, clip_range: float = 1e-4):
        self.config = PpoConfig(name="ppo", clip_range=clip_range)

    def compute_loss(
        self, *, batch: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        advantages = ensure_1d(batch["advantages"], "advantages")
        old_log_probs = ensure_1d(batch["old_log_probs"], "old_log_probs")
        log_prob = ensure_1d(extras["log_prob"], "log_prob")

        ratio = torch.exp(log_prob - old_log_probs)
        unclipped = -advantages * ratio
        clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
        )
        return torch.mean(torch.maximum(unclipped, clipped))

    def get_metrics(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        extras: Dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        old_log_probs = ensure_1d(batch["old_log_probs"], "old_log_probs")
        log_prob = ensure_1d(extras["log_prob"], "log_prob")
        ratio = torch.exp(log_prob - old_log_probs)
        # Schulman KL approx used in the original script
        approx_kl = 0.5 * torch.mean((log_prob - old_log_probs) ** 2)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.clip_range).float())
        return {
            "loss": loss.detach(),
            "approx_kl": approx_kl.detach(),
            "clipfrac": clipfrac.detach(),
        }
