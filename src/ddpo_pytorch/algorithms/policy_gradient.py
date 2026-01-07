from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .base import AlgorithmConfig, ensure_1d


@dataclass(frozen=True)
class PolicyGradientConfig(AlgorithmConfig):
    # optional entropy bonus coefficient (encourages exploration)
    entropy_coef: float = 0.0


class PolicyGradientAlgorithm:
    """REINFORCE / vanilla policy gradient.

    Loss: -E[ A * log pi(a|s) ] - entropy_coef * E[H(pi)]

    In this diffusion setting, `log_prob` is the log-prob of the *observed* transition
    x_{t-1} under the current UNet-induced DDIM transition.

    Note: we approximate entropy via `-log_prob` (since we don't have full distribution
    entropy in closed form here). This is a common practical shortcut.
    """

    def __init__(self, *, entropy_coef: float = 0.0):
        self.config = PolicyGradientConfig(name="policy_gradient", entropy_coef=entropy_coef)

    def compute_loss(
        self, *, batch: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        advantages = ensure_1d(batch["advantages"], "advantages")
        log_prob = ensure_1d(extras["log_prob"], "log_prob")

        pg_loss = -(advantages * log_prob)
        entropy_bonus = self.config.entropy_coef * (-log_prob)
        # subtract entropy bonus -> maximize entropy
        return torch.mean(pg_loss - entropy_bonus)

    def get_metrics(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        extras: Dict[str, torch.Tensor],
        loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        log_prob = ensure_1d(extras["log_prob"], "log_prob")
        return {
            "loss": loss.detach(),
            "log_prob_mean": log_prob.mean().detach(),
        }
