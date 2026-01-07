from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .base import AlgorithmConfig, ensure_1d


@dataclass(frozen=True)
class TrpoConfig(AlgorithmConfig):
    max_kl: float = 1e-3
    damping: float = 1e-2


class TrpoAlgorithm:
    """A pragmatic TRPO-style objective.

    Full TRPO requires a trust-region constrained optimization (CG + line search).
    In this repo, the training loop is built around Adam-style steps, so we implement
    the *penalized* trust region variant:

        L = -E[A * ratio] + beta * E[KL(old || new)]

    where beta is adapted to enforce the KL constraint.

    This gives you TRPO-like behavior (KL-controlled updates) without invasive
    optimizer rewrites.

    Notes:
    - We approximate KL(old||new) as (old_log_prob - log_prob) averaged.
    - `beta` is updated externally by the training loop (or can be kept fixed).

    If you want *exact* TRPO (CG + backtracking line search), we can add it next,
    but it will require direct parameter vector manipulation of the UNet.
    """

    def __init__(self, *, max_kl: float = 1e-3, init_beta: float = 1.0, damping: float = 1e-2):
        self.config = TrpoConfig(name="trpo", max_kl=max_kl, damping=damping)
        self.beta = init_beta

    def compute_loss(
        self, *, batch: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        advantages = ensure_1d(batch["advantages"], "advantages")
        old_log_probs = ensure_1d(batch["old_log_probs"], "old_log_probs")
        log_prob = ensure_1d(extras["log_prob"], "log_prob")

        ratio = torch.exp(log_prob - old_log_probs)
        surrogate = -(advantages * ratio)

        # KL(old || new) approx
        kl = torch.mean(old_log_probs - log_prob)
        return torch.mean(surrogate) + self.beta * kl

    def update_beta(self, observed_kl: torch.Tensor) -> None:
        """Heuristic to keep KL near target."""
        kl_val = float(observed_kl.detach().cpu())
        if kl_val > 1.5 * self.config.max_kl:
            self.beta *= 2.0
        elif kl_val < 0.5 * self.config.max_kl:
            self.beta *= 0.5

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
        approx_kl = torch.mean(old_log_probs - log_prob)
        return {
            "loss": loss.detach(),
            "approx_kl": approx_kl.detach(),
            "ratio_mean": ratio.mean().detach(),
            "trpo_beta": torch.as_tensor(self.beta),
        }
