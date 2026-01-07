from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import AlgorithmConfig, ensure_1d


@dataclass(frozen=True)
class SacConfig(AlgorithmConfig):
    alpha: float = 0.2
    gamma: float = 0.99


class _Critic(nn.Module):
    """Tiny critic MLP over pooled latent + timestep embedding.

    This is intentionally small and generic; the diffusion UNet remains the actor.

    Input features:
    - mean pooled latent (B, 4)
    - mean pooled next latent (B, 4)
    - timestep (B, 1)

    Output:
    - Q-value scalar (B,)

    This is a *minimal* SAC adaptation for the diffusion transition setting.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + 4 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latents: torch.Tensor, next_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                latents.mean(dim=(2, 3)),
                next_latents.mean(dim=(2, 3)),
                timesteps.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


class SacAlgorithm:
    """Soft Actor-Critic (minimal) for diffusion transitions.

    This is provided as an *optional* algorithm; it requires maintaining critic
    networks and their optimizers.

    For now, this module exposes:
    - actor_loss: maximize Q - alpha * log_prob
    - critic_loss: TD target using reward-like advantages as immediate reward proxy

    Because DDPO uses advantages rather than explicit per-step rewards, we treat
    `advantages` as a per-transition reward signal.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        gamma: float = 0.99,
        critic: Optional[nn.Module] = None,
        critic_target: Optional[nn.Module] = None,
    ):
        self.config = SacConfig(name="sac", alpha=alpha, gamma=gamma)
        self.critic = critic if critic is not None else _Critic()
        self.critic_target = critic_target if critic_target is not None else _Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

    def compute_actor_loss(
        self, *, batch: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        log_prob = ensure_1d(extras["log_prob"], "log_prob")
        q = self.critic(batch["latents"], batch["next_latents"], ensure_1d(batch["timesteps"], "timesteps"))
        return torch.mean(self.config.alpha * log_prob - q)

    def compute_critic_loss(
        self, *, batch: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        advantages = ensure_1d(batch["advantages"], "advantages")
        timesteps = ensure_1d(batch["timesteps"], "timesteps")
        with torch.no_grad():
            q_next = self.critic_target(batch["latents"], batch["next_latents"], timesteps)
            target = advantages + self.config.gamma * q_next

        q = self.critic(batch["latents"], batch["next_latents"], timesteps)
        return torch.mean((q - target) ** 2)

    # For compatibility with the training loop API, we expose actor loss as `compute_loss`.
    def compute_loss(
        self, *, batch: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.compute_actor_loss(batch=batch, extras=extras)

    def soft_update_target(self, tau: float = 0.005) -> None:
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

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
