"""RL algorithm implementations for DDPO-style diffusion finetuning.

These algorithms operate on per-step diffusion transition data:
- latent x_t
- next latent x_{t-1}
- timestep t
- old log-prob under behavior policy

The core training loop can choose an algorithm and call `compute_loss(...)`.
"""

from .base import AlgorithmConfig, RlAlgorithm
from .ppo import PpoAlgorithm, PpoConfig
from .policy_gradient import PolicyGradientAlgorithm, PolicyGradientConfig
from .trpo import TrpoAlgorithm, TrpoConfig
from .sac import SacAlgorithm, SacConfig

__all__ = [
    "AlgorithmConfig",
    "RlAlgorithm",
    "PpoAlgorithm",
    "PpoConfig",
    "PolicyGradientAlgorithm",
    "PolicyGradientConfig",
    "TrpoAlgorithm",
    "TrpoConfig",
    "SacAlgorithm",
    "SacConfig",
]
