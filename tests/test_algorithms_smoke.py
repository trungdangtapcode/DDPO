import unittest

import torch

from ddpo_pytorch.algorithms import PpoAlgorithm, PolicyGradientAlgorithm, SacAlgorithm, TrpoAlgorithm


def _make_fake_batch(B: int = 8):
    # fake latents shaped like SD latents
    latents = torch.randn(B, 4, 64, 64)
    next_latents = torch.randn(B, 4, 64, 64)
    timesteps = torch.randint(0, 999, (B,))
    old_log_probs = torch.randn(B)
    advantages = torch.randn(B)
    log_prob = torch.randn(B)
    batch = {
        "latents": latents,
        "next_latents": next_latents,
        "timesteps": timesteps,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
    }
    extras = {"log_prob": log_prob}
    return batch, extras


class TestAlgorithmsSmoke(unittest.TestCase):
    def test_ppo_loss_finite(self):
        batch, extras = _make_fake_batch()
        algo = PpoAlgorithm(clip_range=1e-4)
        loss = algo.compute_loss(batch=batch, extras=extras)
        self.assertTrue(torch.isfinite(loss).item())

    def test_pg_loss_finite(self):
        batch, extras = _make_fake_batch()
        algo = PolicyGradientAlgorithm(entropy_coef=0.01)
        loss = algo.compute_loss(batch=batch, extras=extras)
        self.assertTrue(torch.isfinite(loss).item())

    def test_trpo_loss_finite_and_beta_update(self):
        batch, extras = _make_fake_batch()
        algo = TrpoAlgorithm(max_kl=1e-3, init_beta=1.0)
        loss = algo.compute_loss(batch=batch, extras=extras)
        self.assertTrue(torch.isfinite(loss).item())
        metrics = algo.get_metrics(batch=batch, extras=extras, loss=loss)
        algo.update_beta(metrics["approx_kl"])

    def test_sac_actor_critic_losses_finite(self):
        batch, extras = _make_fake_batch()
        algo = SacAlgorithm(alpha=0.2, gamma=0.99)
        actor_loss = algo.compute_actor_loss(batch=batch, extras=extras)
        critic_loss = algo.compute_critic_loss(batch=batch, extras=extras)
        self.assertTrue(torch.isfinite(actor_loss).item())
        self.assertTrue(torch.isfinite(critic_loss).item())


if __name__ == "__main__":
    unittest.main()
