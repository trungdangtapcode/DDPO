
This repository contains code for **Direct/Distributional Diffusion Policy Optimization (DDPO)**-style finetuning of text-to-image diffusion models using reinforcement learning from scalar reward signals.

The goal is to study how different **policy optimization algorithms** (PPO, TRPO-style, SAC-style, and vanilla policy gradient) behave when the *policy* is a diffusion model (e.g., Stable Diffusion) and the *action sequence* is the denoising trajectory.

## repository scope

This repo focuses on implementation details that are useful for research and ablations:

- **Diffusion log-probabilities**: patched DDIM step that returns a per-step log-prob (used as the policy log-likelihood signal).
- **Reward modeling hooks**: reward functions over generated images and prompts (see `src/ddpo_pytorch/rewards.py`).
- **Algorithm modularity**: selectable RL optimizers for diffusion finetuning (PPO / TRPO-style / SAC-style / REINFORCE).
- **Experiment plumbing**: configuration via `ml_collections`, checkpointing via `accelerate`, and logging (e.g., Weights & Biases).

## Methods overview

### Diffusion as a policy

We treat the denoising trajectory as a stochastic policy over latents:

$$
\pi_\theta(x_{t-1} \mid x_t, t, c)
$$

where $c$ is the text conditioning (prompt embedding) and the transition distribution comes from the sampler (DDIM here) together with the UNet noise prediction.

### Log-prob computation (DDIM)

The module `src/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py` provides `ddim_step_with_logprob(...)`, a patched DDIM scheduler step that returns:

- `prev_sample` ($x_{t-1}$)
- `log_prob = \log \pi_\theta(x_{t-1} \mid x_t, t, c)`

This is the key signal enabling policy-gradient style updates.

### Algorithms implemented

Code is in `src/ddpo_pytorch/algorithms/`:

- **PPO** (`PpoAlgorithm`): clipped surrogate objective, matching the original training logic in `src/scripts/train.py`.
- **Policy Gradient / REINFORCE** (`PolicyGradientAlgorithm`): $-\mathbb{E}[A\log\pi]$ with optional entropy bonus.
- **TRPO-style** (`TrpoAlgorithm`): KL-controlled update implemented as a practical penalty form (adaptive $\beta$) suitable for Adam-based optimization.
- **SAC-style** (`SacAlgorithm`): minimal entropy-regularized actor-critic adaptation, including a lightweight critic and target network.

Notes:

- The TRPO implementation in this repo is a **KL-penalty trust region approximation** (research-friendly and minimally invasive). If you need “true TRPO” (conjugate gradient + line search over UNet parameters), it can be added but requires a larger refactor.
- The SAC variant here is intentionally minimal and intended for **exploratory research** rather than a production-grade baseline.

## Repository layout (research-relevant)

```
src/
  config/
    base.py                  # experiment config (algorithm selection, hyperparams)
  ddpo_pytorch/
    algorithms/              # PPO / TRPO-style / SAC-style / Policy Gradient
    diffusers_patch/         # DDIM + pipeline patches to return log-probs and trajectories
    rewards.py               # reward functions over generated images
    prompts.py               # prompt samplers
    stat_tracking.py         # per-prompt advantage normalization
  scripts/
    train.py                 # sampling + RL update loop
tests/
  test_algorithms_smoke.py   # minimal smoke tests (requires torch)
```

## Reproducible experiments

### Training entry point

The main research training loop is:

- `src/scripts/train.py`

It alternates between:

1. **Sampling**: generate trajectories with `pipeline_with_logprob(...)`.
2. **Scoring**: compute rewards with a chosen reward function.
3. **Advantage computation**: batch-normalized or per-prompt normalized advantages.
4. **Policy optimization**: update UNet parameters using the selected algorithm.

### Configuration

Experiments are configured via:

- `src/config/base.py`

Key fields:

- `train.algorithm`: `"ppo" | "trpo" | "sac" | "policy_gradient"`
- `train.clip_range` (PPO)
- `train.trpo_max_kl`, `train.trpo_init_beta` (TRPO-style)
- `train.pg_entropy_coef` (Policy Gradient)
- `train.sac_alpha`, `train.sac_gamma` (SAC-style)

### Logging and checkpoints

- Checkpointing is handled through `accelerate` state saving.
- Logging is configured for experiment tracking (e.g., W&B).

## Deployment code (secondary)

This repo also includes a full-stack demo (frontend/backend/FastAPI) for serving multiple DDPO-finetuned models. For system-level docs, see:

- `docs/ARCHITECTURE.md`
- `docs/STREAM_DIFFUSION.md`
- `docs/QUICKSTART.md`

## Citation

If you use this repository in academic work, please cite the relevant DDPO/diffusion-RL papers used as your conceptual basis (add your preferred BibTeX entries here).

## License

MIT License.
