# RSL-RL (Extended)

This repository is a fork of [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) that adds custom implementations for robot motion control on top of the original GPU-accelerated reinforcement learning library.

## Relationship to the original RSL-RL

- **Upstream**: [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) (BSD-3-Clause)
- This repository fully retains the base capabilities of the original RSL-RL: the PPO and distillation algorithms, `RolloutStorage`, the RND / Symmetry extensions, multi-GPU training, and ONNX / TorchScript export.
- Custom implementations are layered on top of the upstream code on the `dev` branch. All custom extensions are **opt-in**: when disabled, the behavior is identical to the original RSL-RL.

## Branches

| Branch | Description |
|---|---|
| `dev` | Main development branch; custom extensions are layered on top of the original RSL-RL (see below) |

## Custom implementations on the `dev` branch

### AMP (Adversarial Motion Priors) extension

Integrates adversarial motion priors into PPO (see [Peng et al. 2021](https://arxiv.org/abs/2104.02180)): a discriminator learns natural, human-like motion styles from motion demonstrations and provides a style reward that augments the task reward.

**Core components:**

- `rsl_rl/extensions/amp.py` — the `AMPDiscriminator` and the `resolve_amp_config` config resolver
  - Supports the GAN / LSGAN / WGAN adversarial losses with gradient penalty
  - Extracts states from the `discriminator` / `discriminator_demonstration` observation groups and outputs a style reward
  - Style rewards are correctly scaled by `env.unwrapped.step_dt`
- `rsl_rl/storage/circular_buffer.py` — the `CircularBuffer` ring buffer for storing discriminator observation histories, with mini-batch sampling
- `rsl_rl/algorithms/ppo.py` — integrates AMP into the PPO main loop through a set of hook methods:
  - `_process_step_rewards()`: computes the per-step style reward and interpolates (lerps) it with the task reward
  - `_extra_mini_batch_iter()` / `_compute_aux_loss()`: mini-batch sampling of discriminator observations and loss computation
  - `_compute_aux_gradients()` / `_step_aux_optimizers()`: trains the discriminator with its own optimizer
  - `_aux_save_state()` / `_load_aux_state()` and others: save / load / multi-GPU synchronization of the discriminator
- `rsl_rl/utils/logger.py` — AMP training metrics: `AMP/mean_total_reward`, `AMP/mean_style_reward`, `AMP/style_ratio`
- `rsl_rl/runners/on_policy_runner.py` — extracts the style rewards in the training loop for logging

**Usage:** enable AMP by adding `amp_cfg` to the algorithm config (no need to change the algorithm class or the runner), e.g.:

```python
"algorithm": {
    "class_name": "PPO",
    "amp_cfg": {
        "loss_type": "LSGAN",
        "hidden_dims": [1024, 512],
        "activation": "elu",
        "style_reward_scale": 2.0,
        "task_style_lerp": 0.55,
        "disc_obs_buffer_size": 100,
        "disc_learning_rate": 1.0e-4,
        "grad_penalty_scale": 10.0,
    },
}
```

The environment observations must also provide the `discriminator` and `discriminator_demonstration` observation groups.

## License

BSD-3-Clause (same as the original RSL-RL).