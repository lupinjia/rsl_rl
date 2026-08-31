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

Implementation details: see [extra_docs/amp.md](extra_docs/amp.md).

### CTS (Concurrent Teacher-Student) extension

Integrates concurrent teacher-student distillation into PPO (see the [Concurrent TS](https://clearlab-sustech.github.io/concurrentTS/) approach): the teacher and the student share a single actor policy, distinguished only by the latent source. Teacher environments encode privileged observations with a `privilege_encoder`; student environments encode a stacked observation history with a `history_encoder`. A reconstruction loss (MSE between the history encoder output and the detached privilege encoder output) trains the student to infer the privileged latent from the history alone.

**Usage:** enable CTS by selecting the CTS classes and adding `cts_cfg` to the algorithm config:

```python
"algorithm": {
    "class_name": "rsl_rl.algorithms.ppo_cts:PPO_CTS",
    "cts_cfg": {
        "encoder_lr": 5.0e-4,
        "num_encoder_epochs": 1,
    },
},
"actor": {
    "class_name": "rsl_rl.models.cts_actor:CtsActor",
    "num_latent_dims": 256,
    "privilege_encoder_hidden_dims": [256, 128],
    "history_encoder_hidden_dims": [256, 128],
},
"obs_groups": {
    "actor": ["policy"],
    "critic": ["critic"],
    "privileged": ["privileged"],
    "history": ["history"],
},
```

The environment observations must provide the `policy`, `privileged`, `history`, `critic` and `teacher_mask` groups, where `teacher_mask` is `1` for the first `num_teacher` environments and `0` otherwise. The teacher count is an **environment property** (read from `env.num_teacher`); when the environment config leaves it unset it defaults to 3/4 of the environment count, resolved at construction time — so overriding `num_envs` on the command line (e.g. `--env.scene.num_envs`) keeps the teacher split consistent automatically.

Implementation details: see [extra_docs/cts.md](extra_docs/cts.md).

### CTS-Depth extension

Extends CTS with depth-image perception: the student additionally encodes a stack of depth camera frames (plus the observation history) with a recurrent depth estimator (CNN + GRU), while the teacher perceives a terrain heightmap grid with a CNN encoder. The student is trained to reconstruct both the privileged latent and the heightmap latent from the depth+history input through four auxiliary losses, applied in the same two-phase update as plain CTS. Depth images and history are stored student-only.

This extension implements the **Vision-CTS** learning framework from *LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots* (Su et al., [arXiv:2509.09106](https://www.alphaxiv.org/abs/2509.09106), 2025).

**Usage:** same as CTS, but select the depth-aware classes and add the `heightmap` / `depth_image` observation groups:

```python
"algorithm": {
    "class_name": "rsl_rl.algorithms.ppo_cts_depth:PPO_CTSDepth",
    "cts_cfg": {
        "encoder_lr": 5.0e-4,
        "num_encoder_epochs": 1,
    },
},
"actor": {
    "class_name": "rsl_rl.models.cts_depth_actor:CtsDepthActor",
    "num_privilege_latent_dims": 32,
    "num_heightmap_latent_dims": 64,
    "depth_cnn_channel_dims": [4],
    "depth_cnn_kernel_sizes": [3],
    "history_mlp_dims": [256, 128],
    "rnn_hidden_dim": 512,
},
"obs_groups": {
    "actor": ["actor"],
    "critic": ["critic"],
    "privileged": ["privileged"],
    "history": ["history"],
    "heightmap": ["heightmap"],
    "depth_image": ["depth_image"],
},
```

The environment must provide the CTS groups plus `heightmap` (a 2D terrain height grid, e.g. `[B, 1, H, W]`) and `depth_image` (a stacked depth stack, e.g. `[B, N, H, W]`). `num_teacher` follows the same environment property as CTS.

Implementation details: see [extra_docs/cts_depth.md](extra_docs/cts_depth.md).

## License

BSD-3-Clause (same as the original RSL-RL).