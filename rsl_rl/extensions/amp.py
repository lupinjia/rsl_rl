# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) extension for imitation learning.

Reference:
    - Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control." ACM TOG 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from enum import Enum
from tensordict import TensorDict
from torch import autograd

from rsl_rl.env import VecEnv
from rsl_rl.modules import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class AMPLossType(Enum):
    """AMP discriminator loss types."""

    GAN = 0
    """Standard GAN loss with BCE."""

    LSGAN = 1
    """Least Squares GAN loss."""

    WGAN = 2
    """Wasserstein GAN loss."""


class AMPDiscriminator(nn.Module):
    """AMP discriminator for distinguishing agent and demonstration motion.

    The discriminator provides a style reward that encourages the agent to
    imitate the motion style of the demonstration data.
    """

    def __init__(
        self,
        obs_groups: dict,
        loss_type: AMPLossType = AMPLossType.LSGAN,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        style_reward_scale: float = 1.0,
        task_style_lerp: float = 0.0,
        grad_penalty_scale: float = 10.0,
        learning_rate: float = 5.0e-4,
        trunk_weight_decay: float = 0.0,
        linear_weight_decay: float = 0.0,
        max_grad_norm: float = 0.5,
        step_dt: float = 0.02,
        device: str = "cpu",
    ) -> None:
        """Initialize the AMP discriminator.

        Args:
            obs_groups: Observation groups configuration.
            loss_type: Type of GAN loss (GAN, LSGAN, WGAN).
            hidden_dims: Hidden layer dimensions for discriminator network.
            activation: Activation function name.
            style_reward_scale: Scale factor for style rewards.
            task_style_lerp: Interpolation factor between task and style rewards (0=style only, 1=task only).
            grad_penalty_scale: Scale for gradient penalty.
            learning_rate: Learning rate for discriminator optimizer.
            trunk_weight_decay: Weight decay for trunk layers.
            linear_weight_decay: Weight decay for output layer.
            max_grad_norm: Max gradient norm for discriminator.
            step_dt: Environment control time step (``env.unwrapped.step_dt``), used to scale style rewards.
            device: Device for computation.
        """
        super().__init__()

        self.obs_groups = obs_groups
        self.loss_type = loss_type
        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.grad_penalty_scale = grad_penalty_scale
        self.max_grad_norm = max_grad_norm
        self.step_dt = step_dt
        self.device = device

        # Discriminator dimensions (will be resolved from config)
        self.disc_obs_dim = 0
        self.input_dim = 0

        self.disc_obs_normalizer: nn.Module = nn.Identity()
        self.disc_trunk: nn.Module = nn.Identity()
        self.disc_linear: nn.Module = nn.Identity()
        self.disc_output_normalizer: nn.Module = nn.Identity()

        self.optimizer: torch.optim.Optimizer | None = None
        self._optimizer_cfg = {
            "learning_rate": learning_rate,
            "trunk_weight_decay": trunk_weight_decay,
            "linear_weight_decay": linear_weight_decay,
        }

    def build_networks(
        self,
        disc_obs_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        """Build discriminator networks.

        Args:
            disc_obs_dim: Dimension of discriminator observations.
            hidden_dims: Hidden layer dimensions.
            activation: Activation function.
        """
        # 不需要 disc_obs_steps 了, 因为我们直接把历史维度和特征维度展平了, 输入到 MLP 中
        self.disc_obs_dim = disc_obs_dim
        self.input_dim = disc_obs_dim

        # Discriminator observation normalizer
        self.disc_obs_normalizer = EmpiricalNormalization(shape=self.disc_obs_dim, until=int(1e8)).to(self.device)

        # Build the discriminator network using MLP
        # MLP expects activation as string and resolves it internally
        self.disc_trunk = MLP(
            input_dim=self.input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=list(hidden_dims[:-1]),
            activation=activation,
        ).to(self.device)

        self.disc_linear = nn.Linear(hidden_dims[-1], 1).to(self.device)

        if self.loss_type == AMPLossType.WGAN:
            self.disc_output_normalizer = EmpiricalNormalization(shape=1, until=int(1e8)).to(self.device)
        else:
            self.disc_output_normalizer = nn.Identity()

        # Create optimizer
        disc_params = [
            {
                "name": "disc_trunk",
                "params": self.disc_trunk.parameters(),
                "weight_decay": self._optimizer_cfg["trunk_weight_decay"],
            },
            {
                "name": "disc_linear",
                "params": self.disc_linear.parameters(),
                "weight_decay": self._optimizer_cfg["linear_weight_decay"],
            },
        ]
        self.optimizer = torch.optim.Adam(disc_params, lr=self._optimizer_cfg["learning_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.

        Args:
            x: Input tensor of shape (batch_size, disc_obs_dim * disc_obs_steps).

        Returns:
            Discriminator logits of shape (batch_size, 1).
        """
        h = self.disc_trunk(x)
        d = self.disc_linear(h)
        return d

    def get_disc_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """Extract discriminator observations from environment observations."""
        disc_obs_list = []
        for obs_group in self.obs_groups["discriminator"]:
            obs_tensor = obs[obs_group]
            disc_obs_list.append(obs_tensor)

        disc_obs = torch.cat(disc_obs_list, dim=-1)
        if flatten_history_dim:
            num_envs = disc_obs.shape[0]
            disc_obs = disc_obs.view(num_envs, -1)
        return disc_obs

    def get_disc_obs_from_demo(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """Extract demonstration observations from environment observations."""
        disc_obs_list = []
        for obs_group in self.obs_groups["discriminator_demonstration"]:
            obs_tensor = obs[obs_group]
            disc_obs_list.append(obs_tensor)

        disc_obs = torch.cat(disc_obs_list, dim=-1)
        if flatten_history_dim:
            num_envs = disc_obs.shape[0]
            disc_obs = disc_obs.view(num_envs, -1)
        return disc_obs

    def normalize_disc_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
        """Normalize discriminator observations.

        Args:
            disc_obs: Discriminator observations of shape (num_envs, disc_obs_steps, disc_obs_dim).

        Returns:
            Normalized observations.
        """
        disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
        normed_disc_obs = self.disc_obs_normalizer(disc_obs_reshaped)
        normed_disc_obs = normed_disc_obs.reshape(-1, self.disc_obs_dim)
        return normed_disc_obs

    def update_normalization(self, disc_obs: torch.Tensor) -> None:
        """Update the observation normalizer with new data.

        Args:
            disc_obs: Discriminator observations.
        """
        disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
        self.disc_obs_normalizer.update(disc_obs_reshaped)

    def compute_grad_penalty(self, demo_data: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for discriminator training.

        Args:
            demo_data: Demonstration data of shape (num_samples, disc_obs_dim * disc_obs_steps).

        Returns:
            Gradient penalty loss.
        """
        demo_data_copy = demo_data.clone().detach().requires_grad_(True)

        disc = self.forward(demo_data_copy)
        ones = torch.ones_like(disc, device=demo_data_copy.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=demo_data_copy,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the grad norm approaches 0
        grad_penalty = self.grad_penalty_scale * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_penalty

    def predict_style_reward(self, disc_obs: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute style reward from discriminator observations.

        Args:
            disc_obs: Discriminator observations of shape (num_envs, disc_obs_steps, disc_obs_dim).
            dt: Time step duration for reward scaling.

        Returns:
            Tuple of (style_rewards, disc_scores) both of shape (num_envs,).
        """
        was_training = self.training
        with torch.no_grad():
            self.eval()

            # Normalize the input data
            disc_obs_reshaped = disc_obs.view(-1, self.disc_obs_dim)
            normed_disc_obs = self.disc_obs_normalizer(disc_obs_reshaped)
            normed_disc_obs = normed_disc_obs.view(-1, self.disc_obs_dim)

            disc_score = self.forward(normed_disc_obs)

            if self.loss_type == AMPLossType.GAN:
                prob = 1.0 / (1.0 + torch.exp(-disc_score))
                rew = -torch.log(torch.maximum(1 - prob, torch.tensor(1e-6, device=self.device)))
            elif self.loss_type == AMPLossType.LSGAN:
                rew = torch.clamp(1 - (1 / 4) * torch.square(disc_score - 1), min=0)
            elif self.loss_type == AMPLossType.WGAN:
                rew = self.disc_output_normalizer(disc_score)
            else:
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}")

            style_reward = dt * self.style_reward_scale * rew

            if was_training:
                self.train()
                if self.loss_type == AMPLossType.WGAN:
                    self.disc_output_normalizer.update(disc_score)

        return style_reward.squeeze(-1), disc_score.squeeze(-1)

    def compute_loss(
        self,
        disc_obs_agent: torch.Tensor,
        disc_obs_demo: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the discriminator loss.

        Args:
            disc_obs_agent: Agent discriminator observations.
            disc_obs_demo: Demonstration discriminator observations.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Normalize observations
        disc_obs_agent_normed = self.normalize_disc_obs(disc_obs_agent)
        disc_obs_demo_normed = self.normalize_disc_obs(disc_obs_demo)

        # Flatten for discriminator input
        batch_size_agent = disc_obs_agent_normed.shape[0]
        batch_size_demo = disc_obs_demo_normed.shape[0]

        disc_score_agent = self.discriminator(disc_obs_agent_normed.reshape(batch_size_agent, -1))
        disc_score_demo = self.discriminator(disc_obs_demo_normed.reshape(batch_size_demo, -1))

        # Compute loss based on loss type
        if self.loss_type == AMPLossType.GAN:
            bce = torch.nn.BCEWithLogitsLoss()
            policy_loss = bce(disc_score_agent, torch.zeros_like(disc_score_agent, device=self.device))
            demo_loss = bce(disc_score_demo, torch.ones_like(disc_score_demo, device=self.device))
            disc_loss = 0.5 * (policy_loss + demo_loss)
        elif self.loss_type == AMPLossType.LSGAN:
            policy_loss = torch.nn.MSELoss()(
                disc_score_agent, -1 * torch.ones_like(disc_score_agent, device=self.device)
            )
            demo_loss = torch.nn.MSELoss()(disc_score_demo, torch.ones_like(disc_score_demo, device=self.device))
            disc_loss = 0.5 * (policy_loss + demo_loss)
        elif self.loss_type == AMPLossType.WGAN:
            disc_loss = -torch.mean(disc_score_demo) + torch.mean(disc_score_agent)
        else:
            raise ValueError(f"Unknown AMP loss type: {self.loss_type}")

        # Gradient penalty
        grad_penalty = self.compute_grad_penalty(disc_obs_demo_normed.reshape(batch_size_demo, -1))
        total_loss = disc_loss + grad_penalty

        # Create loss dictionary
        loss_dict = {
            "amp/disc_loss": disc_loss.item(),
            "amp/grad_penalty": grad_penalty.item(),
            "amp/score_agent": disc_score_agent.mean().item(),
            "amp/score_demo": disc_score_demo.mean().item(),
        }

        return total_loss, loss_dict

    def discriminator(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminator forward pass (trunk + linear)."""
        return self.forward(x)

    def lerp_reward(self, task_reward: torch.Tensor, style_reward: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate between task reward and style reward.

        Args:
            task_reward: Task/environment rewards.
            style_reward: Style rewards from discriminator.

        Returns:
            Interpolated rewards.
        """
        return self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward


def resolve_amp_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict, env: VecEnv) -> dict:
    """Resolve AMP configuration by inferring dimensions from observations.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs: Example observations from environment.
        obs_groups: Observation groups configuration.
        env: Vectorized environment.

    Returns:
        Updated algorithm configuration with resolved AMP settings.
    """
    if "amp_cfg" not in alg_cfg or alg_cfg["amp_cfg"] is None:
        return alg_cfg

    # Get example AMP observation to infer dimensions
    disc_obs_dim = 0

    if "discriminator" not in obs_groups or "discriminator_demonstration" not in obs_groups:
        raise ValueError(
            "AMP configuration requires 'discriminator' and 'discriminator_demonstration' observation groups."
        )

    for obs_group in obs_groups["discriminator"]:
        obs_tensor = obs[obs_group]
        disc_obs_dim += obs_tensor.shape[-1]

    disc_demo_obs_dim = 0
    for obs_group in obs_groups["discriminator_demonstration"]:
        obs_tensor = obs[obs_group]
        disc_demo_obs_dim += obs_tensor.shape[-1]

    if disc_demo_obs_dim != disc_obs_dim:
        raise ValueError("The dimension of demonstration and agent discriminator observations must match.")

    alg_cfg["amp_cfg"]["disc_obs_dim"] = disc_obs_dim
    alg_cfg["amp_cfg"]["step_dt"] = env.unwrapped.step_dt

    return alg_cfg
