# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterator
from itertools import chain, repeat
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import (
    RandomNetworkDistillation,
    Symmetry,
    resolve_amp_config,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.extensions.amp import AMPDiscriminator, AMPLossType
from rsl_rl.models import MLPModel
from rsl_rl.storage import CircularBuffer, RolloutStorage
from rsl_rl.utils import compile_model, resolve_class, resolve_obs_groups, resolve_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    amp_discriminator: AMPDiscriminator | None = None
    """The AMP discriminator model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        amp_discriminator: AMPDiscriminator | None = None,
        disc_obs_buffer: CircularBuffer | None = None,
        disc_demo_obs_buffer: CircularBuffer | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_mixed_precision = use_mixed_precision

        # AMP discriminator (optional). ``amp_cfg`` is consumed by construct_algorithm and
        # kept here only for configuration compatibility.
        self.amp_discriminator = amp_discriminator.to(self.device) if amp_discriminator is not None else None
        self._raw_amp_discriminator = amp_discriminator
        # AMP observation buffers
        self.disc_obs_buffer = disc_obs_buffer
        self.disc_demo_obs_buffer = disc_demo_obs_buffer
        # AMP reward tracking
        self.style_rewards: torch.Tensor | None = None
        self.rewards_lerp: torch.Tensor | None = None
        self.disc_score: torch.Tensor | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step."""
        # Allow extension hooks to transform the rewards (e.g. AMP style reward interpolation)
        rewards = self._process_step_rewards(obs, rewards, dones, extras)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute values for the last step
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        # Restore the critic's hidden state so the next rollout is not affected by the forward pass
        self.critic.reset(hidden_state=critic_hidden_state)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            self._normalize_advantages()

    def _normalize_advantages(self) -> None:
        """Normalize the stored advantages over the current rollout.

        The default implementation normalizes globally over all environments. Extensions
        that use per-group normalization (e.g. Concurrent Teacher-Student normalizing the
        teacher and student environments independently) override this method.
        """
        st = self.storage
        st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _compute_policy_loss(
        self, batch: RolloutStorage.Batch, original_batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the policy (surrogate) loss and the entropy for a mini-batch.

        Runs a single actor forward pass over the whole ``batch``, adapts the learning
        rate based on the KL divergence when adaptive scheduling is enabled, and returns
        ``(surrogate_loss, entropy)`` where ``entropy`` corresponds to the original
        (pre-augmentation) samples.

        Extensions that restructure the policy forward pass (e.g. Concurrent Teacher-Student
        with per-observation-group forward passes) override this method while preserving the
        return contract.
        """
        # Recompute actions log prob and entropy for current batch of transitions
        # Note: We need to do this because we updated the policy with new parameters
        self.actor(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            stochastic_output=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
        # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        entropy = self.actor.output_entropy[:original_batch_size]

        # Compute KL divergence and adapt the learning rate
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                kl_mean = torch.mean(kl)

                # Reduce the KL divergence across all GPUs
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

                # Update the learning rate only on the main process
                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                # Update the learning rate for all GPUs
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                # Update the learning rate for all parameter groups
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
        surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        return surrogate_loss, entropy

    def _process_step_rewards(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Transform the per-step task rewards (e.g. AMP style reward interpolation).

        Returns the rewards that are stored in the transition. The default
        implementation returns ``rewards`` unchanged.
        """
        if self.amp_discriminator is None:
            return rewards

        # Get discriminator observations
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_obs_from_demo(obs, flatten_history_dim=False)

        # Handle terminal observations for done environments
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(extras["terminal_obs"], flatten_history_dim=False)
            done_mask = dones.to(dtype=torch.bool)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]

        # Compute style reward
        step_dt = self.amp_discriminator.step_dt
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc_obs, dt=step_dt)

        # Interpolate between task and style rewards
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)

        # Store observations in buffers
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)

        # Update AMP normalizer
        self.amp_discriminator.update_normalization(disc_obs)

        return self.rewards_lerp

    def _extra_mini_batch_iter(self) -> Iterator[tuple]:
        """Yield extra per-mini-batch data, zipped with the storage batches in :meth:`update`.

        Each item is a ``(disc_obs_agent, disc_obs_demo)`` pair of tensors, or
        ``(None, None)`` when no extension provides extra data. The returned iterator
        must yield at least as many items as the storage generator.
        """
        if self.amp_discriminator is None:
            return repeat((None, None))
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        return zip(disc_obs_generator, disc_demo_obs_generator)

    def _compute_aux_loss(
        self, batch: RolloutStorage.Batch, disc_obs_batch: TensorDict | None
    ) -> tuple[torch.Tensor | None, dict]:
        """Compute an auxiliary loss for the mini-batch (e.g. AMP discriminator loss).

        Returns ``(loss, metrics)``; a ``None`` loss means no auxiliary loss is active.
        """
        if self.amp_discriminator is None or disc_obs_batch is None:
            return None, {}
        amp_loss, amp_metrics = self.amp_discriminator.compute_loss(
            disc_obs_batch["disc_obs_agent"], disc_obs_batch["disc_obs_demo"]
        )
        # Expose the loss and its metrics under stable loss-dict keys. The score metrics
        # returned by ``compute_loss`` are python floats.
        return amp_loss, {
            "amp": amp_loss,
            "agent_score": amp_metrics["amp/score_agent"],
            "demo_score": amp_metrics["amp/score_demo"],
        }

    def _compute_aux_gradients(self, aux_loss: torch.Tensor | None) -> None:
        """Zero and backpropagate an auxiliary loss (e.g. AMP discriminator loss)."""
        if self.amp_discriminator is not None and aux_loss is not None:
            self.amp_discriminator.optimizer.zero_grad()
            aux_loss.backward()

    def _step_aux_optimizers(self) -> None:
        """Step the optimizers of auxiliary modules (e.g. AMP discriminator)."""
        if self.amp_discriminator is not None:
            self.amp_discriminator.optimizer.step()

    def _train_aux_modules(self) -> None:
        """Put auxiliary modules into train mode."""
        if self.amp_discriminator is not None:
            self.amp_discriminator.train()
            self.amp_discriminator.disc_obs_normalizer.train()

    def _eval_aux_modules(self) -> None:
        """Put auxiliary modules into eval mode."""
        if self.amp_discriminator is not None:
            self.amp_discriminator.eval()
            self.amp_discriminator.disc_obs_normalizer.eval()

    def _compile_aux_modules(self, mode: str | None) -> None:
        """Compile auxiliary modules."""
        if self.amp_discriminator is not None:
            self.amp_discriminator = compile_model(self._raw_amp_discriminator, mode)  # type: ignore

    def _aux_save_state(self) -> dict:
        """Return state-dict entries of auxiliary modules for saving."""
        if self.amp_discriminator is None:
            return {}
        return {
            "amp_discriminator_state_dict": self.amp_discriminator.state_dict(),
            "amp_discriminator_optimizer_state_dict": self.amp_discriminator.optimizer.state_dict(),
        }

    def _load_aux_state(self, loaded_dict: dict, load_cfg: dict, strict: bool) -> None:
        """Load state-dict entries of auxiliary modules."""
        if load_cfg.get("amp_discriminator") and self.amp_discriminator is not None:
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            if load_cfg.get("optimizer") and "amp_discriminator_optimizer_state_dict" in loaded_dict:
                self.amp_discriminator.optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])

    def _aux_parameters(self) -> list[Iterator[nn.Parameter]]:
        """Return parameter iterables of auxiliary modules for multi-GPU gradient reduction."""
        if self.amp_discriminator is None:
            return []
        return [self.amp_discriminator.parameters()]

    def _aux_broadcast_state_dicts(self) -> list[dict]:
        """Return state dicts of auxiliary modules for multi-GPU parameter broadcasting."""
        if self.amp_discriminator is None:
            return []
        return [self.amp_discriminator.state_dict()]

    def _load_aux_broadcast_state_dicts(self, model_params: list, idx: int) -> int:
        """Consume broadcast state dicts of auxiliary modules; return the updated index."""
        if self.amp_discriminator is not None:
            self.amp_discriminator.load_state_dict(model_params[idx])
            idx += 1
        return idx

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Auxiliary loss metrics (e.g. AMP discriminator, CTS reconstruction)
        aux_metric_accum: dict[str, float] = {}
        has_aux_loss = False

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        # ``zip`` stops at the end of the shortest iterator; ``_extra_mini_batch_iter``
        # yields at least one entry per batch, so the storage generator sets the length.
        for batch, (disc_obs_batch, disc_demo_obs_batch) in zip(generator, self._extra_mini_batch_iter()):
            original_batch_size = batch.observations.batch_size[0]

            # Wrap AMP observations in a TensorDict so symmetry augmentation can
            # transform them together with the main batch
            if disc_obs_batch is not None:
                disc_obs_batch = TensorDict(
                    {
                        "disc_obs_agent": disc_obs_batch,
                        "disc_obs_demo": disc_demo_obs_batch,
                    },
                    batch_size=disc_obs_batch.shape[0],
                )

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # Recompute actions log prob and entropy for current batch of transitions.
                # Note: We need to do this because we updated the policy with new parameters.
                surrogate_loss, entropy = self._compute_policy_loss(batch, original_batch_size)
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                # RND loss
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

                # Symmetry loss
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                    if self.symmetry.use_mirror_loss:
                        loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Auxiliary loss (e.g. AMP discriminator, CTS reconstruction)
            aux_loss, aux_loss_dict = self._compute_aux_loss(batch, disc_obs_batch)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Compute the gradients for auxiliary losses (e.g. AMP discriminator)
            self._compute_aux_gradients(aux_loss)

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()
            # Apply the gradients for auxiliary optimizers (e.g. AMP discriminator)
            self._step_aux_optimizers()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # Accumulate auxiliary loss metrics
            if aux_loss is not None:
                has_aux_loss = True
                for metric_key, metric_value in aux_loss_dict.items():
                    if torch.is_tensor(metric_value):
                        metric_value = metric_value.item()
                    aux_metric_accum[metric_key] = aux_metric_accum.get(metric_key, 0.0) + metric_value

        # Update the normalizers
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)  # type: ignore
        self.critic.update_normalization(obs)  # type: ignore
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if has_aux_loss:
            aux_metric_accum = {key: value / num_updates for key, value in aux_metric_accum.items()}

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if has_aux_loss:
            loss_dict.update(aux_metric_accum)

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
        self._train_aux_modules()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
        self._eval_aux_modules()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        saved_dict.update(self._aux_save_state())
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "amp_discriminator": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        self._load_aux_state(loaded_dict, load_cfg, strict)
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    @staticmethod
    def _create_storage(env: VecEnv, cfg: dict, obs: TensorDict, device: str) -> RolloutStorage:
        """Create the rollout storage for the training setup.

        Extensions that need a custom storage (e.g. Concurrent Teacher-Student with a
        student-only history buffer) override this factory.
        """
        return RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self._compile_aux_modules(mode)

    @staticmethod
    def _create_amp_discriminator(
        obs_groups: dict, amp_cfg: dict, env: VecEnv, device: str
    ) -> tuple[AMPDiscriminator, CircularBuffer, CircularBuffer]:
        """Create the AMP discriminator and its observation buffers from a resolved config."""
        # Parse loss type
        loss_type_str = amp_cfg.get("loss_type", "LSGAN").upper()
        if loss_type_str == "GAN":
            amp_loss_type = AMPLossType.GAN
        elif loss_type_str == "LSGAN":
            amp_loss_type = AMPLossType.LSGAN
        elif loss_type_str == "WGAN":
            amp_loss_type = AMPLossType.WGAN
        else:
            raise ValueError(f"Unknown AMP loss type: {loss_type_str}. Should be 'GAN', 'LSGAN', or 'WGAN'")

        amp_discriminator = AMPDiscriminator(
            obs_groups=obs_groups,
            loss_type=amp_loss_type,
            hidden_dims=amp_cfg.get("hidden_dims", (256, 256, 256)),
            activation=amp_cfg.get("activation", "elu"),
            style_reward_scale=amp_cfg.get("style_reward_scale", 1.0),
            task_style_lerp=amp_cfg.get("task_style_lerp", 0.0),
            grad_penalty_scale=amp_cfg.get("grad_penalty_scale", 10.0),
            learning_rate=amp_cfg.get("disc_learning_rate", 5.0e-4),
            trunk_weight_decay=amp_cfg.get("disc_trunk_weight_decay", 0.0),
            linear_weight_decay=amp_cfg.get("disc_linear_weight_decay", 0.0),
            max_grad_norm=amp_cfg.get("disc_max_grad_norm", 0.5),
            step_dt=amp_cfg.get("step_dt", 0.02),
            device=device,
        )
        amp_discriminator.build_networks(
            disc_obs_dim=amp_cfg["disc_obs_dim"],
            hidden_dims=amp_cfg.get("hidden_dims", (256, 256, 256)),
            activation=amp_cfg.get("activation", "elu"),
        )
        print(f"AMP Discriminator Model: {amp_discriminator}")

        disc_obs_buffer = CircularBuffer(
            max_len=amp_cfg["disc_obs_buffer_size"],
            batch_size=env.num_envs,
            device=device,
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=amp_cfg["disc_obs_buffer_size"],
            batch_size=env.num_envs,
            device=device,
        )
        return amp_discriminator, disc_obs_buffer, disc_demo_obs_buffer

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables and configs
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        # Add AMP observation groups
        if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
            default_sets.extend(["discriminator", "discriminator_demonstration"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve AMP config if used
        alg_cfg = resolve_amp_config(alg_cfg, obs, cfg["obs_groups"], env)

        # Resolve RND config if used
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if alg_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the AMP discriminator and buffers if enabled
        if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
            amp_discriminator, disc_obs_buffer, disc_demo_obs_buffer = PPO._create_amp_discriminator(
                cfg["obs_groups"], alg_cfg["amp_cfg"], env, device
            )
        else:
            amp_discriminator, disc_obs_buffer, disc_demo_obs_buffer = None, None, None

        # Initialize the storage
        storage = alg_class._create_storage(env, cfg, obs, device)

        # Initialize the algorithm
        alg: PPO = alg_class(
            actor,
            critic,
            storage,
            device=device,
            amp_discriminator=amp_discriminator,
            disc_obs_buffer=disc_obs_buffer,
            disc_demo_obs_buffer=disc_demo_obs_buffer,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        model_params.extend(self._aux_broadcast_state_dicts())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        idx = 0
        self._raw_actor.load_state_dict(model_params[idx])
        idx += 1
        self._raw_critic.load_state_dict(model_params[idx])
        idx += 1
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[idx])
            idx += 1
        idx = self._load_aux_broadcast_state_dicts(model_params, idx)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = chain(all_params, *self._aux_parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
