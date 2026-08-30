# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with Concurrent Teacher-Student (CTS) training."""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions.amp import AMPDiscriminator
from rsl_rl.models import MLPModel
from rsl_rl.storage import CircularBuffer, RolloutStorage
from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS
from rsl_rl.utils import resolve_optimizer


class PPO_CTS(PPO):  # ruff: ignore[invalid-class-name]
    """PPO with Concurrent Teacher-Student (CTS) training.

    The teacher and the student share a single actor (see
    :class:`~rsl_rl.models.cts_actor.CtsActor`). The teacher environments (the first
    ``num_teacher`` ones) produce their latent from privileged observations, while the
    student environments use the stacked observation history. Per mini-batch, the teacher
    and student surrogate losses are computed with separate forward passes and aggregated;
    the history encoder is trained by a dedicated optimizer on a reconstruction loss against
    the (detached) privilege encoder output.
    """

    storage: RolloutStorageCTS

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorageCTS,
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
        # CTS parameters
        cts_cfg: dict | None = None,
    ) -> None:
        """Initialize the CTS algorithm with dedicated teacher/student optimizers."""
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            use_mixed_precision=use_mixed_precision,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
            amp_cfg=amp_cfg,
            amp_discriminator=amp_discriminator,
            disc_obs_buffer=disc_obs_buffer,
            disc_demo_obs_buffer=disc_demo_obs_buffer,
        )

        cts_cfg = cts_cfg or {}
        self.num_teacher = self.storage.num_teacher
        if not 0 <= self.num_teacher < self.storage.num_envs:
            raise ValueError(f"num_teacher must be in [0, num_envs), got {self.num_teacher}.")
        self.num_encoder_epochs = int(cts_cfg.get("num_encoder_epochs", 1))
        self.encoder_lr = float(cts_cfg.get("encoder_lr", 1e-3))
        self.scale_encoder_lr_with_rl = bool(cts_cfg.get("scale_encoder_lr_with_rl", True))
        self.initial_learning_rate = learning_rate

        # The history encoder is trained only by the reconstruction loss through a dedicated
        # optimizer; exclude it from the RL optimizer (consistent with Concurrent TS).
        history_encoder_params = set(self.actor.history_encoder.parameters())
        rl_params = [
            param
            for param in chain(self.actor.parameters(), self.critic.parameters())
            if param not in history_encoder_params
        ]
        self.optimizer = resolve_optimizer(optimizer)(rl_params, lr=learning_rate)  # type: ignore
        self.encoder_optimizer = resolve_optimizer(optimizer)(  # type: ignore
            self.actor.history_encoder.parameters(), lr=self.encoder_lr
        )

    def _compute_policy_loss(
        self, batch: RolloutStorageCTS.Batch, original_batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the teacher/student surrogate losses and the combined entropy."""
        if batch.indices is None:
            raise RuntimeError("PPO_CTS requires batch.indices; use RolloutStorageCTS for training.")
        obs = batch.observations[:original_batch_size]
        teacher_mask = obs["teacher_mask"].squeeze(-1).bool()
        teacher_indices = teacher_mask.nonzero(as_tuple=False).squeeze(-1)
        student_indices = (~teacher_mask).nonzero(as_tuple=False).squeeze(-1)
        flat_indices = batch.indices[:original_batch_size]

        if teacher_indices.numel() > 0:
            teacher_obs = obs[teacher_indices]
            teacher_actions = batch.actions[:original_batch_size][teacher_indices]
            teacher_old_log_prob = batch.old_actions_log_prob[:original_batch_size][teacher_indices]
            teacher_advantages = batch.advantages[:original_batch_size][teacher_indices]
            teacher_old_params = tuple(
                params[:original_batch_size][teacher_indices] for params in batch.old_distribution_params
            )
            self.actor(teacher_obs, masks=None, hidden_state=None, stochastic_output=True)
            teacher_log_prob = self.actor.get_output_log_prob(teacher_actions)
            teacher_entropy = self.actor.output_entropy
            teacher_surrogate = self._clipped_surrogate(teacher_log_prob, teacher_old_log_prob, teacher_advantages)
            # The learning rate is adapted from the teacher KL divergence only.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = torch.mean(
                        self.actor.get_kl_divergence(teacher_old_params, self.actor.output_distribution_params)
                    )
                    self._adjust_learning_rate(kl_mean)
        else:
            teacher_surrogate = torch.zeros((), device=obs.device)
            teacher_entropy = torch.zeros(0, device=obs.device)

        if student_indices.numel() > 0:
            student_obs = obs[student_indices]
            student_history = self.storage.get_student_history(flat_indices[student_indices])
            student_obs = TensorDict(
                {**student_obs.to_dict(), "history": student_history}, batch_size=[student_indices.numel()]
            )
            student_actions = batch.actions[:original_batch_size][student_indices]
            student_old_log_prob = batch.old_actions_log_prob[:original_batch_size][student_indices]
            student_advantages = batch.advantages[:original_batch_size][student_indices]
            self.actor(student_obs, masks=None, hidden_state=None, stochastic_output=True)
            student_log_prob = self.actor.get_output_log_prob(student_actions)
            student_entropy = self.actor.output_entropy
            student_surrogate = self._clipped_surrogate(student_log_prob, student_old_log_prob, student_advantages)
        else:
            student_surrogate = torch.zeros((), device=obs.device)
            student_entropy = torch.zeros(0, device=obs.device)

        surrogate_loss = teacher_surrogate + student_surrogate
        entropy = torch.cat([teacher_entropy, student_entropy], dim=0)
        return surrogate_loss, entropy

    def _normalize_advantages(self) -> None:
        """Normalize the teacher and student advantages independently."""
        st = self.storage
        teacher_advantages = st.advantages[:, : self.num_teacher]
        student_advantages = st.advantages[:, self.num_teacher :]
        if teacher_advantages.numel() > 0:
            st.advantages[:, : self.num_teacher] = (teacher_advantages - teacher_advantages.mean()) / (
                teacher_advantages.std() + 1e-8
            )
        if student_advantages.numel() > 0:
            st.advantages[:, self.num_teacher :] = (student_advantages - student_advantages.mean()) / (
                student_advantages.std() + 1e-8
            )

    def _compute_encoder_loss(self, batch: RolloutStorageCTS.Batch) -> torch.Tensor | None:
        """Compute the CTS reconstruction loss on the student samples of a batch.

        Returns None when the batch contains no student environments.
        """
        if batch.indices is None:
            return None
        teacher_mask = batch.observations["teacher_mask"].squeeze(-1).bool()
        student_indices = (~teacher_mask).nonzero(as_tuple=False).squeeze(-1)
        if student_indices.numel() == 0:
            return None
        student_history = self.storage.get_student_history(batch.indices[student_indices])
        student_privileged = batch.observations["privileged"][student_indices]
        encoder_predictions = self.actor.history_encoder(student_history)
        with torch.no_grad():
            encoder_targets = self.actor.privilege_encoder(student_privileged)
        return nn.functional.mse_loss(encoder_predictions, encoder_targets)

    def _post_rl_aux_phase(self) -> dict | None:
        """Train the history encoder after the RL updates (two-phase update).

        Mirrors the original Concurrent-TS update: once all RL losses have been
        applied, the encoder is trained to reconstruct the (detached) privilege
        encoder latent from the stored student histories, re-fitting each
        mini-batch ``num_encoder_epochs`` times.
        """
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        mean_reconstruction_loss = 0.0
        num_encoder_updates = 0
        for batch in generator:
            for _ in range(self.num_encoder_epochs):
                reconstruction_loss = self._compute_encoder_loss(batch)
                if reconstruction_loss is None:
                    continue
                self.encoder_optimizer.zero_grad()
                reconstruction_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.history_encoder.parameters(), self.max_grad_norm)
                self.encoder_optimizer.step()
                mean_reconstruction_loss += reconstruction_loss.item()
                num_encoder_updates += 1
        if num_encoder_updates == 0:
            return None
        return {"cts/reconstruction": mean_reconstruction_loss / num_encoder_updates}

    def _train_aux_modules(self) -> None:
        """Put auxiliary modules into train mode."""
        super()._train_aux_modules()
        self.actor.history_encoder.train()

    def _eval_aux_modules(self) -> None:
        """Put auxiliary modules into eval mode."""
        super()._eval_aux_modules()
        self.actor.history_encoder.eval()

    def _aux_save_state(self) -> dict:
        """Return state-dict entries of auxiliary modules for saving."""
        state = super()._aux_save_state()
        state["cts_encoder_optimizer_state_dict"] = self.encoder_optimizer.state_dict()
        return state

    def _load_aux_state(self, loaded_dict: dict, load_cfg: dict, strict: bool) -> None:
        """Load state-dict entries of auxiliary modules."""
        super()._load_aux_state(loaded_dict, load_cfg, strict)
        if load_cfg.get("optimizer") and "cts_encoder_optimizer_state_dict" in loaded_dict:
            self.encoder_optimizer.load_state_dict(loaded_dict["cts_encoder_optimizer_state_dict"])

    @staticmethod
    def _create_storage(env: VecEnv, cfg: dict, obs: TensorDict, device: str) -> RolloutStorage:
        """Create a student-only history rollout storage for CTS training."""
        num_teacher = int(getattr(env, "num_teacher", 0))
        return RolloutStorageCTS(
            "rl", env.num_envs, num_teacher, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

    def _clipped_surrogate(
        self, actions_log_prob: torch.Tensor, old_actions_log_prob: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute the clipped PPO surrogate loss for a group of samples."""
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _adjust_learning_rate(self, kl_mean: torch.Tensor) -> None:
        """Adapt the learning rate from the KL divergence across all GPUs."""
        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
        if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        # Scale the history encoder LR proportionally to the RL LR (relative to its
        # initial value) so it tracks the privilege-encoder target's evolution rate.
        if self.scale_encoder_lr_with_rl:
            scaled_encoder_lr = self.encoder_lr * self.learning_rate / self.initial_learning_rate
            for param_group in self.encoder_optimizer.param_groups:
                param_group["lr"] = scaled_encoder_lr
