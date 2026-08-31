# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with depth-augmented Concurrent Teacher-Student (CTS) training."""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_cts import PPO_CTS
from rsl_rl.env import VecEnv
from rsl_rl.extensions.amp import AMPDiscriminator
from rsl_rl.models import MLPModel
from rsl_rl.storage import CircularBuffer, RolloutStorage
from rsl_rl.storage.rollout_storage_cts_depth import RolloutStorageCTSDepth
from rsl_rl.utils import resolve_optimizer
from rsl_rl.utils.utils import unpad_trajectories_flat


class PPO_CTSDepth(PPO_CTS):  # ruff: ignore[invalid-class-name]
    """PPO with depth-augmented Concurrent Teacher-Student (CTS) training.

    On top of :class:`PPO_CTS`, the student additionally perceives stacked depth
    images through the recurrent :attr:`~CtsDepthActor.depth_estimator` (CNN + GRU)
    and the teacher perceives a terrain heightmap through
    :attr:`~CtsDepthActor.heightmap_encoder`. The update is two-phase: the RL losses
    are applied first (teacher surrogate + recurrent student surrogate + value),
    then the student-side networks are trained by four auxiliary losses
    (heightmap latent reconstruction / estimation and privilege latent
    reconstruction / estimation) through a single :attr:`aux_optimizer` — sequential
    per-loss steps, mirroring the original ppo_cts_depth.
    """

    storage: RolloutStorageCTSDepth

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorageCTSDepth,
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
        """Initialize the depth-aware CTS algorithm.

        The student-side networks (privilege/depth estimators and the decoders) are
        trained only by the auxiliary losses through a single ``aux_optimizer`` (M2
        design); the teacher-side networks (privilege/heightmap encoders) plus the
        shared actor and critic are trained by the RL optimizer.
        """
        # Skip PPO_CTS.__init__ (it references a plain history_encoder): call PPO directly.
        PPO.__init__(
            self,
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

        # The student-side networks are trained only by the auxiliary losses through a
        # dedicated optimizer; exclude them from the RL optimizer (consistent with the
        # original ppo_cts_depth, which also keeps the student estimators out of the
        # teacher/RL optimizer).
        aux_params = set(
            chain(
                self.actor.privilege_estimator.parameters(),
                self.actor.privilege_decoder.parameters(),
                self.actor.heightmap_decoder.parameters(),
                self.actor.depth_estimator.parameters(),
            )
        )
        rl_params = [
            param
            for param in chain(self.actor.parameters(), self.critic.parameters())
            if param not in aux_params
        ]
        self.optimizer = resolve_optimizer(optimizer)(rl_params, lr=learning_rate)  # type: ignore
        self.aux_optimizer = resolve_optimizer(optimizer)(list(aux_params), lr=self.encoder_lr)  # type: ignore
        # The student GRU hidden state is student-only; reset needs the teacher offset.
        self.actor.num_teacher = self.num_teacher

    def _compute_policy_loss(
        self, batch: RolloutStorageCTSDepth.Batch, original_batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the teacher (flat) and student (recurrent) surrogate losses."""
        obs = batch.observations[:original_batch_size]
        teacher_mask = obs["teacher_mask"].squeeze(-1).bool()
        teacher_indices = teacher_mask.nonzero(as_tuple=False).squeeze(-1)
        student_indices = (~teacher_mask).nonzero(as_tuple=False).squeeze(-1)

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

        if batch.student_observations is not None and student_indices.numel() > 0:
            student_actions = batch.actions[:original_batch_size][student_indices]
            student_old_log_prob = batch.old_actions_log_prob[:original_batch_size][student_indices]
            student_advantages = batch.advantages[:original_batch_size][student_indices]
            self.actor(
                batch.student_observations,
                masks=batch.student_masks,
                hidden_state=batch.student_hidden_states,
                stochastic_output=True,
            )
            student_log_prob = self.actor.get_output_log_prob(student_actions)
            student_entropy = self.actor.output_entropy
            student_surrogate = self._clipped_surrogate(student_log_prob, student_old_log_prob, student_advantages)
        else:
            student_surrogate = torch.zeros((), device=obs.device)
            student_entropy = torch.zeros(0, device=obs.device)

        surrogate_loss = teacher_surrogate + student_surrogate
        entropy = torch.cat([teacher_entropy, student_entropy], dim=0)
        return surrogate_loss, entropy

    def _post_rl_aux_phase(self) -> dict | None:
        """Train the student-side networks after the RL updates (two-phase update).

        Mirrors the original ppo_cts_depth: once all RL losses have been applied, the
        four auxiliary losses are optimized sequentially (M2 design, one shared
        ``aux_optimizer``): heightmap latent reconstruction, heightmap estimation,
        privilege latent reconstruction and privilege estimation.
        """
        generator = self.storage.recurrent_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        metrics = {
            "heightmap_recon": 0.0,
            "heightmap_estimation": 0.0,
            "privilege_recon": 0.0,
            "privilege_estimation": 0.0,
        }
        num_aux_updates = 0
        for batch in generator:
            for _ in range(self.num_encoder_epochs):
                losses = self._compute_depth_aux_losses(batch)
                if losses is None:
                    continue
                for key in metrics:
                    metrics[key] += losses[key].item()
                num_aux_updates += 1
        if num_aux_updates == 0:
            return None
        return {f"cts_depth/{key}": value / num_aux_updates for key, value in metrics.items()}

    def _compute_depth_aux_losses(self, batch: RolloutStorageCTSDepth.Batch) -> dict[str, torch.Tensor] | None:
        """Run one sequential auxiliary update and return the per-loss values.

        The four losses are optimized sequentially: each loss sees the parameters
        updated by the previous one, mirroring the original ppo_cts_depth. This is
        equivalent to the source's four separate optimizers because Adam state is
        per-parameter and all of them share the same learning rate.
        """
        if batch.student_observations is None:
            return None
        student_masks = batch.student_masks
        hist_padded = batch.student_observations["history"]
        depth_padded = batch.student_observations["depth_image"]
        hidden_states = batch.student_hidden_states

        teacher_mask = batch.observations["teacher_mask"].squeeze(-1).bool()
        student_indices = (~teacher_mask).nonzero(as_tuple=False).squeeze(-1)
        if student_indices.numel() == 0:
            return None
        student_privileged = batch.observations["privileged"][student_indices]
        student_heightmap = batch.observations["heightmap"][student_indices]
        student_heightmap_flat = student_heightmap.reshape(student_heightmap.shape[0], -1)
        hist_flat = unpad_trajectories_flat(hist_padded, student_masks)

        with torch.no_grad():
            teacher_hm_latent = self.actor.heightmap_encoder(student_heightmap)
            teacher_priv_latent = self.actor.privilege_encoder(student_privileged)

        losses: dict[str, torch.Tensor] = {}
        # 1. Heightmap latent reconstruction: student depth estimator matches the teacher
        #    heightmap encoder output.
        hm_latent = self.actor.depth_estimator(hist_padded, depth_padded, hidden_states, student_masks)
        losses["heightmap_recon"] = nn.functional.mse_loss(hm_latent, teacher_hm_latent)
        self._aux_optimizer_step(
            losses["heightmap_recon"], chain(self.actor.depth_estimator.parameters())
        )
        # 2. Heightmap estimation: the decoder reconstructs the heightmap from the latent
        #    (recomputed with the updated depth estimator).
        hm_latent = self.actor.depth_estimator(hist_padded, depth_padded, hidden_states, student_masks)
        losses["heightmap_estimation"] = nn.functional.mse_loss(
            self.actor.heightmap_decoder(hm_latent), student_heightmap_flat
        )
        self._aux_optimizer_step(
            losses["heightmap_estimation"],
            chain(self.actor.depth_estimator.parameters(), self.actor.heightmap_decoder.parameters()),
        )
        # 3. Privilege latent reconstruction: the student history estimator matches the
        #    teacher privilege encoder output.
        priv_latent = self.actor.privilege_estimator(hist_flat)
        losses["privilege_recon"] = nn.functional.mse_loss(priv_latent, teacher_priv_latent)
        self._aux_optimizer_step(losses["privilege_recon"], chain(self.actor.privilege_estimator.parameters()))
        # 4. Privilege estimation: the decoder reconstructs the privileged observations.
        priv_latent = self.actor.privilege_estimator(hist_flat)
        losses["privilege_estimation"] = nn.functional.mse_loss(
            self.actor.privilege_decoder(priv_latent), student_privileged
        )
        self._aux_optimizer_step(
            losses["privilege_estimation"],
            chain(self.actor.privilege_estimator.parameters(), self.actor.privilege_decoder.parameters()),
        )
        return losses

    def _aux_optimizer_step(
        self, loss: torch.Tensor, params: list[torch.Tensor] | tuple[torch.Tensor, ...] | chain
    ) -> None:
        """Apply one gradient step of the shared auxiliary optimizer."""
        self.aux_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(params), self.max_grad_norm)
        self.aux_optimizer.step()

    def _train_aux_modules(self) -> None:
        """Put the student-side auxiliary modules into train mode."""
        # Skip PPO_CTS._train_aux_modules (it references a plain history encoder).
        PPO._train_aux_modules(self)
        self.actor.privilege_estimator.train()
        self.actor.privilege_decoder.train()
        self.actor.heightmap_decoder.train()
        self.actor.depth_estimator.train()

    def _eval_aux_modules(self) -> None:
        """Put the student-side auxiliary modules into eval mode."""
        # Skip PPO_CTS._eval_aux_modules (it references a plain history encoder).
        PPO._eval_aux_modules(self)
        self.actor.privilege_estimator.eval()
        self.actor.privilege_decoder.eval()
        self.actor.heightmap_decoder.eval()
        self.actor.depth_estimator.eval()

    def _aux_save_state(self) -> dict:
        """Return state-dict entries of auxiliary modules for saving."""
        state = PPO._aux_save_state(self)
        state["cts_depth_aux_optimizer_state_dict"] = self.aux_optimizer.state_dict()
        return state

    def _load_aux_state(self, loaded_dict: dict, load_cfg: dict, strict: bool) -> None:
        """Load state-dict entries of auxiliary modules."""
        PPO._load_aux_state(self, loaded_dict, load_cfg, strict)
        if load_cfg.get("optimizer") and "cts_depth_aux_optimizer_state_dict" in loaded_dict:
            self.aux_optimizer.load_state_dict(loaded_dict["cts_depth_aux_optimizer_state_dict"])

    @staticmethod
    def _create_storage(env: VecEnv, cfg: dict, obs: TensorDict, device: str) -> RolloutStorage:
        """Create a student-only history+depth rollout storage for CTS-depth training."""
        num_teacher = int(getattr(env, "num_teacher", 0))
        return RolloutStorageCTSDepth(
            "rl", env.num_envs, num_teacher, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

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
        # Scale the auxiliary optimizer LR proportionally to the RL LR (relative to its
        # initial value) so the student estimators track the teacher's evolution rate.
        if self.scale_encoder_lr_with_rl:
            scaled_aux_lr = self.encoder_lr * self.learning_rate / self.initial_learning_rate
            for param_group in self.aux_optimizer.param_groups:
                param_group["lr"] = scaled_aux_lr
