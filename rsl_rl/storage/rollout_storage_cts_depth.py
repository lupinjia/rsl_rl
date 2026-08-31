# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for depth-augmented Concurrent Teacher-Student (CTS) training."""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS
from rsl_rl.utils.utils import split_and_pad_trajectories, unpad_trajectories_flat


class RolloutStorageCTSDepth(RolloutStorageCTS):
    """Rollout storage for depth-augmented Concurrent Teacher-Student training.

    On top of :class:`RolloutStorageCTS`, the (potentially large) depth images are
    kept in a dedicated student-only buffer of shape
    ``(num_transitions_per_env, num_students, C, H, W)`` so that teacher environments
    never allocate depth memory.

    The recurrent mini-batch generator supports the student GRU: student transitions
    are split into trajectories (padded to the longest episode) while the teacher
    transitions are kept flat. The generated :class:`Batch` carries the flat
    per-sample fields for all samples (teacher first, then student) plus dedicated
    ``student_*`` padded trajectory fields for the recurrent student forward pass.
    """

    class Batch(RolloutStorage.Batch):
        """A recurrent training batch with separate teacher/student data.

        The standard fields (``observations``, ``actions``, ``values``, ...) hold
        FLAT per-sample data ordered as [teacher samples, student samples]. The
        ``student_*`` fields hold the same student samples in PADDED trajectory
        order, aligned with ``student_masks`` and ``student_hidden_states``.
        """

        def __init__(
            self,
            student_observations: TensorDict | None = None,
            student_actions: torch.Tensor | None = None,
            student_old_actions_log_prob: torch.Tensor | None = None,
            student_advantages: torch.Tensor | None = None,
            student_old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            student_masks: torch.Tensor | None = None,
            student_hidden_states: torch.Tensor | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize a CTS-depth recurrent batch."""
            super().__init__(**kwargs)
            self.student_observations = student_observations
            self.student_actions = student_actions
            self.student_old_actions_log_prob = student_old_actions_log_prob
            self.student_advantages = student_advantages
            self.student_old_distribution_params = student_old_distribution_params
            self.student_masks = student_masks
            self.student_hidden_states = student_hidden_states

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_teacher: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Initialize the storage with student-only history and depth buffers."""
        if "depth_image" not in obs:
            raise ValueError("RolloutStorageCTSDepth requires a 'depth_image' observation group.")
        depth_shape = obs["depth_image"].shape[1:]
        depth_dtype = obs["depth_image"].dtype
        # Strip the depth group so the base storage does not allocate it for all envs.
        obs = obs.exclude("depth_image")
        super().__init__(training_type, num_envs, num_teacher, num_transitions_per_env, obs, actions_shape, device)
        self.observation_depths = torch.zeros(
            num_transitions_per_env,
            self.num_students,
            *depth_shape,
            dtype=depth_dtype,
            device=device,
        )

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Store one transition, keeping only the student depth slice."""
        depth = transition.observations["depth_image"]
        self.observation_depths[self.step] = depth[self.num_teacher :]
        transition.observations = transition.observations.exclude("depth_image")
        super().add_transition(transition)

    def get_student_depth(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the depth for student samples given flattened ``(transition, env)`` indices.

        ``indices`` are the flattened sample indices produced by :meth:`mini_batch_generator`,
        i.e. ``idx = t * num_envs + env`` with ``env >= num_teacher``.
        """
        t = indices // self.num_envs
        env = indices % self.num_envs
        student_buffer_indices = t * self.num_students + (env - self.num_teacher)
        return self.observation_depths.flatten(0, 1)[student_buffer_indices]

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield CTS-depth mini-batches with flat teacher and padded student data.

        The teacher environments ``[0, num_teacher)`` are sampled as flat random
        samples; the student environments ``[num_teacher, num_envs)`` are split
        into trajectories and padded to the longest episode (the recurrent GRU
        path). Flat per-sample fields are ordered [teacher samples, student samples].
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        teacher_slice = slice(0, self.num_teacher)
        student_slice = slice(self.num_teacher, self.num_envs)
        student_dones = self.dones[:, student_slice]

        padded_student_obs, trajectory_masks = split_and_pad_trajectories(
            self.observations[:, student_slice], student_dones
        )
        padded_student_history = split_and_pad_trajectories(self.observation_histories, student_dones)[0]
        padded_student_depth = split_and_pad_trajectories(self.observation_depths, student_dones)[0]
        padded_student_actions = split_and_pad_trajectories(self.actions[:, student_slice], student_dones)[0]
        padded_student_log_prob = split_and_pad_trajectories(
            self.actions_log_prob[:, student_slice], student_dones
        )[0]
        padded_student_advantages = split_and_pad_trajectories(self.advantages[:, student_slice], student_dones)[0]
        padded_student_values = split_and_pad_trajectories(self.values[:, student_slice], student_dones)[0]
        padded_student_returns = split_and_pad_trajectories(self.returns[:, student_slice], student_dones)[0]
        padded_student_dones = split_and_pad_trajectories(self.dones[:, student_slice], student_dones)[0]
        padded_student_dist_params = tuple(
            split_and_pad_trajectories(p[:, student_slice], student_dones)[0] for p in self.distribution_params
        )

        teacher_batch_size = self.num_teacher * self.num_transitions_per_env // num_mini_batches
        teacher_indices = torch.randperm(num_mini_batches * teacher_batch_size, device=self.device)
        student_mini_batch_size = self.num_students // num_mini_batches

        last_was_done = torch.zeros_like(student_dones.squeeze(-1), dtype=torch.bool)
        last_was_done[1:] = student_dones[:-1].squeeze(-1)
        last_was_done[0] = True

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                teacher_idx = teacher_indices[i * teacher_batch_size : (i + 1) * teacher_batch_size]
                teacher_obs = self.observations[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_actions = self.actions[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_log_prob = self.actions_log_prob[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_advantages = self.advantages[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_values = self.values[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_returns = self.returns[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_dones = self.dones[:, teacher_slice].flatten(0, 1)[teacher_idx]
                teacher_dist_params = tuple(
                    p[:, teacher_slice].flatten(0, 1)[teacher_idx] for p in self.distribution_params
                )

                start = i * student_mini_batch_size
                stop = (i + 1) * student_mini_batch_size
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                student_masks = trajectory_masks[:, first_traj:last_traj]
                student_obs_padded = padded_student_obs[:, first_traj:last_traj]
                student_history_padded = padded_student_history[:, first_traj:last_traj]
                student_depth_padded = padded_student_depth[:, first_traj:last_traj]
                student_actions_padded = padded_student_actions[:, first_traj:last_traj]
                student_log_prob_padded = padded_student_log_prob[:, first_traj:last_traj]
                student_advantages_padded = padded_student_advantages[:, first_traj:last_traj]
                student_values_padded = padded_student_values[:, first_traj:last_traj]
                student_returns_padded = padded_student_returns[:, first_traj:last_traj]
                student_dones_padded = padded_student_dones[:, first_traj:last_traj]
                student_dist_params_padded = tuple(p[:, first_traj:last_traj] for p in padded_student_dist_params)

                hidden_state_a_batch = None
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state
                        .permute(2, 0, 1, 3)[last_was_done.permute(1, 0)][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    hidden_state_a_batch = (
                        hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else tuple(hidden_state_a_batch)
                    )

                student_flat_obs = unpad_trajectories_flat(student_obs_padded, student_masks)
                student_flat_actions = unpad_trajectories_flat(student_actions_padded, student_masks)
                student_flat_log_prob = unpad_trajectories_flat(student_log_prob_padded, student_masks)
                student_flat_advantages = unpad_trajectories_flat(student_advantages_padded, student_masks)
                student_flat_values = unpad_trajectories_flat(student_values_padded, student_masks)
                student_flat_returns = unpad_trajectories_flat(student_returns_padded, student_masks)
                student_flat_dones = unpad_trajectories_flat(student_dones_padded, student_masks)
                student_flat_dist_params = tuple(
                    unpad_trajectories_flat(p, student_masks) for p in student_dist_params_padded
                )

                obs = torch.cat([teacher_obs, student_flat_obs], dim=0)
                actions = torch.cat([teacher_actions, student_flat_actions], dim=0)
                log_prob = torch.cat([teacher_log_prob, student_flat_log_prob], dim=0)
                advantages = torch.cat([teacher_advantages, student_flat_advantages], dim=0)
                values = torch.cat([teacher_values, student_flat_values], dim=0)
                returns = torch.cat([teacher_returns, student_flat_returns], dim=0)
                dones = torch.cat([teacher_dones, student_flat_dones], dim=0)
                dist_params = tuple(
                    torch.cat([t_p, s_p], dim=0)
                    for t_p, s_p in zip(teacher_dist_params, student_flat_dist_params)
                )

                assert isinstance(student_obs_padded, TensorDict)
                student_obs_full = TensorDict(
                    {
                        **student_obs_padded.to_dict(),
                        "history": student_history_padded,
                        "depth_image": student_depth_padded,
                    },
                    batch_size=student_obs_padded.shape,
                )

                yield RolloutStorageCTSDepth.Batch(
                    observations=obs,
                    actions=actions,
                    values=values,
                    returns=returns,
                    old_actions_log_prob=log_prob,
                    old_distribution_params=dist_params,
                    advantages=advantages,
                    dones=dones,
                    student_observations=student_obs_full,
                    student_actions=student_actions_padded,
                    student_old_actions_log_prob=student_log_prob_padded,
                    student_advantages=student_advantages_padded,
                    student_old_distribution_params=student_dist_params_padded,
                    student_masks=student_masks,
                    student_hidden_states=hidden_state_a_batch,
                )

                first_traj = last_traj
