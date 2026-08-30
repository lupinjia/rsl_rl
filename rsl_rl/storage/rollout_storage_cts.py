# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for Concurrent Teacher-Student (CTS) training."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class RolloutStorageCTS(RolloutStorage):
    """Rollout storage for Concurrent Teacher-Student (CTS) training.

    The cheap per-step observation groups (policy, privileged, critic, teacher_mask, ...)
    are stored for all environments in the base storage. The (potentially large)
    observation history is kept in a dedicated student-only buffer of shape
    ``(num_transitions_per_env, num_students, *history_shape)`` so that teacher
    environments never allocate history memory.

    The feedforward mini-batch generator augments each :class:`Batch` with an ``indices``
    field (flattened samples over ``(transition, env)``) from which extensions can recover
    the per-sample history via :meth:`get_student_history`.
    """

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
        """Initialize the CTS storage with a student-only history buffer."""
        if "history" not in obs:
            raise ValueError("RolloutStorageCTS requires a 'history' observation group in the obs TensorDict.")
        self.num_teacher = num_teacher
        self.num_students = num_envs - num_teacher
        if self.num_students <= 0:
            raise ValueError(f"RolloutStorageCTS requires num_teacher < num_envs, got num_teacher={num_teacher}.")
        history_shape = obs["history"].shape[1:]
        # Strip the history group so the base storage does not allocate it for all envs
        main_obs = obs.exclude("history")
        super().__init__(training_type, num_envs, num_transitions_per_env, main_obs, actions_shape, device)
        self.observation_histories = torch.zeros(
            num_transitions_per_env,
            self.num_students,
            *history_shape,
            dtype=obs["history"].dtype,
            device=device,
        )

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Store one transition, keeping only the student history slice."""
        history = transition.observations["history"]
        self.observation_histories[self.step] = history[self.num_teacher :]
        transition.observations = transition.observations.exclude("history")
        super().add_transition(transition)

    def get_student_history(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the history for student samples given flattened ``(transition, env)`` indices.

        ``indices`` are the flattened sample indices produced by :meth:`mini_batch_generator`,
        i.e. ``idx = t * num_envs + env`` with ``env >= num_teacher``.
        """
        t = indices // self.num_envs
        env = indices % self.num_envs
        student_buffer_indices = t * self.num_students + (env - self.num_teacher)
        return self.observation_histories.flatten(0, 1)[student_buffer_indices]
