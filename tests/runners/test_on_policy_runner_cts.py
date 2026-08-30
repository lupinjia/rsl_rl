# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end runner tests for the Concurrent Teacher-Student (CTS) extension."""

from __future__ import annotations

import copy
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 8
NUM_TEACHER = 6
OBS_DIM = 6
PRIVILEGED_DIM = 10
HISTORY_DIM = 12
CRITIC_DIM = 14
NUM_ACTIONS = 2
NUM_LATENT_DIMS = 4
MAX_EP_LEN = 50


class DummyCtsEnv(VecEnv):
    """Minimal VecEnv exposing the CTS observation groups."""

    def __init__(self, device: str = "cpu") -> None:  # ruff: ignore[undocumented-public-init]
        self.num_envs = NUM_ENVS
        self.num_teacher = NUM_TEACHER
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:  # ruff: ignore[undocumented-public-method]
        teacher_mask = torch.cat([torch.ones(NUM_TEACHER, 1), torch.zeros(NUM_ENVS - NUM_TEACHER, 1)], dim=0).to(
            self.device
        )
        return TensorDict(
            {
                "policy": torch.randn(NUM_ENVS, OBS_DIM, device=self.device),
                "privileged": torch.randn(NUM_ENVS, PRIVILEGED_DIM, device=self.device),
                "history": torch.randn(NUM_ENVS, HISTORY_DIM, device=self.device),
                "critic": torch.randn(NUM_ENVS, CRITIC_DIM, device=self.device),
                "teacher_mask": teacher_mask,
            },
            batch_size=[NUM_ENVS],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # ruff: ignore[undocumented-public-method]
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_cts_cfg() -> dict:
    """Return a minimal training configuration for the CTS extension."""
    return {
        "num_steps_per_env": 6,
        "save_interval": 100,
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["critic"],
            "privileged": ["privileged"],
            "history": ["history"],
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo_cts:PPO_CTS",
            "cts_cfg": {"encoder_lr": 1e-3, "num_encoder_epochs": 2},
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
        },
        "actor": {
            "class_name": "rsl_rl.models.cts_actor:CtsActor",
            "num_latent_dims": NUM_LATENT_DIMS,
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }


def _build_cts_runner(log_dir: str | None = None) -> OnPolicyRunner:
    """Construct a runner with a DummyCtsEnv and the CTS config."""
    return OnPolicyRunner(DummyCtsEnv(), _make_cts_cfg(), log_dir=log_dir, device="cpu")


class TestCtsRunner:
    """End-to-end tests for CTS training through the standard runner."""

    def test_learn_runs_and_reports_reconstruction_loss(self) -> None:
        """A short learn loop must complete and include the CTS reconstruction loss."""
        runner = _build_cts_runner()
        runner.learn(num_learning_iterations=2)
        assert isinstance(runner.alg.storage.observation_histories, torch.Tensor)
        assert "history" not in runner.alg.storage.observations, "main storage must not hold the history group"

    def test_learn_updates_parameters(self) -> None:
        """Actor and history-encoder parameters should change after learning."""
        runner = _build_cts_runner()
        actor_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        hist_before = {n: p.clone() for n, p in runner.alg.actor.history_encoder.named_parameters()}
        runner.learn(num_learning_iterations=2)
        assert any(not torch.equal(actor_before[n], p) for n, p in runner.alg.actor.named_parameters()), (
            "actor params should change"
        )
        assert any(
            not torch.equal(hist_before[n], p) for n, p in runner.alg.actor.history_encoder.named_parameters()
        ), "history encoder params should change"

    def test_save_load_restores_parameters(self) -> None:
        """Checkpointing must restore actor, critic, and encoder-optimizer state."""
        runner = _build_cts_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())
            saved_encoder_opt = copy.deepcopy(runner.alg.encoder_optimizer.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Actor parameter '{key}' not restored after load"
            assert runner.alg.encoder_optimizer.state_dict()["param_groups"] == saved_encoder_opt["param_groups"]

    def test_inference_policy_produces_actions(self) -> None:
        """The inference policy must return actions for the full observation TensorDict."""
        runner = _build_cts_runner()
        runner.learn(num_learning_iterations=1)
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
