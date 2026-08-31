# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end runner tests for the depth-augmented CTS extension."""

from __future__ import annotations

import copy
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 8
NUM_TEACHER = 6
NUM_STUDENTS = NUM_ENVS - NUM_TEACHER
OBS_DIM = 6
PRIVILEGED_DIM = 10
HISTORY_DIM = 12
CRITIC_DIM = 14
NUM_ACTIONS = 2
DEPTH_SHAPE = (2, 6, 8)
HEIGHTMAP_SHAPE = (1, 4, 3)
MAX_EP_LEN = 50


class DummyCtsDepthEnv(VecEnv):
    """Minimal VecEnv exposing the CTS-depth observation groups."""

    def __init__(self, device: str = "cpu") -> None:  # ruff: ignore[undocumented-public-init]
        self.num_envs = NUM_ENVS
        self.num_teacher = NUM_TEACHER
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:  # ruff: ignore[undocumented-public-method]
        teacher_mask = torch.cat([torch.ones(NUM_TEACHER, 1), torch.zeros(NUM_STUDENTS, 1)], dim=0).to(self.device)
        return TensorDict(
            {
                "actor": torch.randn(NUM_ENVS, OBS_DIM, device=self.device),
                "privileged": torch.randn(NUM_ENVS, PRIVILEGED_DIM, device=self.device),
                "history": torch.randn(NUM_ENVS, HISTORY_DIM, device=self.device),
                "depth_image": torch.randn(NUM_ENVS, *DEPTH_SHAPE, device=self.device),
                "heightmap": torch.randn(NUM_ENVS, *HEIGHTMAP_SHAPE, device=self.device),
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


def _make_cts_depth_cfg() -> dict:
    """Return a minimal training configuration for the CTS-depth extension."""
    return {
        "num_steps_per_env": 6,
        "save_interval": 100,
        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"],
            "privileged": ["privileged"],
            "history": ["history"],
            "heightmap": ["heightmap"],
            "depth_image": ["depth_image"],
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.ppo_cts_depth:PPO_CTSDepth",
            "cts_cfg": {"encoder_lr": 1e-3, "num_encoder_epochs": 1},
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
        },
        "actor": {
            "class_name": "rsl_rl.models.cts_depth_actor:CtsDepthActor",
            "num_privilege_latent_dims": 4,
            "num_heightmap_latent_dims": 6,
            "privilege_encoder_hidden_dims": [8],
            "privilege_estimator_hidden_dims": [8],
            "privilege_decoder_hidden_dims": [8],
            "heightmap_decoder_hidden_dims": [8],
            "heightmap_encoder_fc_layer_dims": [8],
            "depth_cnn_fc_layer_dims": [8],
            "history_mlp_dims": [8],
            "rnn_hidden_dim": 16,
            "hidden_dims": [16, 16],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "activation": "elu",
        },
    }


def _build_runner(log_dir: str | None = None) -> OnPolicyRunner:
    """Construct a runner with a DummyCtsDepthEnv and the CTS-depth config."""
    return OnPolicyRunner(DummyCtsDepthEnv(), _make_cts_depth_cfg(), log_dir=log_dir, device="cpu")


class TestCtsDepthRunner:
    """End-to-end tests for CTS-depth training through the standard runner."""

    def test_learn_runs_and_reports_aux_losses(self) -> None:
        """A short learn loop must complete and store depth student-only."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)
        assert isinstance(runner.alg.storage.observation_depths, torch.Tensor)
        assert "depth_image" not in runner.alg.storage.observations
        assert runner.alg.storage.observation_depths.shape == (6, NUM_STUDENTS, *DEPTH_SHAPE)

    def test_learn_updates_student_networks(self) -> None:
        """The depth estimator and privilege estimator should change after learning."""
        runner = _build_runner()
        before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        depth_changed = any(
            not torch.equal(before[n], p)
            for n, p in runner.alg.actor.named_parameters()
            if n.startswith("depth_estimator")
        )
        priv_changed = any(
            not torch.equal(before[n], p)
            for n, p in runner.alg.actor.named_parameters()
            if n.startswith("privilege_estimator")
        )
        assert depth_changed and priv_changed, "student networks must change after learning"

    def test_save_load_restores_aux_optimizer(self) -> None:
        """Checkpointing must restore the auxiliary optimizer state."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())
            saved_aux_opt = copy.deepcopy(runner.alg.aux_optimizer.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Actor parameter '{key}' not restored after load"
            assert runner.alg.aux_optimizer.state_dict()["param_groups"] == saved_aux_opt["param_groups"]

    def test_inference_policy_produces_actions(self) -> None:
        """The inference policy must return actions for the full observation TensorDict."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
