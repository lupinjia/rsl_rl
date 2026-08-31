# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the depth-augmented Concurrent Teacher-Student (CTS) extension."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_cts_depth import PPO_CTSDepth
from rsl_rl.models.cts_depth_actor import CtsDepthActor
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage.rollout_storage_cts_depth import RolloutStorageCTSDepth

NUM_ENVS = 8
NUM_TEACHER = 5
NUM_STUDENTS = NUM_ENVS - NUM_TEACHER
NUM_STEPS = 4
OBS_DIM = 6
PRIVILEGED_DIM = 10
HISTORY_DIM = 12
CRITIC_DIM = 14
NUM_ACTIONS = 2
DEPTH_SHAPE = (2, 6, 8)  # C, H, W
HEIGHTMAP_SHAPE = (1, 4, 3)
NUM_PRIV_LATENT = 4
NUM_HM_LATENT = 6

OBS_GROUPS = {
    "actor": ["actor"],
    "critic": ["critic"],
    "privileged": ["privileged"],
    "history": ["history"],
    "heightmap": ["heightmap"],
    "depth_image": ["depth_image"],
}


def _make_obs(num_envs: int = NUM_ENVS) -> TensorDict:
    """Create a CTS-depth observation TensorDict."""
    return TensorDict(
        {
            "actor": torch.randn(num_envs, OBS_DIM),
            "privileged": torch.randn(num_envs, PRIVILEGED_DIM),
            "history": torch.randn(num_envs, HISTORY_DIM),
            "depth_image": torch.randn(num_envs, *DEPTH_SHAPE),
            "heightmap": torch.randn(num_envs, *HEIGHTMAP_SHAPE),
            "critic": torch.randn(num_envs, CRITIC_DIM),
            "teacher_mask": torch.cat(
                [torch.ones(NUM_TEACHER, 1), torch.zeros(num_envs - NUM_TEACHER, 1)], dim=0
            ),
        },
        batch_size=[num_envs],
    )


def _make_actor(obs: TensorDict) -> CtsDepthActor:
    """Create a small CtsDepthActor."""
    return CtsDepthActor(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS,
        num_privilege_latent_dims=NUM_PRIV_LATENT,
        num_heightmap_latent_dims=NUM_HM_LATENT,
        privilege_encoder_hidden_dims=(8,),
        privilege_estimator_hidden_dims=(8,),
        privilege_decoder_hidden_dims=(8,),
        heightmap_decoder_hidden_dims=(8,),
        heightmap_encoder_fc_layer_dims=(8,),
        depth_cnn_fc_layer_dims=(8,),
        history_mlp_dims=(8,),
        rnn_hidden_dim=16,
        hidden_dims=(16, 16),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def _build_ctsdepth(overrides: dict | None = None) -> tuple[PPO_CTSDepth, TensorDict]:
    """Build a PPO_CTSDepth instance with an empty storage (fill via _rollout)."""
    obs = _make_obs()
    actor = _make_actor(obs)
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16, 16], activation="elu")
    storage = RolloutStorageCTSDepth("rl", NUM_ENVS, NUM_TEACHER, NUM_STEPS, obs, [NUM_ACTIONS])
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        cts_cfg={"encoder_lr": 5e-4, "num_encoder_epochs": 1},
    )
    if overrides:
        defaults.update(overrides)
    alg = PPO_CTSDepth(actor, critic, storage, **defaults)
    return alg, obs


def _rollout(alg: PPO_CTSDepth, obs: TensorDict, steps: int = NUM_STEPS) -> None:
    """Fill the storage through the algorithm's rollout interface."""
    for _ in range(steps):
        alg.act(obs)
        alg.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})


class TestCtsDepthStorage:
    """Tests for the depth student-only storage."""

    def test_depth_stored_student_only(self) -> None:
        """The main storage must not hold the depth group; only the student buffer does."""
        obs = _make_obs()
        storage = RolloutStorageCTSDepth("rl", NUM_ENVS, NUM_TEACHER, NUM_STEPS, obs, [NUM_ACTIONS])
        assert "depth_image" not in storage.observations
        assert storage.observation_depths.shape == (NUM_STEPS, NUM_STUDENTS, *DEPTH_SHAPE)
        assert storage.observation_histories.shape == (NUM_STEPS, NUM_STUDENTS, HISTORY_DIM)

    def test_student_depth_slice_matches_transition(self) -> None:
        """add_transition stores only the student rows of the depth group."""
        obs = _make_obs()
        storage = RolloutStorageCTSDepth("rl", NUM_ENVS, NUM_TEACHER, NUM_STEPS, obs, [NUM_ACTIONS])
        for t in range(2):
            step_obs = _make_obs()
            trans = RolloutStorageCTSDepth.Transition()
            trans.observations = step_obs
            trans.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            trans.rewards = torch.randn(NUM_ENVS)
            trans.dones = torch.zeros(NUM_ENVS)
            trans.values = torch.randn(NUM_ENVS, 1)
            trans.actions_log_prob = torch.randn(NUM_ENVS)
            trans.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.rand(NUM_ENVS, NUM_ACTIONS))
            storage.add_transition(trans)
            for s in range(NUM_STUDENTS):
                assert torch.equal(storage.observation_depths[t, s], step_obs["depth_image"][NUM_TEACHER + s])

    def test_recurrent_generator_aligns_flat_and_padded(self) -> None:
        """The recurrent generator's flat student rows match the padded valid rows."""
        alg, obs = _build_ctsdepth()
        alg.train_mode()
        _rollout(alg, obs)
        alg.compute_returns(obs)
        generator = alg.storage.recurrent_mini_batch_generator(alg.num_mini_batches, 1)
        for batch in generator:
            n_flat = batch.observations.batch_size[0]
            n_teacher = int(batch.observations["teacher_mask"].sum())
            assert n_flat - n_teacher == int(batch.student_masks.sum()), "flat/padded student mismatch"
            assert "history" in batch.student_observations
            assert "depth_image" in batch.student_observations
            assert batch.student_observations["depth_image"].shape[-3:] == DEPTH_SHAPE


class TestCtsDepthUpdate:
    """Tests for the CTS-depth training update."""

    def test_update_runs_and_reports_all_losses(self) -> None:
        """A full rollout + update must produce all six finite losses."""
        alg, obs = _build_ctsdepth()
        alg.train_mode()
        _rollout(alg, obs)
        alg.compute_returns(obs)

        loss_dict = alg.update()
        for key in (
            "surrogate",
            "value",
            "cts_depth/heightmap_recon",
            "cts_depth/heightmap_estimation",
            "cts_depth/privilege_recon",
            "cts_depth/privilege_estimation",
        ):
            assert key in loss_dict, f"missing loss key {key}"
            assert torch.isfinite(torch.tensor(loss_dict[key])), f"loss {key} is not finite"

    def test_student_networks_trained_by_aux_losses(self) -> None:
        """All student-side networks must change after a full update."""
        alg, obs = _build_ctsdepth()
        alg.train_mode()
        _rollout(alg, obs)
        alg.compute_returns(obs)

        before = {n: p.clone() for n, p in alg.actor.named_parameters()}
        alg.update()
        for module_name in (
            "privilege_estimator",
            "privilege_decoder",
            "heightmap_decoder",
            "depth_estimator",
        ):
            changed = any(
                not torch.equal(before[n], p)
                for n, p in alg.actor.named_parameters()
                if n.startswith(module_name)
            )
            assert changed, f"{module_name} did not change after update"

    def test_aux_optimizer_lr_scales_with_rl(self) -> None:
        """The auxiliary optimizer LR tracks the adaptive RL LR."""
        alg, _ = _build_ctsdepth({"schedule": "adaptive", "learning_rate": 1e-3})
        alg._adjust_learning_rate(torch.tensor(0.05))
        expected = alg.encoder_lr * alg.learning_rate / alg.initial_learning_rate
        assert abs(alg.aux_optimizer.param_groups[0]["lr"] - expected) < 1e-12


class TestCtsDepthInference:
    """Tests for the student-mode inference path."""

    def test_eval_mode_uses_student_for_all(self) -> None:
        """In eval mode the actor must use the student path regardless of the teacher mask."""
        alg, obs = _build_ctsdepth()
        obs_teacher = obs.clone()
        obs_teacher["teacher_mask"] = torch.ones(NUM_ENVS, 1)
        obs_student = obs.clone()
        obs_student["teacher_mask"] = torch.zeros(NUM_ENVS, 1)
        with torch.no_grad():
            alg.train_mode()
            out_t = alg.actor(obs_teacher)
            out_s = alg.actor(obs_student)
            assert not torch.allclose(out_t, out_s), "train mode must dispatch different encoders by mask"
            alg.eval_mode()
            alg.actor.reset()  # fresh GRU state so both calls see identical inputs
            out_t_eval = alg.actor(obs_teacher)
            alg.actor.reset()
            out_s_eval = alg.actor(obs_student)
            assert torch.allclose(out_t_eval, out_s_eval), "eval mode must use the student path for all envs"

    def test_jit_export_produces_actions(self) -> None:
        """The student-mode JIT export must run on concatenated (obs, history, depth)."""
        alg, _ = _build_ctsdepth()
        alg.eval_mode()
        jit_mod = alg.actor.as_jit()
        flat = torch.cat(
            [
                torch.randn(1, OBS_DIM),
                torch.randn(1, HISTORY_DIM),
                torch.randn(1, DEPTH_SHAPE[0] * DEPTH_SHAPE[1] * DEPTH_SHAPE[2]),
            ],
            dim=-1,
        )
        out = jit_mod(flat)
        assert out.shape == (1, NUM_ACTIONS)
