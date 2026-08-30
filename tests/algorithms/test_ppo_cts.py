# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Concurrent Teacher-Student (CTS) PPO extension."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_cts import PPO_CTS
from rsl_rl.models import MLPModel
from rsl_rl.models.cts_actor import CtsActor
from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS

NUM_ENVS = 8
NUM_TEACHER = 6
NUM_STEPS = 6
POLICY_DIM = 6
PRIVILEGED_DIM = 10
HISTORY_DIM = 12
CRITIC_DIM = 14
NUM_ACTIONS = 2
NUM_LATENT_DIMS = 4


def _make_obs() -> TensorDict:
    """Create a CTS observation TensorDict with a teacher/student mask."""
    teacher_mask = torch.cat([torch.ones(NUM_TEACHER, 1), torch.zeros(NUM_ENVS - NUM_TEACHER, 1)], dim=0)
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, POLICY_DIM),
            "privileged": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
            "history": torch.randn(NUM_ENVS, HISTORY_DIM),
            "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
            "teacher_mask": teacher_mask,
        },
        batch_size=[NUM_ENVS],
    )


def _obs_groups() -> dict[str, list[str]]:
    """Return the observation-group mapping used by the CTS extension."""
    return {
        "actor": ["policy"],
        "critic": ["critic"],
        "privileged": ["privileged"],
        "history": ["history"],
    }


def _build_cts(schedule: str = "fixed", **overrides: object) -> tuple[PPO_CTS, TensorDict]:
    """Build a PPO_CTS instance with small networks for testing."""
    obs = _make_obs()
    obs_groups = _obs_groups()
    actor = CtsActor(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        num_latent_dims=NUM_LATENT_DIMS,
        hidden_dims=[32, 32],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[32, 32], activation="elu")
    storage = RolloutStorageCTS("rl", NUM_ENVS, NUM_TEACHER, NUM_STEPS, obs, [NUM_ACTIONS])
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule=schedule,
        desired_kl=0.01,
        cts_cfg={"num_encoder_epochs": 2, "encoder_lr": 1e-3},
    )
    defaults.update(overrides)
    alg = PPO_CTS(actor, critic, storage, **defaults)
    return alg, obs


def _rollout(alg: PPO_CTS, obs: TensorDict, steps: int = NUM_STEPS) -> None:
    """Run a rollout filling the storage."""
    for _ in range(steps):
        alg.act(obs)
        alg.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})


class TestCtsStorage:
    """Tests for the student-only history rollout storage."""

    def test_history_excluded_from_main_storage(self) -> None:
        """The main storage must not allocate the history group for teacher environments."""
        obs = _make_obs()
        storage = RolloutStorageCTS("rl", NUM_ENVS, NUM_TEACHER, NUM_STEPS, obs, [NUM_ACTIONS])
        assert "history" not in storage.observations
        assert storage.observation_histories.shape == (NUM_STEPS, NUM_ENVS - NUM_TEACHER, HISTORY_DIM)

    def test_student_history_slice_matches_transition(self) -> None:
        """add_transition stores only the student rows of the history group."""
        alg, obs = _build_cts()
        alg.act(obs)
        _rollout(alg, obs, steps=1)
        assert torch.equal(alg.storage.observation_histories[0], obs["history"][NUM_TEACHER:])

    def test_student_history_index_mapping(self) -> None:
        """get_student_history maps flattened (transition, env) indices to the student buffer."""
        obs = _make_obs()
        storage = RolloutStorageCTS("rl", NUM_ENVS, NUM_TEACHER, NUM_STEPS, obs, [NUM_ACTIONS])
        # Fill two transitions with distinct history values
        for t in range(2):
            step_obs = TensorDict(
                {
                    "policy": torch.randn(NUM_ENVS, POLICY_DIM),
                    "privileged": torch.randn(NUM_ENVS, PRIVILEGED_DIM),
                    "history": torch.randn(NUM_ENVS, HISTORY_DIM),
                    "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
                    "teacher_mask": torch.cat(
                        [torch.ones(NUM_TEACHER, 1), torch.zeros(NUM_ENVS - NUM_TEACHER, 1)], dim=0
                    ),
                },
                batch_size=[NUM_ENVS],
            )
            trans = RolloutStorageCTS.Transition()
            trans.observations = step_obs
            trans.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            trans.rewards = torch.randn(NUM_ENVS)
            trans.dones = torch.zeros(NUM_ENVS)
            trans.values = torch.randn(NUM_ENVS, 1)
            trans.actions_log_prob = torch.randn(NUM_ENVS)
            trans.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.rand(NUM_ENVS, NUM_ACTIONS))
            storage.add_transition(trans)
        for t in range(2):
            for s in range(NUM_ENVS - NUM_TEACHER):
                flat_idx = t * NUM_ENVS + (NUM_TEACHER + s)
                got = storage.get_student_history(torch.tensor([flat_idx]))[0]
                assert torch.equal(got, storage.observation_histories[t, s])


class TestCtsAdvantages:
    """Tests for per-group advantage normalization."""

    def test_teacher_and_student_advantages_normalized_independently(self) -> None:
        """Each group should be zero-mean/unit-std on its own after compute_returns."""
        alg, obs = _build_cts()
        _rollout(alg, obs)
        alg.compute_returns(obs)

        teacher_adv = alg.storage.advantages[:, :NUM_TEACHER].flatten()
        student_adv = alg.storage.advantages[:, NUM_TEACHER:].flatten()
        assert abs(teacher_adv.mean().item()) < 1e-5
        assert abs(teacher_adv.std().item() - 1.0) < 0.1
        assert abs(student_adv.mean().item()) < 1e-5
        assert abs(student_adv.std().item() - 1.0) < 0.1


class TestCtsUpdate:
    """Tests for the CTS training update."""

    def test_update_runs_and_changes_parameters(self) -> None:
        """A full rollout + update must produce finite losses and move the parameters."""
        alg, obs = _build_cts()
        alg.train_mode()
        _rollout(alg, obs)
        alg.compute_returns(obs)

        before = {name: p.clone() for name, p in alg.actor.named_parameters()}
        loss_dict = alg.update()

        assert "cts/reconstruction" in loss_dict
        assert torch.isfinite(torch.tensor(loss_dict["cts/reconstruction"]))
        assert torch.isfinite(torch.tensor(loss_dict["surrogate"]))
        assert torch.isfinite(torch.tensor(loss_dict["value"]))
        assert any(not torch.equal(before[name], p) for name, p in alg.actor.named_parameters()), (
            "actor params should change"
        )

    def test_history_encoder_trained_only_by_reconstruction(self) -> None:
        """The history encoder must update from the reconstruction loss even with zero history."""
        alg, obs = _build_cts()
        alg.train_mode()
        _rollout(alg, obs)
        alg.compute_returns(obs)

        hist_before = {name: p.clone() for name, p in alg.actor.history_encoder.named_parameters()}
        # Zero out all stored student histories; the encoder must still update toward the
        # privilege-encoder targets via the reconstruction loss.
        with torch.no_grad():
            for t in range(NUM_STEPS):
                alg.storage.observation_histories[t] = torch.zeros_like(alg.storage.observation_histories[t])
        alg.update()
        hist_after = {name: p.clone() for name, p in alg.actor.history_encoder.named_parameters()}
        assert any(not torch.equal(hist_before[name], hist_after[name]) for name in hist_before), (
            "history encoder should update from the reconstruction loss"
        )

    def test_reconstruction_trained_in_post_rl_phase(self) -> None:
        """Reconstruction loss is trained in a post-RL phase.

        The RL mini-batch loop must not compute the reconstruction loss; it runs in a
        dedicated post-RL phase, mirroring the original two-phase update.
        """
        alg, obs = _build_cts()
        alg.train_mode()
        _rollout(alg, obs)
        alg.compute_returns(obs)

        # The per-mini-batch auxiliary loss must not contain the reconstruction metric.
        generator = alg.storage.mini_batch_generator(alg.num_mini_batches, alg.num_learning_epochs)
        _, metrics = alg._compute_aux_loss(next(iter(generator)), None)
        assert "cts/reconstruction" not in metrics, "reconstruction must not be in the RL loop"

        # The encoder must not change from the RL mini-batch loop alone: run only the
        # policy/aux backward+step pass and check the encoder stays frozen.
        encoder_lr_before = alg.encoder_optimizer.param_groups[0]["lr"]
        hist_before = {name: p.clone() for name, p in alg.actor.history_encoder.named_parameters()}
        for batch, (_disc_obs, _disc_demo) in zip(generator, alg._extra_mini_batch_iter()):
            surrogate_loss, _ = alg._compute_policy_loss(batch, batch.observations.batch_size[0])
            aux_loss, _ = alg._compute_aux_loss(batch, None)
            alg.optimizer.zero_grad()
            surrogate_loss.backward()
            alg._compute_aux_gradients(aux_loss)
            alg.optimizer.step()
            alg._step_aux_optimizers()
        hist_after = {name: p.clone() for name, p in alg.actor.history_encoder.named_parameters()}
        assert all(torch.equal(hist_before[name], hist_after[name]) for name in hist_before), (
            "history encoder must stay frozen during the RL mini-batch loop"
        )
        assert alg.encoder_optimizer.param_groups[0]["lr"] == encoder_lr_before


class TestCtsEncoderLearningRate:
    """Tests for the history-encoder learning-rate scaling."""

    def test_encoder_lr_scales_with_rl_lr(self) -> None:
        """When scale_encoder_lr_with_rl is on, the encoder LR tracks the RL LR ratio."""
        alg, _ = _build_cts(schedule="adaptive", learning_rate=1e-3)
        alg._adjust_learning_rate(torch.tensor(0.05))  # KL > 2*desired_kl -> LR / 1.5
        expected_encoder_lr = alg.encoder_lr * alg.learning_rate / alg.initial_learning_rate
        actual_encoder_lr = alg.encoder_optimizer.param_groups[0]["lr"]
        assert alg.learning_rate < alg.initial_learning_rate, "RL LR should have decreased"
        assert abs(actual_encoder_lr - expected_encoder_lr) < 1e-12, (
            f"encoder LR {actual_encoder_lr} != expected {expected_encoder_lr}"
        )

    def test_encoder_lr_fixed_when_scaling_disabled(self) -> None:
        """With scale_encoder_lr_with_rl=False, the encoder LR stays at its initial value."""
        alg, _ = _build_cts(
            schedule="adaptive", cts_cfg={"scale_encoder_lr_with_rl": False}
        )
        alg._adjust_learning_rate(torch.tensor(0.05))
        assert alg.learning_rate < alg.initial_learning_rate, "RL LR should have decreased"
        assert abs(alg.encoder_optimizer.param_groups[0]["lr"] - alg.encoder_lr) < 1e-12, (
            "encoder LR must stay fixed when scaling is disabled"
        )


class TestCtsExport:
    """Tests for student-mode model export."""

    def test_inference_uses_student_encoder(self) -> None:
        """In eval mode the actor must use the history encoder regardless of the teacher mask."""
        alg, obs = _build_cts()
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
            out_t_eval = alg.actor(obs_teacher)
            out_s_eval = alg.actor(obs_student)
            assert torch.allclose(out_t_eval, out_s_eval), "inference must ignore the teacher mask"
            assert torch.allclose(out_t_eval, out_s), "inference must use the student (history) encoder"

    def test_jit_export_runs_student_mode(self) -> None:
        """The JIT export must use the history encoder for all inputs."""
        alg, _ = _build_cts()
        jit_model = alg.get_policy().as_jit()
        input_tensor = torch.randn(2, POLICY_DIM + HISTORY_DIM)
        out = jit_model(input_tensor)
        assert out.shape == (2, NUM_ACTIONS)
        assert torch.isfinite(out).all()

    def test_onnx_export_runs(self) -> None:
        """The ONNX export must trace successfully with dummy inputs."""
        alg, _ = _build_cts()
        onnx_model = alg.get_policy().as_onnx(verbose=False)
        assert onnx_model.input_size == POLICY_DIM + HISTORY_DIM
        dummy = onnx_model.get_dummy_inputs()[0]
        out = onnx_model(dummy)
        assert out.shape == (1, NUM_ACTIONS)
