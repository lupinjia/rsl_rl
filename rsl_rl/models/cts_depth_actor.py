# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor network for depth-augmented Concurrent Teacher-Student (CTS) training."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, MLP, HiddenState
from rsl_rl.modules.heightmap_estimator import HeightmapEstimator
from rsl_rl.utils.utils import unpad_trajectories_flat


class _HeightmapEncoder(nn.Module):
    """Teacher heightmap encoder: CNN backbone + MLP head to the heightmap latent."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_latent_dims: int,
        channel_dims: tuple[int, ...] | list[int],
        kernel_sizes: tuple[int, ...] | list[int],
        fc_layer_dims: tuple[int, ...] | list[int],
        strides: tuple[int, ...] | list[int] = (1,),
        activation: str = "elu",
    ) -> None:
        """Initialize the heightmap encoder."""
        super().__init__()
        self.cnn = CNN(
            input_dim=(input_shape[1], input_shape[2]),
            input_channels=input_shape[0],
            output_channels=list(channel_dims),
            kernel_size=list(kernel_sizes),
            stride=list(strides),
            padding="none",
            activation=activation,
            flatten=True,
        )
        fc_layers: list[nn.Module] = []
        fc_in = self.cnn.output_dim  # type: ignore[assignment]
        for dim in fc_layer_dims:
            fc_layers.append(nn.Linear(fc_in, dim))
            fc_layers.append(nn.ELU())
            fc_in = dim
        fc_layers.append(nn.Linear(fc_in, num_latent_dims))
        self.mlp = nn.Sequential(*fc_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Encode a heightmap grid ``[B, C, H, W]`` into a flat latent."""
        return self.mlp(self.cnn(input))


class CtsDepthActor(MLPModel):
    """Actor for depth-augmented Concurrent Teacher-Student training.

    Teacher environments encode privileged observations with :attr:`privilege_encoder`
    and the terrain heightmap with :attr:`heightmap_encoder`. Student environments
    encode the observation history with :attr:`privilege_estimator` and the stacked
    depth images (plus history) with the recurrent :attr:`depth_estimator`. All paths
    share a single policy MLP. The student decoders (:attr:`privilege_decoder`,
    :attr:`heightmap_decoder`) drive the auxiliary reconstruction/estimation losses.

    The observation TensorDict must provide ``teacher_mask`` (1 for teacher), plus the
    ``privileged``, ``history``, ``heightmap`` and ``depth_image`` groups.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_privilege_latent_dims: int,
        num_heightmap_latent_dims: int,
        privilege_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        privilege_estimator_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        privilege_decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
        heightmap_decoder_hidden_dims: tuple[int, ...] | list[int] = (128, 256),
        heightmap_encoder_channel_dims: tuple[int, ...] | list[int] = (4,),
        heightmap_encoder_kernel_sizes: tuple[int, ...] | list[int] = (2,),
        heightmap_encoder_fc_layer_dims: tuple[int, ...] | list[int] = (128, 64),
        heightmap_encoder_strides: tuple[int, ...] | list[int] = (1,),
        depth_cnn_channel_dims: tuple[int, ...] | list[int] = (4,),
        depth_cnn_kernel_sizes: tuple[int, ...] | list[int] = (3,),
        depth_cnn_strides: tuple[int, ...] | list[int] = (1,),
        depth_cnn_fc_layer_dims: tuple[int, ...] | list[int] = (128, 64),
        history_mlp_dims: tuple[int, ...] | list[int] = (256, 128),
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the depth-aware CTS actor."""
        self._num_privilege_latent_dims = num_privilege_latent_dims
        self._num_heightmap_latent_dims = num_heightmap_latent_dims
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        for set_name in ("privileged", "history", "heightmap", "depth_image"):
            if set_name not in obs_groups:
                raise ValueError(f"CtsDepthActor requires an '{set_name}' entry in obs_groups.")

        self.privileged_groups, privileged_dim = self._get_obs_dim(obs, obs_groups, "privileged")
        self.history_groups, self.history_dim = self._get_obs_dim(obs, obs_groups, "history")
        heightmap_group = obs_groups["heightmap"][0]
        depth_group = obs_groups["depth_image"][0]
        heightmap_shape = obs[heightmap_group].shape[1:]
        depth_shape = obs[depth_group].shape[1:]
        self.heightmap_dim = math.prod(heightmap_shape)
        self.depth_shape = depth_shape

        self.privilege_encoder = MLP(
            privileged_dim, num_privilege_latent_dims, privilege_encoder_hidden_dims, activation
        )
        self.num_teacher = 0
        self.privilege_estimator = MLP(
            self.history_dim, num_privilege_latent_dims, privilege_estimator_hidden_dims, activation
        )
        self.privilege_decoder = MLP(
            num_privilege_latent_dims, privileged_dim, privilege_decoder_hidden_dims, activation
        )
        self.heightmap_decoder = MLP(
            num_heightmap_latent_dims, self.heightmap_dim, heightmap_decoder_hidden_dims, activation
        )
        self.heightmap_encoder = _HeightmapEncoder(
            heightmap_shape,
            num_heightmap_latent_dims,
            channel_dims=heightmap_encoder_channel_dims,
            kernel_sizes=heightmap_encoder_kernel_sizes,
            fc_layer_dims=heightmap_encoder_fc_layer_dims,
            strides=heightmap_encoder_strides,
            activation=activation,
        )
        self.depth_estimator = HeightmapEstimator(
            depth_image_resolution=(depth_shape[1], depth_shape[2]),
            num_obs_history=self.history_dim,
            output_dim=num_heightmap_latent_dims,
            cnn_input_channel=depth_shape[0],
            cnn_channel_dims=list(depth_cnn_channel_dims),
            cnn_strides=list(depth_cnn_strides),
            cnn_fc_layer_dims=list(depth_cnn_fc_layer_dims),
            cnn_kernel_sizes=list(depth_cnn_kernel_sizes),
            history_mlp_dims=list(history_mlp_dims),
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            activation=activation,
        )

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the shared policy MLP."""
        return self.obs_dim + self._num_privilege_latent_dims + self._num_heightmap_latent_dims

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass dispatching between the teacher and the student paths."""
        if masks is not None:
            # Recurrent student update: obs is a padded trajectory TensorDict with the
            # history and depth_image groups merged. Output is flat over valid rows.
            latent = self._student_recurrent_latent(obs, masks, hidden_state)
        else:
            # Rollout / teacher update: flat per-sample obs, dispatched by teacher_mask.
            latent = self._flat_mixed_latent(obs)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _flat_mixed_latent(self, obs: TensorDict) -> torch.Tensor:
        """Assemble the actor latent for a flat batch with mixed teacher/student samples."""
        policy_latent = super().get_latent(obs)
        latent_dim = self._num_privilege_latent_dims + self._num_heightmap_latent_dims
        if not self.training:
            # Inference (play/export): always use the deployable student path.
            priv_latent = self.privilege_estimator(obs["history"])
            hm_latent = self.depth_estimator(obs["history"], obs["depth_image"], hidden_states=None, masks=None)
            encoder_latent = torch.cat([priv_latent, hm_latent], dim=-1)
            return torch.cat([policy_latent, encoder_latent], dim=-1)
        teacher_mask = obs["teacher_mask"].squeeze(-1).bool()
        encoder_latent = torch.zeros(
            policy_latent.shape[0], latent_dim, device=policy_latent.device, dtype=policy_latent.dtype
        )
        if teacher_mask.any():
            priv_latent = self.privilege_encoder(obs["privileged"][teacher_mask])
            hm_latent = self.heightmap_encoder(obs["heightmap"][teacher_mask])
            encoder_latent[teacher_mask] = torch.cat([priv_latent, hm_latent], dim=-1)
        if (~teacher_mask).any():
            student = ~teacher_mask
            priv_latent = self.privilege_estimator(obs["history"][student])
            hm_latent = self.depth_estimator(
                obs["history"][student], obs["depth_image"][student], hidden_states=None, masks=None
            )
            encoder_latent[student] = torch.cat([priv_latent, hm_latent], dim=-1)
        return torch.cat([policy_latent, encoder_latent], dim=-1)

    def _student_recurrent_latent(
        self, obs: TensorDict, masks: torch.Tensor, hidden_state: HiddenState
    ) -> torch.Tensor:
        """Assemble the actor latent for the recurrent student update path."""
        policy_latent_padded = self.obs_normalizer(
            torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        )
        policy_latent = unpad_trajectories_flat(policy_latent_padded, masks)
        history_flat = unpad_trajectories_flat(obs["history"], masks)
        priv_latent = self.privilege_estimator(history_flat)
        hm_latent = self.depth_estimator(obs["history"], obs["depth_image"], hidden_state, masks)
        return torch.cat([policy_latent, priv_latent, hm_latent], dim=-1)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the depth estimator (student only)."""
        return self.depth_estimator.get_hidden_states()  # type: ignore[return-value]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the depth estimator hidden state, optionally for done environments.

        ``dones`` covers all environments; the GRU state is student-only, so it is
        sliced to the student environments (``[num_teacher, num_envs)``) first.
        """
        if dones is not None:
            dones = dones[self.num_teacher :]
        self.depth_estimator.reset_hidden_states(dones)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the depth estimator hidden state for truncated backpropagation."""
        self.depth_estimator.detach_hidden_states()

    def as_jit(self) -> nn.Module:
        """Return a student-mode version of the model compatible with Torch JIT export."""
        return _TorchCtsDepthActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a student-mode version of the model compatible with ONNX export."""
        return _OnnxCtsDepthActor(self, verbose)


class _TorchCtsDepthActor(nn.Module):
    """Exportable student-mode depth-aware CTS actor for JIT."""

    def __init__(self, model: CtsDepthActor) -> None:
        """Create a TorchScript-friendly copy of a CtsDepthActor."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.privilege_estimator = copy.deepcopy(model.privilege_estimator)
        self.depth_cnn = copy.deepcopy(model.depth_estimator.cnn)
        self.history_mlp = copy.deepcopy(model.depth_estimator.history_mlp)
        self.rnn = copy.deepcopy(model.depth_estimator.rnn)
        self.latent_output_mlp = copy.deepcopy(model.depth_estimator.latent_output_mlp)
        self.mlp = copy.deepcopy(model.mlp)
        self.policy_obs_dim = model.obs_dim
        self.history_dim = model.history_dim
        self.depth_shape = model.depth_shape
        self.deterministic_output = model.distribution.as_deterministic_output_module()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic student-mode inference on concatenated (obs, history, depth)."""
        policy_obs = self.obs_normalizer(x[:, : self.policy_obs_dim])
        obs_history = x[:, self.policy_obs_dim : self.policy_obs_dim + self.history_dim]
        depth = x[:, self.policy_obs_dim + self.history_dim :].view(-1, *self.depth_shape)
        depth_encoding = self.depth_cnn(depth)
        history_encoding = self.history_mlp(obs_history)
        rnn_input = torch.cat([history_encoding, depth_encoding], dim=-1)
        rnn_out, h = self.rnn(rnn_input.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        hm_latent = self.latent_output_mlp(rnn_out.squeeze(0))
        priv_latent = self.privilege_estimator(obs_history)
        out = self.mlp(torch.cat([policy_obs, priv_latent, hm_latent], dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset the recurrent hidden state to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore


class _OnnxCtsDepthActor(nn.Module):
    """Exportable student-mode depth-aware CTS actor for ONNX.

    Stateful inference is not supported by ONNX Runtime (module buffers are static
    in the graph), so the GRU hidden state is an explicit input/output pair managed
    by the caller. The three observation sources arrive as separate tensors: the
    current policy observation, the flat history, and the depth stack kept in its
    ``[B, C, H, W]`` layout.
    """

    is_recurrent: bool = True

    def __init__(self, model: CtsDepthActor, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CtsDepthActor."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.privilege_estimator = copy.deepcopy(model.privilege_estimator)
        self.depth_cnn = copy.deepcopy(model.depth_estimator.cnn)
        self.history_mlp = copy.deepcopy(model.depth_estimator.history_mlp)
        self.rnn = copy.deepcopy(model.depth_estimator.rnn)
        self.latent_output_mlp = copy.deepcopy(model.depth_estimator.latent_output_mlp)
        self.mlp = copy.deepcopy(model.mlp)
        self.policy_obs_dim = model.obs_dim
        self.history_dim = model.history_dim
        self.depth_shape = model.depth_shape
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self,
        policy_obs: torch.Tensor,
        obs_history: torch.Tensor,
        depth: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run deterministic student-mode inference with an external GRU state."""
        policy_obs = self.obs_normalizer(policy_obs)
        depth_encoding = self.depth_cnn(depth)
        history_encoding = self.history_mlp(obs_history)
        rnn_input = torch.cat([history_encoding, depth_encoding], dim=-1)
        rnn_out, h = self.rnn(rnn_input.unsqueeze(0), hidden_state)
        hm_latent = self.latent_output_mlp(rnn_out.squeeze(0))
        priv_latent = self.privilege_estimator(obs_history)
        out = self.mlp(torch.cat([policy_obs, priv_latent, hm_latent], dim=-1))
        return self.deterministic_output(out), h

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        return (
            torch.zeros(1, self.policy_obs_dim),
            torch.zeros(1, self.history_dim),
            torch.zeros(1, *self.depth_shape),
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        """ONNX input tensor names."""
        return ["current_obs", "obs_history", "depth_image", "hidden_state"]

    @property
    def output_names(self) -> list[str]:
        """ONNX output tensor names.

        The carried state output is named ``new_hidden_state`` (distinct from the
        ``hidden_state`` input) because ONNX graphs are in single static assignment
        form: an input and an output may not share a name.
        """
        return ["actions", "new_hidden_state"]
