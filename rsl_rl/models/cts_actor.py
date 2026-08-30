# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor network for Concurrent Teacher-Student (CTS) training."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState


class CtsActor(MLPModel):
    """Actor for Concurrent Teacher-Student training.

    The actor shares a single policy MLP between the teacher and the student. The teacher
    encodes privileged observations with :attr:`privilege_encoder`, while the student
    encodes the stacked observation history with :attr:`history_encoder`; both produce a
    latent vector that is concatenated with the policy observation before the shared MLP.

    The observation TensorDict must provide a ``teacher_mask`` group (1 for teacher
    environments, 0 otherwise) so that the per-sample latent source can be dispatched
    without materializing history for teacher samples.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_latent_dims: int,
        privilege_encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        history_encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the CTS actor with two latent encoders and a shared policy MLP."""
        self._num_latent_dims = num_latent_dims
        # The base model builds the policy MLP with input dim ``obs_dim + num_latent_dims``
        # through the overridden ``_get_latent_dim``.
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
        for set_name in ("privileged", "history"):
            if set_name not in obs_groups:
                raise ValueError(f"CtsActor requires an '{set_name}' entry in obs_groups.")
        self.privileged_groups, privileged_dim = self._get_obs_dim(obs, obs_groups, "privileged")
        self.history_groups, self.history_dim = self._get_obs_dim(obs, obs_groups, "history")
        self.privilege_encoder = MLP(privileged_dim, num_latent_dims, privilege_encoder_hidden_dims, activation)
        self.history_encoder = MLP(self.history_dim, num_latent_dims, history_encoder_hidden_dims, activation)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the shared policy MLP."""
        return self.obs_dim + self._num_latent_dims

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the actor latent by concatenating the policy obs with the encoder latent."""
        policy_latent = super().get_latent(obs, masks, hidden_state)
        if self.training:
            teacher_mask = obs["teacher_mask"].squeeze(-1).bool()
            encoder_latent = torch.zeros(
                policy_latent.shape[0], self._num_latent_dims, device=policy_latent.device, dtype=policy_latent.dtype
            )
            if teacher_mask.any():
                encoder_latent[teacher_mask] = self.privilege_encoder(obs["privileged"][teacher_mask])
            if (~teacher_mask).any():
                encoder_latent[~teacher_mask] = self.history_encoder(obs["history"][~teacher_mask])
        else:
            # Inference (play/export): always use the deployable student (history) encoder.
            encoder_latent = self.history_encoder(obs["history"])
        return torch.cat([policy_latent, encoder_latent], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a student-mode version of the model compatible with Torch JIT export."""
        return _TorchCtsActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a student-mode version of the model compatible with ONNX export."""
        return _OnnxCtsActor(self, verbose)


class _TorchCtsActor(nn.Module):
    """Exportable student-mode CTS actor for JIT."""

    def __init__(self, model: CtsActor) -> None:
        """Create a TorchScript-friendly copy of a CtsActor."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.policy_obs_dim = model.obs_dim
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic student-mode inference on concatenated (policy_obs, history)."""
        policy_obs = self.obs_normalizer(x[:, : self.policy_obs_dim])
        obs_history = x[:, self.policy_obs_dim :]
        latent = self.history_encoder(obs_history)
        out = self.mlp(torch.cat([policy_obs, latent], dim=-1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxCtsActor(nn.Module):
    """Exportable student-mode CTS actor for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: CtsActor, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CtsActor."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.policy_obs_dim = model.obs_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim + model.history_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic student-mode inference for ONNX export."""
        policy_obs = self.obs_normalizer(x[:, : self.policy_obs_dim])
        obs_history = x[:, self.policy_obs_dim :]
        latent = self.history_encoder(obs_history)
        out = self.mlp(torch.cat([policy_obs, latent], dim=-1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """ONNX output tensor names."""
        return ["actions"]
