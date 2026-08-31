# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heightmap (depth) estimator for depth-augmented CTS training."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MLP
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.utils.utils import unpad_trajectories_flat


class HeightmapEstimator(nn.Module):
    """Student depth-perception network: CNN(depth) + MLP(history) + RNN -> latent.

    Encodes a stack of depth images with a small CNN, the observation history with an
    MLP, fuses both and passes them through a recurrent (GRU/LSTM) layer. The final
    latent is the student's heightmap latent, aligned with the teacher's
    ``heightmap_encoder`` output for the reconstruction loss.

    Mirrors the student network of the original ppo_cts_depth. Note that (matching the
    source) no activation is applied after the first convolutional layer.
    """

    def __init__(
        self,
        depth_image_resolution: tuple[int, int],
        num_obs_history: int,
        output_dim: int,
        cnn_input_channel: int,
        cnn_channel_dims: list[int] | tuple[int, ...],
        cnn_strides: list[int] | tuple[int, ...],
        cnn_fc_layer_dims: list[int] | tuple[int, ...],
        cnn_kernel_sizes: list[int] | tuple[int, ...],
        history_mlp_dims: list[int] | tuple[int, ...],
        rnn_type: str,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        activation: str = "elu",
    ) -> None:
        """Initialize the depth estimator."""
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        cnn_layers: list[nn.Module] = []
        in_channels = cnn_input_channel
        in_h, in_w = depth_image_resolution
        for i, (out_ch, k, s) in enumerate(zip(cnn_channel_dims, cnn_kernel_sizes, cnn_strides)):
            cnn_layers.append(nn.Conv2d(in_channels, out_ch, k, s))
            if i != 0:
                cnn_layers.append(activation_fn)
            in_h, in_w = (in_h - k) // s + 1, (in_w - k) // s + 1
            in_channels = out_ch
        cnn_layers.append(nn.Flatten())
        cnn_out = in_channels * in_h * in_w

        fc_layers: list[nn.Module] = []
        fc_in = cnn_out
        for dim in cnn_fc_layer_dims:
            fc_layers.append(nn.Linear(fc_in, dim))
            fc_layers.append(activation_fn)
            fc_in = dim
        self.cnn = nn.Sequential(*cnn_layers, *fc_layers)

        self.history_mlp = MLP(num_obs_history, history_mlp_dims[-1], list(history_mlp_dims), activation)
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=history_mlp_dims[-1] + cnn_fc_layer_dims[-1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
        )
        self.latent_output_mlp = nn.Linear(rnn_hidden_size, output_dim)

        self.hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        obs_history: torch.Tensor,
        depth_image: torch.Tensor,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode depth images and observation history into a heightmap latent.

        In batch mode (``masks`` not None) the inputs are padded trajectories
        ``[T, N, ...]`` and the output is the flat sequence of valid rows (trajectory
        order). In rollout mode the inputs are per-step ``[N, ...]`` and the output is
        ``[N, output_dim]``; the recurrent hidden state is carried internally.
        """
        if masks is not None:
            depth_encoding = self.cnn(depth_image.flatten(0, 1)).view(*depth_image.shape[:2], -1)
            history_encoding = self.history_mlp(obs_history)
            rnn_input = torch.cat([history_encoding, depth_encoding], dim=-1)
            rnn_out, _ = self.rnn(rnn_input, hidden_states)
            rnn_out = unpad_trajectories_flat(rnn_out, masks)
        else:
            depth_encoding = self.cnn(depth_image)
            history_encoding = self.history_mlp(obs_history)
            rnn_input = torch.cat([history_encoding, depth_encoding], dim=-1)
            self._ensure_hidden_state(rnn_input.shape[0], rnn_input.device)
            rnn_out, self.hidden_states = self.rnn(rnn_input.unsqueeze(0), self.hidden_states)
            # Detach the carried hidden state so the rollout graph never spans steps: the
            # runner does not call detach_hidden_states, and the training backward must
            # only traverse the trajectory-start states saved in the storage.
            self.hidden_states = self._detach(self.hidden_states)
            rnn_out = rnn_out.squeeze(0)
        return self.latent_output_mlp(rnn_out)

    @staticmethod
    def _detach(
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Detach a GRU or LSTM hidden state from the computation graph."""
        if isinstance(hidden_states, tuple):
            return tuple(h.detach() for h in hidden_states)
        return hidden_states.detach()

    def _ensure_hidden_state(self, batch_size: int, device: torch.device) -> None:
        """Initialize the internal recurrent hidden state, resizing it if the batch changed."""
        if isinstance(self.hidden_states, tuple):
            h, c = self.hidden_states
            if h is None or h.shape[-2] != batch_size:
                h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
                c = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device)
            self.hidden_states = (h, c)
        else:
            if self.hidden_states is None or self.hidden_states.shape[-2] != batch_size:
                self.hidden_states = torch.zeros(
                    self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=device
                )

    def reset_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        """Zero the recurrent hidden state, optionally only for done environments."""
        if dones is None:
            self.hidden_states = None
            return
        if self.hidden_states is None:
            return
        if isinstance(self.hidden_states, tuple):
            for hidden_state in self.hidden_states:
                hidden_state[..., dones == 1, :] = 0.0  # type: ignore
        else:
            self.hidden_states[..., dones == 1, :] = 0.0  # type: ignore

    def detach_hidden_states(self) -> None:
        """Detach the recurrent hidden state from the computation graph."""
        if self.hidden_states is None:
            return
        if isinstance(self.hidden_states, tuple):
            self.hidden_states = tuple(h.detach().clone() for h in self.hidden_states)
        else:
            self.hidden_states = self.hidden_states.detach().clone()

    def get_hidden_states(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Return the current recurrent hidden state (student only)."""
        return self.hidden_states
