# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circular buffer for storing AMP discriminator observations."""

from __future__ import annotations

import torch
from collections.abc import Generator, Sequence


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data.

    This class implements a circular buffer for storing a history of batched tensor data. The buffer is
    initialized with a maximum length and a batch size. The data is stored in a circular fashion, and the
    data can be retrieved in a LIFO (Last-In-First-Out) fashion. The buffer is designed to be used in
    multi-environment settings, where each environment has its own data.

    The shape of the appended data is expected to be (batch_size, ...), where the first dimension is the
    batch dimension. Correspondingly, the shape of the ring buffer is (max_len, batch_size, ...).
    """

    def __init__(self, max_len: int, batch_size: int, device: str) -> None:
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum allowed value is 1.
            batch_size: The batch dimension of the data.
            device: The device used for processing.

        Raises:
            ValueError: If the buffer size is less than one.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        # Set the parameters
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        # Max length tensor for comparisons
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        # Number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # The pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        # The actual buffer for data storage (initialized on first append)
        self._buffer: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        return self._batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._device

    @property
    def max_length(self) -> int:
        """The maximum length of the ring buffer."""
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        """The current length of the buffer. Shape is (batch_size,).

        Since the buffer is circular, the current length is the minimum of the number of pushes
        and the maximum length.
        """
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """Complete circular buffer with most recent entry at the end and oldest entry at the beginning.

        Returns:
            Complete circular buffer with most recent entry at the end and oldest entry at the beginning
            of dimension 1. The shape is [batch_size, max_length, data.shape[1:]].
        """
        if self._buffer is None:
            raise RuntimeError("Buffer is empty. Please append data first.")
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    def reset(self, batch_ids: Sequence[int] | None = None) -> None:
        """Reset the circular buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        # Resolve all indices
        if batch_ids is None:
            batch_ids = slice(None)
        # Reset the number of pushes for the specified batch indices
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            # Zero the buffer at the reset batch ids so the buffer() getter returns cleared data
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor) -> None:
        """Append the data to the circular buffer.

        Args:
            data: The data to append to the circular buffer. The first dimension should be the batch dimension.
                Shape is (batch_size, ...).

        Raises:
            ValueError: If the input data has a different batch size than the buffer.
        """
        # Check the batch size
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")

        # Move the data to the device
        data = data.to(self._device)
        # At the first call, initialize the buffer size
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
        # Move the head to the next slot
        self._pointer = (self._pointer + 1) % self.max_length
        # Add the new data to the last layer
        self._buffer[self._pointer] = data
        # Check for batches with zero pushes and initialize all values in batch to first append
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        # Increment number of number of pushes for all batches
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """Retrieve the data from the circular buffer in last-in-first-out (LIFO) fashion.

        If the requested index is larger than the number of pushes since the last call to :meth:`reset`,
        the oldest stored data is returned.

        Args:
            key: The index to retrieve from the circular buffer. The index should be less than the number of pushes
                since the last call to :meth:`reset`. Shape is (batch_size,).

        Returns:
            The data from the circular buffer. Shape is (batch_size, ...).

        Raises:
            ValueError: If the input key has a different batch size than the buffer.
            RuntimeError: If the buffer is empty.
        """
        # Check the batch size
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        # Check if the buffer is empty
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        # Admissible lag
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        # The index in the circular buffer (pointer points to the last+1 index)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        # Return output
        return self._buffer[index_in_buffer, self._ALL_INDICES]

    def mini_batch_generator(
        self, fetch_length: int, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[torch.Tensor, None, None]:
        """Generate mini-batches from the circular buffer.

        Args:
            fetch_length: Number of time steps to fetch from the buffer.
            num_mini_batches: Number of mini-batches per epoch.
            num_epochs: Number of epochs.

        Yields:
            Mini-batches of data.

        Raises:
            RuntimeError: If the buffer is empty.
            ValueError: If fetch_length exceeds the minimum current buffer length.
        """
        # Check if the buffer is empty
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        min_current_length = torch.min(self.current_length).item()
        if fetch_length > min_current_length:
            raise ValueError(f"Fetch length {fetch_length} exceeds minimum current length {min_current_length}.")

        epoch_batch_size = self.batch_size * fetch_length
        mini_batch_size = epoch_batch_size // num_mini_batches

        if epoch_batch_size % num_mini_batches != 0:
            raise ValueError(
                f"Epoch batch size {epoch_batch_size} is not divisible by number of mini-batches {num_mini_batches}."
            )

        # Assume that every element in self._num_pushes are equal
        total_combinations = self.current_length[0].item() * self.batch_size
        linear_indices = torch.randperm(int(total_combinations), device=self.device)[:epoch_batch_size]
        # Convert linear indices to batch and fetch indices
        indices_0 = linear_indices // self.batch_size
        indices_1 = linear_indices % self.batch_size

        for _epoch in range(num_epochs):
            indices = torch.randperm(epoch_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mini_indices_0 = indices_0[indices[start:end]]
                mini_indices_1 = indices_1[indices[start:end]]

                yield self._buffer[mini_indices_0, mini_indices_1]
