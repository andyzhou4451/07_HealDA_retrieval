# -*- coding: utf-8 -*-
"""Retrieval output head."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ProfileRetrievalDecoder(nn.Module):
    """Ensure output channels represent T/Q profiles.

    The decoder returns ``[B, 26, H, W]`` by default, or ``[B, 2, 13, H, W]``
    when ``as_profile=True`` is requested by the caller.
    """

    def __init__(self, in_channels: int, output_channels: int = 26, pressure_levels: Sequence[int] | None = None) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.pressure_levels = list(pressure_levels or [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        self.head = nn.Conv2d(in_channels, self.output_channels, kernel_size=1) if in_channels != output_channels else nn.Identity()

    def forward(self, x: torch.Tensor, as_profile: bool = False) -> torch.Tensor:
        y = self.head(x)
        if as_profile:
            b, c, h, w = y.shape
            if c != 2 * len(self.pressure_levels):
                raise ValueError(f"Cannot reshape {c} channels into [2,{len(self.pressure_levels)}]")
            return y.view(b, 2, len(self.pressure_levels), h, w)
        return y
