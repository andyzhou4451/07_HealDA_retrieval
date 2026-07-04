# -*- coding: utf-8 -*-
"""Observation encoder and sensor fusion for HealDA-style retrieval."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn

from .healda_sensor_embedder import (
    AMSUA_SensorEmbedder,
    ATMS_SensorEmbedder,
    GDASPrebufr_SensorEmbedder,
    HIRS4_SensorEmbedder,
    MHS_SensorEmbedder,
    SensorSpecificEmbedder,
)


class ObservabilityMaskBuilder(nn.Module):
    """Build a dense per-sensor observability mask from observation lists."""

    def __init__(self, grid_shape: Sequence[int] = (181, 360)) -> None:
        super().__init__()
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))

    def forward(self, observations: Mapping[str, List[Mapping[str, torch.Tensor]]], device: torch.device) -> torch.Tensor:
        h, w = self.grid_shape
        masks = []
        for sensor, obs_batch in observations.items():
            per_batch = []
            for obs in obs_batch:
                lat = obs["lat"].to(device)
                lon = obs["lon"].to(device)
                m = torch.zeros(h * w, dtype=torch.float32, device=device)
                if lat.numel() > 0:
                    row = torch.round((90.0 - torch.clamp(lat.float(), -90.0, 90.0)) / 180.0 * (h - 1)).long().clamp(0, h - 1)
                    col = torch.floor(torch.remainder(lon.float(), 360.0) / 360.0 * w).long().clamp(0, w - 1)
                    idx = row * w + col
                    m[idx] = 1.0
                per_batch.append(m.view(1, h, w))
            masks.append(torch.stack(per_batch, dim=0))
        return torch.cat(masks, dim=1) if masks else torch.empty(0, device=device)


class SensorFusion(nn.Module):
    """Fuse per-sensor maps by uniform HealDA-style reduction plus optional gates."""

    def __init__(self, sensors: Sequence[str], dim: int, learned_gates: bool = True) -> None:
        super().__init__()
        self.sensors = list(sensors)
        if learned_gates:
            self.logits = nn.Parameter(torch.zeros(len(self.sensors)))
        else:
            self.register_parameter("logits", None)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.SiLU(), nn.Conv2d(dim, dim, kernel_size=1))

    def forward(self, sensor_maps: Mapping[str, torch.Tensor]) -> torch.Tensor:
        maps = [sensor_maps[s] for s in self.sensors if s in sensor_maps]
        if not maps:
            raise ValueError("No sensor maps were produced by the observation encoder")
        stacked = torch.stack(maps, dim=0)  # [S,B,D,H,W]
        if self.logits is not None:
            weights = torch.softmax(self.logits[: stacked.shape[0]], dim=0).view(-1, 1, 1, 1, 1)
            fused = (stacked * weights).sum(dim=0) * math.sqrt(stacked.shape[0])
        else:
            fused = stacked.sum(dim=0) / math.sqrt(stacked.shape[0])
        return self.proj(fused)


class HealDAObservationEncoder(nn.Module):
    """Observation Encoder: sensor-specific embedders + sensor fusion."""

    EMBEDDER_CLS = {
        "atms": ATMS_SensorEmbedder,
        "amsua": AMSUA_SensorEmbedder,
        "mhs": MHS_SensorEmbedder,
        "hrs4": HIRS4_SensorEmbedder,
        "gdas_prebufr": GDASPrebufr_SensorEmbedder,
    }

    def __init__(
        self,
        sensors: Sequence[str],
        grid_shape: Sequence[int] = (181, 360),
        token_dim: int = 32,
        sensor_embed_dim: int = 256,
        grid_backend: str = "latlon",
        hpx_nside: int = 64,
        channel_vocab_size: int = 256,
        platform_vocab_size: int = 128,
        obs_type_vocab_size: int = 512,
    ) -> None:
        super().__init__()
        self.sensors = list(sensors)
        embedders: Dict[str, SensorSpecificEmbedder] = {}
        for sensor in self.sensors:
            if sensor not in self.EMBEDDER_CLS:
                raise ValueError(f"Unsupported retrieval sensor {sensor!r}")
            embedders[sensor] = self.EMBEDDER_CLS[sensor](
                grid_shape=grid_shape,
                token_dim=token_dim,
                sensor_embed_dim=sensor_embed_dim,
                grid_backend=grid_backend,
                hpx_nside=hpx_nside,
                channel_vocab_size=channel_vocab_size,
                platform_vocab_size=platform_vocab_size,
                obs_type_vocab_size=obs_type_vocab_size,
            )
        self.sensor_embedders = nn.ModuleDict(embedders)
        self.fusion = SensorFusion(self.sensors, dim=sensor_embed_dim, learned_gates=True)
        self.mask_builder = ObservabilityMaskBuilder(grid_shape=grid_shape)

    def forward(self, observations: Mapping[str, List[Mapping[str, torch.Tensor]]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        sensor_maps = {}
        for sensor, embedder in self.sensor_embedders.items():
            if sensor not in observations:
                raise KeyError(f"Missing sensor {sensor!r} in batch observations")
            sensor_maps[sensor] = embedder(observations[sensor], device=device)
        fused = self.fusion(sensor_maps)
        masks = self.mask_builder(observations, device=device)
        return fused, masks
