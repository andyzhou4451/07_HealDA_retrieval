# -*- coding: utf-8 -*-
"""Top-level HealDA-style multi-source T/Q profile retrieval model."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from torch import nn

from .healda_hpx_vit import HPXViTBackbone, LatLonViTBackbone
from .healda_obs_encoder import HealDAObservationEncoder
from .retrieval_decoder import ProfileRetrievalDecoder

MODEL_SIZES: Dict[str, Dict[str, int]] = {
    "tiny": {"dim": 256, "depth": 6, "heads": 4, "obs_token_dim": 32, "sensor_embed_dim": 128},
    "base": {"dim": 512, "depth": 12, "heads": 8, "obs_token_dim": 32, "sensor_embed_dim": 256},
    "full_healda_like": {"dim": 1024, "depth": 24, "heads": 16, "obs_token_dim": 32, "sensor_embed_dim": 512},
}


class HealDAXiChenRetrieval(nn.Module):
    """Observation-only, background-free retrieval model.

    Input batch format is produced by ``collate_retrieval_batch``.  Forward output
    is ``[B, 26, 181, 360]`` where channels are ``t-50..t-1000`` followed by
    ``q-50..q-1000``.
    """

    def __init__(
        self,
        sensors: Sequence[str] = ("atms", "amsua", "mhs", "hrs4", "gdas_prebufr"),
        target_vars: Sequence[str] | None = None,
        pressure_levels: Sequence[int] | None = None,
        output_channels: int = 26,
        output_grid: Sequence[int] = (181, 360),
        grid_backend: str = "hpx",
        fallback_grid_backend: str = "latlon",
        hpx_nside: int = 64,
        model_size: str = "base",
        dim: int | None = None,
        depth: int | None = None,
        heads: int | None = None,
        obs_token_dim: int | None = None,
        sensor_embed_dim: int | None = None,
        patch_size: Sequence[int] = (6, 6),
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        drop_path: float = 0.1,
        concat_observability_mask: bool = True,
        use_gradient_checkpointing: bool = False,
        channels_last: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown model_size {model_size!r}; choose one of {sorted(MODEL_SIZES)}")
        preset = MODEL_SIZES[model_size]
        self.sensors = list(sensors)
        self.pressure_levels = list(pressure_levels or [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        self.target_vars = list(target_vars or [*(f"t-{p}" for p in self.pressure_levels), *(f"q-{p}" for p in self.pressure_levels)])
        self.output_channels = int(output_channels)
        self.output_grid = (int(output_grid[0]), int(output_grid[1]))
        self.grid_backend = grid_backend
        self.fallback_grid_backend = fallback_grid_backend
        self.hpx_nside = int(hpx_nside)
        self.model_size = model_size
        self.concat_observability_mask = bool(concat_observability_mask)
        self.channels_last = bool(channels_last)

        self.dim = int(dim or preset["dim"])
        self.depth = int(depth or preset["depth"])
        self.heads = int(heads or preset["heads"])
        self.obs_token_dim = int(obs_token_dim or preset["obs_token_dim"])
        self.sensor_embed_dim = int(sensor_embed_dim or preset["sensor_embed_dim"])

        active_grid_backend = grid_backend
        if grid_backend == "hpx":
            # The current XiChen-compatible implementation keeps the HealDA HPX API but
            # trains on the public02 [181, 360] lat-lon labels.  Native HPX scatter/regrid
            # remains available through tools/regrid_hpx_latlon.py, but the training path
            # deliberately falls back to lat-lon unless a future native HPX module replaces
            # HPXAggregation with an earth2grid-backed implementation.
            if fallback_grid_backend != "latlon":
                raise ImportError("grid_backend=hpx requested, but this package currently requires fallback_grid_backend=latlon for training")
            active_grid_backend = "latlon"
        self.active_grid_backend = active_grid_backend

        self.obs_encoder = HealDAObservationEncoder(
            sensors=self.sensors,
            grid_shape=self.output_grid,
            token_dim=self.obs_token_dim,
            sensor_embed_dim=self.sensor_embed_dim,
            grid_backend=self.active_grid_backend,
            hpx_nside=self.hpx_nside,
        )
        in_channels = self.sensor_embed_dim + (len(self.sensors) if self.concat_observability_mask else 0)
        backbone_cls = HPXViTBackbone if self.active_grid_backend == "hpx" else LatLonViTBackbone
        self.backbone = backbone_cls(
            in_channels=in_channels,
            out_channels=self.output_channels,
            img_size=self.output_grid,
            patch_size=patch_size,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
            use_checkpoint=use_gradient_checkpointing,
        )
        self.decoder = ProfileRetrievalDecoder(self.output_channels, output_channels=self.output_channels, pressure_levels=self.pressure_levels)

    def forward(self, batch: Mapping[str, Any] | None = None, *, observations: Mapping[str, Any] | None = None, as_profile: bool = False) -> torch.Tensor:
        if batch is not None:
            observations = batch["observations"]
        if observations is None:
            raise ValueError("HealDAXiChenRetrieval.forward requires a batch or observations mapping")
        obs_features, obs_masks = self.obs_encoder(observations)
        if self.concat_observability_mask:
            x = torch.cat([obs_features, obs_masks.to(dtype=obs_features.dtype)], dim=1)
        else:
            x = obs_features
        if self.channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        y = self.backbone(x)
        return self.decoder(y, as_profile=as_profile)

    def estimate_vram_gb(self, batch_size: int = 1) -> float:
        """Very coarse activation-memory estimate for run planning."""
        h, w = self.output_grid
        ph, pw = self.backbone.patch_size
        tokens = ((h + self.backbone.pad_h) // ph) * ((w + self.backbone.pad_w) // pw)
        attention = batch_size * self.depth * self.heads * tokens * tokens * 2 / 1024**3
        activations = batch_size * self.depth * tokens * self.dim * 8 / 1024**3
        params = sum(p.numel() for p in self.parameters()) * 4 / 1024**3
        return float(params + activations + attention)
