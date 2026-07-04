# -*- coding: utf-8 -*-
"""HealDA-style sensor-specific observation embedders.

Each scalar observation is tokenized from a measurement, continuous metadata,
and integer metadata.  Tokens are scatter-reduced to a dense grid and mixed with
an observability mask, following the design in the HealDA paper while remaining
compatible with XiChen 1-degree NPY files.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn


def nan_to_num(x: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=val, posinf=val, neginf=val)


def fourier_features(x: torch.Tensor, num_freqs: int = 4) -> torch.Tensor:
    """Return sin/cos Fourier features for a normalized scalar tensor."""
    x = nan_to_num(x.float())
    freqs = torch.arange(1, num_freqs + 1, device=x.device, dtype=x.dtype)
    phase = x.unsqueeze(-1) * freqs * math.pi
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class MetadataEncoder(nn.Module):
    """Continuous metadata encoder used before the observation tokenizer.

    Output order is fixed and documented for reproducibility:
    ``lat/lon sin-cos`` (4), relative-time polynomial (2), local-solar-time
    Fourier (4), pressure Fourier (8), height Fourier (8), scan polynomial (2),
    satellite zenith cos/cos^2 (2), solar zenith sin/cos (2), quality flag (1).
    Total: 33 dimensions.
    """

    output_dim = 33

    def __init__(self, pressure_max_hpa: float = 1100.0, height_max_m: float = 60000.0) -> None:
        super().__init__()
        self.pressure_max_hpa = float(pressure_max_hpa)
        self.height_max_m = float(height_max_m)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        lat = obs["lat"].float()
        lon = torch.remainder(obs["lon"].float(), 360.0)
        rel_t = obs["relative_time"].float() / 24.0
        pressure = obs.get("pressure", torch.full_like(lat, float("nan"))).float()
        height = obs.get("height", torch.full_like(lat, float("nan"))).float()
        scan = obs.get("scan_angle", torch.full_like(lat, float("nan"))).float()
        satza = obs.get("sat_zenith_angle", torch.full_like(lat, float("nan"))).float()
        solza = obs.get("solar_zenith_angle", torch.full_like(lat, float("nan"))).float()
        quality = obs.get("quality_flag", torch.zeros_like(lat)).float()

        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
        loc = torch.stack([torch.sin(lat_rad), torch.cos(lat_rad), torch.sin(lon_rad), torch.cos(lon_rad)], dim=-1)
        time_poly = torch.stack([nan_to_num(rel_t), nan_to_num(rel_t) ** 2], dim=-1)
        # Local solar time from relative hour proxy and longitude; absolute UTC hour is not needed for the model interface.
        lst = torch.remainder(rel_t * 24.0 + lon / 15.0, 24.0) / 24.0
        lst_feat = fourier_features(lst, num_freqs=2)
        p = torch.clamp(nan_to_num(pressure / self.pressure_max_hpa), 0.0, 1.0)
        h = torch.clamp(nan_to_num(height / self.height_max_m), 0.0, 1.0)
        p_feat = fourier_features(p, num_freqs=4)
        h_feat = fourier_features(h, num_freqs=4)
        scan_norm = nan_to_num(scan / 50.0)
        scan_feat = torch.stack([scan_norm, scan_norm ** 2], dim=-1)
        sat_rad = torch.deg2rad(nan_to_num(satza))
        sat_feat = torch.stack([torch.cos(sat_rad), torch.cos(sat_rad) ** 2], dim=-1)
        sol_rad = torch.deg2rad(nan_to_num(solza))
        sol_feat = torch.stack([torch.cos(sol_rad), torch.sin(sol_rad)], dim=-1)
        q_feat = nan_to_num(quality).unsqueeze(-1)
        return torch.cat([loc, time_poly, lst_feat, p_feat, h_feat, scan_feat, sat_feat, sol_feat, q_feat], dim=-1)


class ObservationTokenizerMLP(nn.Module):
    """MLP that combines scalar measurement, float metadata and integer embeddings."""

    def __init__(
        self,
        metadata_dim: int,
        token_dim: int = 32,
        hidden_dim: int = 128,
        channel_vocab_size: int = 256,
        platform_vocab_size: int = 128,
        obs_type_vocab_size: int = 512,
        int_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.channel_emb = nn.Embedding(channel_vocab_size, int_embed_dim)
        self.platform_emb = nn.Embedding(platform_vocab_size, int_embed_dim)
        self.obs_type_emb = nn.Embedding(obs_type_vocab_size, int_embed_dim)
        in_dim = 1 + metadata_dim + 3 * int_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max(token_dim - 1, 1)),
            nn.SiLU(),
        )

    @staticmethod
    def _safe_int(x: torch.Tensor, vocab: int) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).long()
        return torch.remainder(torch.clamp(x, min=0), vocab)

    def forward(self, measurement: torch.Tensor, metadata: torch.Tensor, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        y = measurement.float().unsqueeze(-1)
        ch = self._safe_int(obs.get("channel", torch.zeros_like(measurement, dtype=torch.long)), self.channel_emb.num_embeddings)
        pl = self._safe_int(obs.get("platform", torch.zeros_like(ch)), self.platform_emb.num_embeddings)
        obs_type_src = obs.get("variable_type", obs.get("obs_type", ch))
        typ = self._safe_int(obs_type_src, self.obs_type_emb.num_embeddings)
        x = torch.cat([y, metadata, self.channel_emb(ch), self.platform_emb(pl), self.obs_type_emb(typ)], dim=-1)
        v = self.net(x)
        return torch.cat([y, v], dim=-1)


class LatLonAggregation(nn.Module):
    """Scatter-reduce point tokens to a regular latitude-longitude grid."""

    def __init__(self, grid_shape: Sequence[int] = (181, 360)) -> None:
        super().__init__()
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.npix = self.grid_shape[0] * self.grid_shape[1]

    def latlon_to_index(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        h, w = self.grid_shape
        lat = torch.clamp(lat.float(), -90.0, 90.0)
        lon = torch.remainder(lon.float(), 360.0)
        row = torch.round((90.0 - lat) / 180.0 * (h - 1)).long().clamp(0, h - 1)
        col = torch.floor(lon / 360.0 * w).long().clamp(0, w - 1)
        return row * w + col

    def forward(self, tokens: torch.Tensor, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = tokens.device
        k = tokens.shape[-1]
        out = torch.zeros(self.npix, k, dtype=tokens.dtype, device=device)
        count = torch.zeros(self.npix, 1, dtype=tokens.dtype, device=device)
        if tokens.numel() > 0:
            idx = self.latlon_to_index(obs["lat"].to(device), obs["lon"].to(device))
            out.index_add_(0, idx, tokens)
            ones = torch.ones(tokens.shape[0], 1, dtype=tokens.dtype, device=device)
            count.index_add_(0, idx, ones)
            out = out / count.clamp_min(1.0)
        mask = (count > 0).to(tokens.dtype)
        h, w = self.grid_shape
        return out.view(h, w, k).permute(2, 0, 1).contiguous(), mask.view(h, w, 1).permute(2, 0, 1).contiguous()


class HPXAggregation(LatLonAggregation):
    """HEALPix aggregation placeholder with safe lat-lon fallback.

    The production route can replace this class with an earth2grid-backed HPX
    scatter.  Keeping the class here preserves the HPX/lat-lon switch in configs
    and makes missing HEALPix dependencies explicit rather than silent.
    """

    def __init__(self, grid_shape: Sequence[int] = (181, 360), hpx_nside: int = 64, fallback_to_latlon: bool = True) -> None:
        super().__init__(grid_shape=grid_shape)
        self.hpx_nside = int(hpx_nside)
        self.fallback_to_latlon = bool(fallback_to_latlon)
        try:
            import earth2grid  # noqa: F401
            self.earth2grid_available = True
        except Exception:
            self.earth2grid_available = False
        if not self.earth2grid_available and not self.fallback_to_latlon:
            raise ImportError("earth2grid is required for HPXAggregation when fallback_to_latlon=False")


class SensorSpecificEmbedder(nn.Module):
    """Base class for all sensor-specific HealDA embedders."""

    def __init__(
        self,
        sensor_name: str,
        grid_shape: Sequence[int] = (181, 360),
        token_dim: int = 32,
        sensor_embed_dim: int = 256,
        hidden_dim: int = 128,
        grid_backend: str = "latlon",
        hpx_nside: int = 64,
        channel_vocab_size: int = 256,
        platform_vocab_size: int = 128,
        obs_type_vocab_size: int = 512,
    ) -> None:
        super().__init__()
        self.sensor_name = sensor_name
        self.metadata_encoder = MetadataEncoder()
        self.tokenizer = ObservationTokenizerMLP(
            metadata_dim=MetadataEncoder.output_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            channel_vocab_size=channel_vocab_size,
            platform_vocab_size=platform_vocab_size,
            obs_type_vocab_size=obs_type_vocab_size,
        )
        if grid_backend == "hpx":
            self.aggregator = HPXAggregation(grid_shape=grid_shape, hpx_nside=hpx_nside, fallback_to_latlon=True)
        else:
            self.aggregator = LatLonAggregation(grid_shape=grid_shape)
        self.mixing_mlp = nn.Sequential(
            nn.Conv2d(token_dim + 1, sensor_embed_dim, kernel_size=1),
            nn.GroupNorm(8 if sensor_embed_dim % 8 == 0 else 1, sensor_embed_dim),
            nn.SiLU(),
            nn.Conv2d(sensor_embed_dim, sensor_embed_dim, kernel_size=1),
        )

    def _to_device(self, obs: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v for k, v in obs.items()}

    def forward(self, obs_batch: List[Mapping[str, torch.Tensor]], device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        outputs = []
        h, w = self.aggregator.grid_shape
        for obs in obs_batch:
            obs_dev = self._to_device(obs, device)
            if obs_dev["measurement"].numel() == 0:
                zeros = torch.zeros(1, h, w, device=device, dtype=next(self.parameters()).dtype)
                feat = self.mixing_mlp(torch.zeros(1, self.tokenizer.token_dim + 1, h, w, device=device, dtype=next(self.parameters()).dtype))[0]
                outputs.append(feat)
                continue
            metadata = self.metadata_encoder(obs_dev)
            tokens = self.tokenizer(obs_dev["measurement"].float(), metadata, obs_dev)
            grid, mask = self.aggregator(tokens, obs_dev)
            mixed = self.mixing_mlp(torch.cat([grid, mask], dim=0).unsqueeze(0))[0]
            outputs.append(mixed)
        return torch.stack(outputs, dim=0)


class ATMS_SensorEmbedder(SensorSpecificEmbedder):
    def __init__(self, **kwargs) -> None:
        super().__init__(sensor_name="atms", **kwargs)


class AMSUA_SensorEmbedder(SensorSpecificEmbedder):
    def __init__(self, **kwargs) -> None:
        super().__init__(sensor_name="amsua", **kwargs)


class MHS_SensorEmbedder(SensorSpecificEmbedder):
    def __init__(self, **kwargs) -> None:
        super().__init__(sensor_name="mhs", **kwargs)


class HIRS4_SensorEmbedder(SensorSpecificEmbedder):
    def __init__(self, **kwargs) -> None:
        super().__init__(sensor_name="hrs4", **kwargs)


class GDASPrebufr_SensorEmbedder(SensorSpecificEmbedder):
    def __init__(self, **kwargs) -> None:
        super().__init__(sensor_name="gdas_prebufr", **kwargs)
