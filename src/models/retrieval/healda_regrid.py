# -*- coding: utf-8 -*-
"""Optional HPX <-> lat-lon regridding helpers.

The functions use ``earth2grid`` when available and otherwise raise a clear
message.  The training model can still run through its lat-lon fallback without
these dependencies.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def require_earth2grid():
    try:
        import earth2grid  # type: ignore
        return earth2grid
    except Exception as exc:  # pragma: no cover - dependency-specific
        raise ImportError(
            "earth2grid is required for native HPX regridding. Install earth2grid "
            "or set model.grid_backend=latlon / model.fallback_grid_backend=latlon."
        ) from exc


def latlon_to_hpx(x: torch.Tensor, nside: int = 64, lat_descending: bool = True) -> torch.Tensor:
    """Regrid ``[B,C,H,W]`` lat-lon tensor to HPX pixels when earth2grid is installed."""
    earth2grid = require_earth2grid()
    h, w = x.shape[-2:]
    ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(h, w)
    hpx_grid = earth2grid.healpix.Grid(int(nside), pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
    regridder = earth2grid.get_regridder(ll_grid, hpx_grid)
    return regridder(x)


def hpx_to_latlon(x: torch.Tensor, output_grid: Sequence[int] = (181, 360), nside: int = 64) -> torch.Tensor:
    """Regrid HPX pixel tensor to ``[B,C,H,W]`` when earth2grid is installed."""
    earth2grid = require_earth2grid()
    h, w = int(output_grid[0]), int(output_grid[1])
    hpx_grid = earth2grid.healpix.Grid(int(nside), pixel_order=earth2grid.healpix.HEALPIX_PAD_XY)
    ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(h, w)
    regridder = earth2grid.get_regridder(hpx_grid, ll_grid)
    return regridder(x)


def regrid_consistency_check(x: torch.Tensor, nside: int = 64, output_grid: Sequence[int] = (181, 360)) -> float:
    """Return relative RMS error of latlon -> HPX -> latlon regridding."""
    y = hpx_to_latlon(latlon_to_hpx(x, nside=nside), output_grid=output_grid, nside=nside)
    if y.shape != x.shape:
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
    denom = torch.sqrt(torch.mean(x.float() ** 2)).clamp_min(1e-12)
    return float(torch.sqrt(torch.mean((x.float() - y.float()) ** 2)) / denom)
