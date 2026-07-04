# -*- coding: utf-8 -*-
"""Metrics for T/Q retrieval."""

from __future__ import annotations

from typing import Dict, Sequence

import torch


def latitude_weights(nlat: int, device=None, dtype=torch.float32) -> torch.Tensor:
    lat = torch.linspace(90.0, -90.0, nlat, device=device, dtype=dtype)
    w = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    return w / w.mean().clamp_min(1e-12)


def _rmse(x: torch.Tensor, dim=None) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2, dim=dim))


def retrieval_metrics(pred: torch.Tensor, target: torch.Tensor, pressure_levels: Sequence[int] | None = None) -> Dict[str, float]:
    if pred.ndim == 5:
        pred = pred.reshape(pred.shape[0], 26, pred.shape[-2], pred.shape[-1])
    if target.ndim == 5:
        target = target.reshape(target.shape[0], 26, target.shape[-2], target.shape[-1])
    levels = list(pressure_levels or [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    finite = torch.isfinite(pred) & torch.isfinite(target)
    err = torch.where(finite, pred.float() - target.float(), torch.zeros_like(pred, dtype=torch.float32))
    t_err, q_err = err[:, :13], err[:, 13:]
    t_den = finite[:, :13].float().sum().clamp_min(1.0)
    q_den = finite[:, 13:].float().sum().clamp_min(1.0)
    out: Dict[str, float] = {
        "overall_rmse": float(_rmse(err).detach().cpu()),
        "temperature_rmse": float(torch.sqrt((t_err ** 2).sum() / t_den).detach().cpu()),
        "humidity_rmse": float(torch.sqrt((q_err ** 2).sum() / q_den).detach().cpu()),
        "temperature_mae": float(torch.abs(t_err).sum().div(t_den).detach().cpu()),
        "humidity_mae": float(torch.abs(q_err).sum().div(q_den).detach().cpu()),
        "temperature_bias": float(t_err.sum().div(t_den).detach().cpu()),
        "humidity_bias": float(q_err.sum().div(q_den).detach().cpu()),
    }
    w = latitude_weights(pred.shape[-2], device=pred.device, dtype=pred.dtype).view(1, 1, -1, 1)
    out["latitude_weighted_rmse"] = float(torch.sqrt(torch.mean(err.float() ** 2 * w)).detach().cpu())
    per_t = _rmse(t_err, dim=(0, 2, 3)).detach().cpu()
    per_q = _rmse(q_err, dim=(0, 2, 3)).detach().cpu()
    for i, p in enumerate(levels):
        out[f"t_rmse_{p}"] = float(per_t[i])
        out[f"q_rmse_{p}"] = float(per_q[i])
    high = [i for i, p in enumerate(levels) if p <= 300]
    mid = [i for i, p in enumerate(levels) if 300 < p < 700]
    low = [i for i, p in enumerate(levels) if p >= 700]
    for name, idx in {"high": high, "mid": mid, "low": low}.items():
        if idx:
            out[f"temperature_rmse_{name}"] = float(_rmse(t_err[:, idx]).detach().cpu())
            out[f"humidity_rmse_{name}"] = float(_rmse(q_err[:, idx]).detach().cpu())
    return out
