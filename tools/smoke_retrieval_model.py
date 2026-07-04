#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a synthetic forward/loss/backward smoke test for the retrieval model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.losses.retrieval_tq_loss import RetrievalTQLoss
from src.models.retrieval.healda_xichen_retrieval import HealDAXiChenRetrieval
from src.utils.device import autocast, configure_accelerator_performance


def make_obs(n: int, device_cpu: torch.device) -> Dict[str, torch.Tensor]:
    """Create one synthetic point-cloud observation dictionary on CPU."""
    lat = torch.linspace(-80.0, 80.0, n, device=device_cpu)
    lon = torch.linspace(0.0, 359.0, n, device=device_cpu)
    ch = torch.arange(n, device=device_cpu) % 16
    return {
        "measurement": torch.randn(n, device=device_cpu) * 0.1,
        "lat": lat,
        "lon": lon,
        "relative_time": torch.zeros(n, device=device_cpu),
        "channel": ch.long(),
        "platform": torch.zeros(n, dtype=torch.long, device=device_cpu),
        "scan_angle": torch.zeros(n, device=device_cpu),
        "sat_zenith_angle": torch.zeros(n, device=device_cpu),
        "solar_zenith_angle": torch.zeros(n, device=device_cpu),
        "pressure": torch.full((n,), 500.0, device=device_cpu),
        "height": torch.full((n,), 5000.0, device=device_cpu),
        "variable_type": ch.long(),
        "report_type": torch.zeros(n, dtype=torch.long, device=device_cpu),
        "station_type": torch.zeros(n, dtype=torch.long, device=device_cpu),
        "quality_flag": torch.zeros(n, device=device_cpu),
        "mask": torch.ones(n, device=device_cpu),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_size", choices=["tiny", "base", "full_healda_like"], default="tiny")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--grid", nargs=2, type=int, default=[181, 360])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fast_cpu", action="store_true", help="use a tiny custom model for CPU-only CI smoke tests")
    args = parser.parse_args()

    if args.fast_cpu:
        # CI / login-node CPU smoke tests can become extremely slow if PyTorch
        # spawns many OpenMP threads on a heavily oversubscribed node.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    device_type = "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if device_type == "cuda" else "cpu")
    configure_accelerator_performance(device_type=device_type)

    sensors = ["atms", "amsua", "mhs", "hrs4", "gdas_prebufr"]
    observations: Dict[str, List[Dict[str, torch.Tensor]]] = {
        sensor: [make_obs(args.points, torch.device("cpu")) for _ in range(args.batch_size)] for sensor in sensors
    }
    grid = (int(args.grid[0]), int(args.grid[1]))
    batch = {"observations": observations, "target": torch.randn(args.batch_size, 26, *grid, device=device)}
    model_kwargs = dict(model_size=args.model_size, sensors=sensors, output_grid=grid)
    if args.fast_cpu:
        model_kwargs.update(dict(dim=32, depth=1, heads=4, obs_token_dim=8, sensor_embed_dim=32, patch_size=(6, 6), dropout=0.0, drop_path=0.0))
    model = HealDAXiChenRetrieval(**model_kwargs).to(device)
    loss_fn = RetrievalTQLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dtype = torch.bfloat16 if args.bf16 and device_type == "cuda" else torch.float32

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type, dtype=dtype):
        pred = model(batch)
        losses = loss_fn(pred, batch["target"])
    losses["total_loss"].backward()
    optimizer.step()
    print(f"forward_shape={tuple(pred.shape)}")
    print(f"loss={float(losses['total_loss'].detach()):.6f}")
    print("smoke_status=ok")


if __name__ == "__main__":
    main()
