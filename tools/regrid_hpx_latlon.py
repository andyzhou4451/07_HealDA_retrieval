#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HPX <-> lat-lon regridding utility for retrieval outputs and labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.models.retrieval.healda_regrid import hpx_to_latlon, latlon_to_hpx, regrid_consistency_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["latlon_to_hpx", "hpx_to_latlon", "check"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--output_grid", nargs=2, type=int, default=[181, 360])
    args = parser.parse_args()
    x = torch.from_numpy(np.load(args.input)).float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if args.mode == "latlon_to_hpx":
        y = latlon_to_hpx(x, nside=args.nside)
        np.save(args.output or "latlon_to_hpx.npy", y.cpu().numpy())
    elif args.mode == "hpx_to_latlon":
        y = hpx_to_latlon(x, output_grid=args.output_grid, nside=args.nside)
        np.save(args.output or "hpx_to_latlon.npy", y.cpu().numpy())
    else:
        err = regrid_consistency_check(x, nside=args.nside, output_grid=args.output_grid)
        print(f"relative_rms_error={err:.6e}")


if __name__ == "__main__":
    main()
