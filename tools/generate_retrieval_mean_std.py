#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate ERA5 T/Q target mean/std and optional observation stats."""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np

from src.datamodules.retrieval.healda_dataset import (
    TARGET_VARS,
    XICHEN_ERA5_ALL_VARS,
    _safe_np_load,
    _as_channel_first,
    SENSOR_DIR_CANDIDATES,
    SENSOR_CHANNEL_VARS,
    SENSOR_DROP_TRAILING_CHANNELS,
    canonical_sensor,
    collect_era5_target_times,
    find_era5_fullstate_file,
    find_era5_variable_file,
)


def iter_files(root: str, limit: int = 0):
    files = sorted(glob(os.path.join(root, "**", "*.npy"), recursive=True))
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit, dtype=int)
        files = [files[i] for i in idx]
    return files


def load_target_file(path: str, target_vars: list[str], grid_shape=(181, 360)) -> np.ndarray:
    """Load a full-state ERA5 file and select the requested target variables."""
    data = _safe_np_load(path)
    if isinstance(data, dict):
        return np.stack([np.asarray(data[v]).squeeze() for v in target_vars], axis=0).astype(np.float64)
    arr = _as_channel_first(np.asarray(data), grid_shape).astype(np.float64)
    if arr.shape[0] == len(target_vars):
        return arr
    idx = [XICHEN_ERA5_ALL_VARS.index(v) for v in target_vars]
    return arr[idx]


def load_target_at_time(era5_dir: str, target_time, target_vars: list[str], grid_shape=(181, 360)) -> np.ndarray:
    """Load 26-channel T/Q label for one time from full-state or per-variable files."""
    full = find_era5_fullstate_file(era5_dir, target_time)
    if full is not None:
        return load_target_file(full, target_vars, grid_shape)
    pieces = []
    missing = []
    for var in target_vars:
        path = find_era5_variable_file(era5_dir, target_time, var)
        if path is None:
            missing.append(var)
            continue
        data = _safe_np_load(path)
        if isinstance(data, dict):
            if var in data:
                arr = np.asarray(data[var]).squeeze()
            elif len(data) == 1:
                arr = np.asarray(next(iter(data.values()))).squeeze()
            else:
                raise KeyError(f"{path} does not contain {var!r}; keys={sorted(data.keys())}")
        else:
            arr = np.asarray(data).squeeze()
        if arr.ndim == 3:
            cf = _as_channel_first(arr, grid_shape)
            if cf.shape[0] != 1:
                raise ValueError(f"{path} should contain one variable, got {cf.shape}")
            arr = cf[0]
        if arr.shape != tuple(grid_shape):
            raise ValueError(f"{path} shape={arr.shape}, expected {tuple(grid_shape)}")
        pieces.append(arr.astype(np.float64))
    if missing:
        raise FileNotFoundError(f"Missing ERA5 target files for {target_time}: {missing}")
    return np.stack(pieces, axis=0)


def generate_target_stats(
    era5_dir: str,
    scale_dir: str,
    target_vars: list[str],
    limit: int = 0,
    start_year: int = 0,
    end_year: int = 9999,
    dt_data: int = 1,
) -> None:
    times = collect_era5_target_times(era5_dir, target_vars, start_year, end_year, dt_data=dt_data)
    if not times:
        raise FileNotFoundError(
            f"No complete ERA5 target times found under {era5_dir}. Expected either HH:MM:SS.npy "
            "full-state files or per-variable files such as HH:MM:SS-t-1000.npy for all T/Q targets."
        )
    if limit and len(times) > limit:
        idx = np.linspace(0, len(times) - 1, limit, dtype=int)
        times = [times[i] for i in idx]
    n = np.zeros(len(target_vars), dtype=np.float64)
    s = np.zeros(len(target_vars), dtype=np.float64)
    ss = np.zeros(len(target_vars), dtype=np.float64)
    for i, target_time in enumerate(times, 1):
        arr = load_target_at_time(era5_dir, target_time, target_vars)
        flat = arr.reshape(arr.shape[0], -1)
        valid = np.isfinite(flat)
        n += valid.sum(axis=1)
        s += np.nan_to_num(flat, nan=0.0).sum(axis=1)
        ss += np.nan_to_num(flat * flat, nan=0.0).sum(axis=1)
        if i % 100 == 0:
            print(f"processed ERA5 target times {i}/{len(times)}")
    mean = s / np.maximum(n, 1)
    var = ss / np.maximum(n, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    os.makedirs(scale_dir, exist_ok=True)
    np.savez(os.path.join(scale_dir, "normalize_mean.npz"), **{v: np.array(mean[i], dtype=np.float32) for i, v in enumerate(target_vars)})
    np.savez(os.path.join(scale_dir, "normalize_std.npz"), **{v: np.array(std[i], dtype=np.float32) for i, v in enumerate(target_vars)})
    print(f"saved target stats to {scale_dir}; target_times={len(times)}")


def find_sensor_dir(obs_dir: str, sensor: str) -> str | None:
    for cand in SENSOR_DIR_CANDIDATES[sensor]:
        p = os.path.join(obs_dir, cand)
        if os.path.isdir(p):
            return p
    return None


def generate_obs_stats(obs_dir: str, scale_dir: str, sensors: list[str], limit: int = 256) -> None:
    out_dir = os.path.join(scale_dir, "retrieval_obs_stats")
    os.makedirs(out_dir, exist_ok=True)
    for sensor in sensors:
        root = find_sensor_dir(obs_dir, sensor)
        if root is None:
            print(f"skip {sensor}: directory not found")
            continue
        files = [f for f in iter_files(root, limit=0) if any(k in Path(f).name for k in ("brightness_temperature", "tmbrs", "obs_value", "measurement"))]
        if limit and len(files) > limit:
            idx = np.linspace(0, len(files) - 1, limit, dtype=int)
            files = [files[i] for i in idx]
        sums = None; sqs = None; ns = None
        for path in files:
            data = _safe_np_load(path)
            if isinstance(data, dict):
                continue
            try:
                arr = _as_channel_first(np.asarray(data), (181, 360)).astype(np.float64)
                drop_tail = int(SENSOR_DROP_TRAILING_CHANNELS.get(sensor, 0))
                expected_channels = len(SENSOR_CHANNEL_VARS.get(sensor, []))
                if drop_tail > 0 and expected_channels > 0 and arr.shape[0] == expected_channels + drop_tail:
                    arr = arr[:-drop_tail]
            except Exception:
                continue
            flat = arr.reshape(arr.shape[0], -1)
            valid = np.isfinite(flat)
            if sums is None:
                sums = np.zeros(arr.shape[0]); sqs = np.zeros(arr.shape[0]); ns = np.zeros(arr.shape[0])
            if arr.shape[0] != len(sums):
                continue
            ns += valid.sum(axis=1)
            sums += np.nan_to_num(flat, nan=0.0).sum(axis=1)
            sqs += np.nan_to_num(flat * flat, nan=0.0).sum(axis=1)
        if sums is None:
            print(f"skip {sensor}: no numeric measurement files")
            continue
        mean = sums / np.maximum(ns, 1)
        std = np.sqrt(np.maximum(sqs / np.maximum(ns, 1) - mean * mean, 1e-12))
        np.savez(os.path.join(out_dir, f"{sensor}.npz"), mean=mean.astype(np.float32), std=std.astype(np.float32))
        print(f"saved {sensor} obs stats: channels={len(mean)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5_dir", required=True)
    parser.add_argument("--scale_dir", required=True)
    parser.add_argument("--target_vars", nargs="+", default=TARGET_VARS)
    parser.add_argument("--limit", type=int, default=0, help="Limit ERA5 target times for a quick dry run")
    parser.add_argument("--start_year", type=int, default=0)
    parser.add_argument("--end_year", type=int, default=9999)
    parser.add_argument("--dt_data", type=int, default=1, help="Use 1 for all hourly files, 6 for 00/06/12/18 UTC style stats")
    parser.add_argument("--include_obs_stats", action="store_true")
    parser.add_argument("--obs_dir", default=None)
    parser.add_argument("--sensors", nargs="+", default=["atms", "amsua", "mhs", "hrs4", "gdas_prebufr"])
    parser.add_argument("--obs_limit", type=int, default=256)
    args = parser.parse_args()
    generate_target_stats(args.era5_dir, args.scale_dir, args.target_vars, args.limit, args.start_year, args.end_year, args.dt_data)
    if args.include_obs_stats:
        if not args.obs_dir:
            raise ValueError("--include_obs_stats requires --obs_dir")
        generate_obs_stats(args.obs_dir, args.scale_dir, [canonical_sensor(s) for s in args.sensors], args.obs_limit)


if __name__ == "__main__":
    main()
