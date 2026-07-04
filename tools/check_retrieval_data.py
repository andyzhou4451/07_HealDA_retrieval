#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect multi-source retrieval data without assuming fixed channel counts."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import timedelta
from glob import glob
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Iterable

import numpy as np

from src.datamodules.retrieval.healda_dataset import (
    SENSOR_DIR_CANDIDATES,
    SATELLITE_SENSORS,
    SENSOR_CHANNEL_VARS,
    SENSOR_DROP_TRAILING_CHANNELS,
    canonical_sensor,
    datetime_path,
    parse_datetime_from_path,
    _safe_np_load,
    _as_channel_first,
    TARGET_VARS,
    XICHEN_ERA5_ALL_VARS,
    collect_era5_target_times,
    find_era5_fullstate_file,
    find_era5_variable_file,
)

MEAS_SUFFIXES = {
    "sat": ("brightness_temperature_value.npy", "tmbrs_value.npy", "obs_value.npy", "measurement.npy", "value.npy"),
    "conv": ("obs_value.npy", "observation_value.npy", "measurement.npy", "value.npy"),
}


def find_dir(obs_dir: str, sensor: str) -> str | None:
    for cand in SENSOR_DIR_CANDIDATES[sensor]:
        p = os.path.join(obs_dir, cand)
        if os.path.isdir(p):
            return p
    return None


def sample_npy_files(root: str, limit: int = 32) -> list[str]:
    files = sorted(glob(os.path.join(root, "**", "*.npy"), recursive=True))
    if len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit, dtype=int)
        return [files[i] for i in idx]
    return files


def summarize_array(path: str) -> tuple[tuple[int, ...], int | None, float, float, float]:
    arr = _safe_np_load(path)
    if isinstance(arr, dict):
        vals = [np.asarray(v).reshape(-1) for v in arr.values() if np.asarray(v).size]
        data = np.concatenate(vals) if vals else np.array([])
        shape = tuple(np.asarray(next(iter(arr.values()))).shape) if arr else ()
        channels = None
    else:
        arr = np.asarray(arr)
        data = arr.reshape(-1)
        shape = tuple(arr.shape)
        try:
            cf = _as_channel_first(arr, (181, 360))
            channels = cf.shape[0]
        except Exception:
            channels = arr.shape[0] if arr.ndim >= 3 else None
    if data.size == 0:
        return shape, channels, float("nan"), float("nan"), float("nan")
    finite = np.isfinite(data)
    miss = 1.0 - float(finite.mean())
    return shape, channels, miss, float(np.nanmin(data)), float(np.nanmax(data))


def collect_times(files: Iterable[str]) -> list:
    out = []
    for path in files:
        t = parse_datetime_from_path(path)
        if t is not None:
            out.append(t)
    return sorted(set(out))


def find_time_file(root: str, t, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        path = datetime_path(root, t, suffix)
        if os.path.exists(path):
            return path
    base = os.path.join(root, f"{t.year:04d}", f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
    if os.path.isdir(base):
        stamp = f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
        for suffix in suffixes:
            matches = sorted(glob(os.path.join(base, f"{stamp}*{suffix}")))
            if matches:
                return matches[0]
    return None


def count_valid_points(root: str, sensor: str, target_time, start_h: int = -21, end_h: int = 3, dt_h: int = 3) -> int:
    suffixes = MEAS_SUFFIXES["sat" if sensor in SATELLITE_SENSORS else "conv"]
    count = 0
    for h in range(start_h, end_h + 1, dt_h):
        path = find_time_file(root, target_time + timedelta(hours=h), suffixes)
        if path is None:
            continue
        arr = _safe_np_load(path)
        if isinstance(arr, dict):
            vals = [np.asarray(v).reshape(-1) for k, v in arr.items() if any(x in k.lower() for x in ("obs", "measurement", "value", "brightness"))]
            data = np.concatenate(vals) if vals else np.array([])
        else:
            data = np.asarray(arr).reshape(-1)
        count += int(np.isfinite(data).sum())
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_dir", required=True)
    parser.add_argument("--era5_dir", required=True)
    parser.add_argument("--sensors", nargs="+", default=["atms", "amsua", "mhs", "hrs4", "gdas_prebufr"])
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--count_sample_times", type=int, default=5)
    args = parser.parse_args()

    sensors = [canonical_sensor(s) for s in args.sensors]
    print("=== Observation sources ===")
    sensor_times = {}
    sensor_roots = {}
    for sensor in sensors:
        root = find_dir(args.obs_dir, sensor)
        print(f"\n[{sensor}]")
        if root is None:
            print("  directory: NOT FOUND")
            continue
        sensor_roots[sensor] = root
        files = sorted(glob(os.path.join(root, "**", "*.npy"), recursive=True))
        sensor_times[sensor] = collect_times(files)
        print(f"  directory: {root}")
        print(f"  npy file count: {len(files)}")
        if sensor_times[sensor]:
            print(f"  sample time range: {sensor_times[sensor][0]} -> {sensor_times[sensor][-1]} ({len(sensor_times[sensor])} unique times)")
        suffix_counts = Counter(Path(f).name.split("-", maxsplit=1)[-1] for f in files)
        print(f"  common file suffixes: {suffix_counts.most_common(10)}")
        print("  lat/lon range assumption for gridded 1.0deg files: lat=[-90,90], lon=[0,360)")
        for path in sample_npy_files(root, limit=min(args.limit, 8))[:4]:
            shape, channels, miss, vmin, vmax = summarize_array(path)
            print(f"  sample {Path(path).name}: shape={shape}, channels={channels}, missing={miss:.3f}, range=[{vmin:.3g}, {vmax:.3g}]")
        schema_files = glob(os.path.join(root, "*schema*.json"))
        expected_channels = len(SENSOR_CHANNEL_VARS.get(sensor, []))
        drop_tail = int(SENSOR_DROP_TRAILING_CHANNELS.get(sensor, 0))
        if expected_channels:
            print(f"  XiChen expected measurement channels: {expected_channels}" + (f"; loader drops trailing {drop_tail} redundant channel(s)" if drop_tail else ""))
        print(f"  schema files / field names source: {[Path(p).name for p in schema_files[:4]] or 'not found; loader will infer from file layout'}")
        if sensor == "gdas_prebufr":
            value_files = [f for f in files if "obs_value" in Path(f).name or "measurement" in Path(f).name]
            if value_files:
                arr = np.asarray(_safe_np_load(value_files[0])).squeeze()
                print(f"  GDAS PREPBUFR first obs/measurement shape: {arr.shape}")
                finite = np.isfinite(arr)
                print(f"  GDAS PREPBUFR finite entries in first file: {int(finite.sum())}")
                print("  GDAS PREPBUFR variable/report/station/pressure/height names are read from schema/auxiliary files when present.")

    print("\n=== ERA5 labels ===")
    era5_files = sorted(glob(os.path.join(args.era5_dir, "**", "*.npy"), recursive=True))
    all_era5_times = collect_times(era5_files)
    era5_times = collect_era5_target_times(args.era5_dir, TARGET_VARS, 0, 9999, dt_data=1)
    print(f"  ERA5 file count: {len(era5_files)}")
    if all_era5_times:
        print(f"  raw time range: {all_era5_times[0]} -> {all_era5_times[-1]} ({len(all_era5_times)} unique times)")
    print(f"  complete 26-channel T/Q target times: {len(era5_times)}")
    if era5_times:
        t0 = era5_times[0]
        print(f"  complete target time range: {era5_times[0]} -> {era5_times[-1]}")
        full = find_era5_fullstate_file(args.era5_dir, t0)
        if full is not None:
            print("  layout: full-state file per time")
            arr = _safe_np_load(full)
            if isinstance(arr, dict):
                keys = sorted(arr.keys())
                print(f"  sample full-state file: {full}")
                print(f"  first file keys count={len(keys)}")
                print(f"  all 26 T/Q target variables present: {set(TARGET_VARS).issubset(keys)}")
                print(f"  pressure levels complete: {all(f't-{p}' in keys and f'q-{p}' in keys for p in [50,100,150,200,250,300,400,500,600,700,850,925,1000])}")
            else:
                arr = np.asarray(arr)
                print(f"  sample full-state file: {full}")
                print(f"  first file shape: {arr.shape}")
                try:
                    cf = _as_channel_first(arr, (181, 360))
                    ch = cf.shape[0]
                except Exception:
                    ch = arr.shape[0] if arr.ndim >= 3 else None
                if ch == 26:
                    print("  T/Q target status: file already has 26 channels")
                elif ch and ch >= len(XICHEN_ERA5_ALL_VARS):
                    print("  T/Q target status: selectable from XiChen full state order; all 13 levels covered by config")
                else:
                    print("  T/Q target status: CHECK era5_all_vars or pre-extract labels")
        else:
            print("  layout: per-variable files, e.g. HH:MM:SS-t-1000.npy")
            sample_var = TARGET_VARS[0]
            sample_path = find_era5_variable_file(args.era5_dir, t0, sample_var)
            sample_arr = np.asarray(_safe_np_load(sample_path)).squeeze() if sample_path else np.array([])
            print(f"  sample variable file: {sample_path}")
            print(f"  sample variable shape: {sample_arr.shape}")
            missing = [v for v in TARGET_VARS if find_era5_variable_file(args.era5_dir, t0, v) is None]
            print(f"  all 26 T/Q target variables present at first complete time: {not missing}")
            print(f"  pressure levels complete: {not missing}")

    print("\n=== Observation/ERA5 time matching ===")
    if era5_times:
        era5_set = set(era5_times)
        for sensor, times in sensor_times.items():
            direct = len(era5_set.intersection(times)) if times else 0
            print(f"  {sensor}: exact matched target times {direct}/{len(era5_times)}")
        print("\n=== Observation counts for sampled target times ===")
        for t in era5_times[: args.count_sample_times]:
            pieces = []
            for sensor, root in sensor_roots.items():
                pieces.append(f"{sensor}={count_valid_points(root, sensor, t)}")
            print(f"  {t}: " + ", ".join(pieces))


if __name__ == "__main__":
    main()
