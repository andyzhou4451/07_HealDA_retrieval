# -*- coding: utf-8 -*-
"""Datasets for HealDA-style multi-source observation -> ERA5 T/Q retrieval.

The dataset deliberately keeps observations as variable-length point clouds.  It
can read the gridded 1.0 degree XiChen-style NPY files by flattening observed
pixels into scalar observations, and it can also consume point-like npy/npz files
when fields such as ``lat``/``lon``/``observation`` are available.

Returned target shape is always ``[26, 181, 360]`` in this order:
``t-50 ... t-1000, q-50 ... q-1000``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PRESSURE_LEVELS: List[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
TARGET_VARS: List[str] = [*(f"t-{p}" for p in PRESSURE_LEVELS), *(f"q-{p}" for p in PRESSURE_LEVELS)]

SENSOR_ALIAS: Dict[str, str] = {
    "atms": "atms",
    "1batms": "atms",
    "amsu-a": "amsua",
    "amsua": "amsua",
    "amsu_a": "amsua",
    "1bamsua": "amsua",
    "mhs": "mhs",
    "1bmhs": "mhs",
    "hirs": "hrs4",
    "hirs-4": "hrs4",
    "hirs4": "hrs4",
    "hrs4": "hrs4",
    "1bhrs4": "hrs4",
    "gdas_prebufr": "gdas_prebufr",
    "gdas-prebufr": "gdas_prebufr",
    "gdas_prepbufr": "gdas_prebufr",
    "gdas-prepbufr": "gdas_prebufr",
    "prepbufr": "gdas_prebufr",
    "prebufr": "gdas_prebufr",
    "gdas_prebufr_corrected_npy_1.0deg": "gdas_prebufr",
    "gdas_prepbufr_merged_npy_1.0deg": "gdas_prebufr",
    "GDAS_prebufr_corrected_npy_1.0deg": "gdas_prebufr",
    "GDAS_prepbufr_merged_npy_1.0deg": "gdas_prebufr",
}

SENSOR_DIR_CANDIDATES: Dict[str, Sequence[str]] = {
    "atms": ("ATMS", "atms", "1batms_merged_npy_1.0deg", "1batms"),
    "amsua": ("AMSU-A", "AMSUA", "amsua", "1bamsua_merged_npy_1.0deg", "1bamsua"),
    "mhs": ("MHS", "mhs", "1bmhs_merged_npy_1.0deg", "1bmhs"),
    "hrs4": ("HIRS4", "HRS4", "hrs4", "hirs4", "1bhrs4_merged_npy_1.0deg", "1bhrs4"),
    "gdas_prebufr": (
        "GDAS_prebufr_corrected_npy_1.0deg",
        "GDAS_prepbufr_merged_npy_1.0deg",
        "gdas_prebufr",
        "prepbufr",
    ),
}

SATELLITE_SENSORS = {"atms", "amsua", "mhs", "hrs4"}
CONVENTIONAL_SENSORS = {"gdas_prebufr"}

# XiChen full 13-level state order, used only to select labels when an ERA5 file
# stores all channels.  It is copied from the existing XiChen configs, not from
# file-name guesswork.
XICHEN_ERA5_ALL_VARS: List[str] = [
    "t2m", "u10", "v10", "msl",
    *(f"z-{p}" for p in PRESSURE_LEVELS),
    *(f"u-{p}" for p in PRESSURE_LEVELS),
    *(f"v-{p}" for p in PRESSURE_LEVELS),
    *(f"t-{p}" for p in PRESSURE_LEVELS),
    *(f"q-{p}" for p in PRESSURE_LEVELS),
]

# XiChen 观测目录中已有 normalize_mean.npz / normalize_std.npz；这些变量顺序
# 与原可运行工程的 datamodule 配置一致，用于把每个传感器现有归一化文件
# 转成当前 point-cloud loader 可直接按 channel 下标索引的 mean/std 数组。
SENSOR_CHANNEL_VARS: Dict[str, List[str]] = {
    "atms": [f"tmbrs_{i}" for i in range(1, 23)],
    "amsua": [f"tmbrs_{i}" for i in range(1, 16)],
    "mhs": [f"tmbrs_{i}" for i in range(1, 6)],
    "hrs4": [f"tmbrs_{i}" for i in range(1, 20)],
    "gdas_prebufr": list(XICHEN_ERA5_ALL_VARS),
}

# 原 XiChen HRS4 reader 会丢弃 tmbrs_value.npy 的最后一个冗余通道；这里保留
# 同样行为，避免模型把无效第 20 通道当作真实 HIRS4 观测。
SENSOR_DROP_TRAILING_CHANNELS: Dict[str, int] = {"hrs4": 1}

_TIME_PATTERNS = (
    re.compile(
        r"(?P<year>\d{4})[-_](?P<month>\d{2})[-_](?P<day>\d{2})[\\/]+"
        r"(?P<hour>\d{2})[:_-](?P<minute>\d{2})[:_-](?P<second>\d{2})"
    ),
    re.compile(
        r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2}).*?"
        r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
    ),
)


def canonical_sensor(name: str) -> str:
    """Return canonical sensor name and reject satwnd/ascat for this task."""
    key = str(name).strip()
    canon = SENSOR_ALIAS.get(key) or SENSOR_ALIAS.get(key.lower())
    if canon is None:
        raise ValueError(f"Unknown sensor alias {name!r}. Allowed aliases: {sorted(SENSOR_ALIAS)}")
    if canon in {"satwnd", "ascat"}:
        raise ValueError("satwnd/ascat are intentionally disabled for the T/Q retrieval task")
    return canon


def parse_datetime_from_path(path: str | os.PathLike[str]) -> Optional[datetime]:
    """Parse XiChen-style ``YYYY-MM-DD/HH:MM:SS`` or compact variants from a path."""
    text = str(path)
    for pattern in _TIME_PATTERNS:
        for match in pattern.finditer(text):
            gd = {k: int(v) for k, v in match.groupdict().items()}
            try:
                return datetime(gd["year"], gd["month"], gd["day"], gd["hour"], gd["minute"], gd["second"])
            except ValueError:
                continue
    return None


def datetime_path(root: str | os.PathLike[str], t: datetime, suffix: str) -> str:
    """XiChen-style time path helper."""
    return os.path.join(
        str(root),
        f"{t.year:04d}",
        f"{t.year:04d}-{t.month:02d}-{t.day:02d}",
        f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}-{suffix}",
    )


def datetime_era5_path(root: str | os.PathLike[str], t: datetime) -> str:
    """XiChen-style full-state ERA5 file path helper.

    Some XiChen datasets store a complete state in one file named
    ``HH:MM:SS.npy``.  The retrieval dataset also supports the per-variable
    layout used on ``/public02`` where each target is stored separately, for
    example ``HH:MM:SS-t-1000.npy``.
    """
    return os.path.join(
        str(root),
        f"{t.year:04d}",
        f"{t.year:04d}-{t.month:02d}-{t.day:02d}",
        f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.npy",
    )


def datetime_era5_variable_path(root: str | os.PathLike[str], t: datetime, var: str) -> str:
    """Path for per-variable ERA5 labels, e.g. ``22:00:00-t-1000.npy``."""
    return os.path.join(
        str(root),
        f"{t.year:04d}",
        f"{t.year:04d}-{t.month:02d}-{t.day:02d}",
        f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}-{var}.npy",
    )


def _era5_day_dir(root: str | os.PathLike[str], t: datetime) -> str:
    return os.path.join(str(root), f"{t.year:04d}", f"{t.year:04d}-{t.month:02d}-{t.day:02d}")


def _era5_stamp(t: datetime) -> str:
    return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"


def find_era5_fullstate_file(root: str | os.PathLike[str], t: datetime) -> Optional[str]:
    """Return a complete-state ERA5 file for ``t`` if one exists.

    This intentionally does not match ``HH:MM:SS-t-1000.npy``; per-variable
    files are handled by :func:`find_era5_variable_file`.
    """
    exact = datetime_era5_path(root, t)
    if os.path.exists(exact):
        return exact
    base = _era5_day_dir(root, t)
    if not os.path.isdir(base):
        return None
    stamp = _era5_stamp(t)
    for name in (f"{stamp}.npy", f"{stamp}-era5.npy", f"{stamp}-state.npy", f"{stamp}-all.npy"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None


def find_era5_variable_file(root: str | os.PathLike[str], t: datetime, var: str) -> Optional[str]:
    """Return one per-variable ERA5 file for ``t`` and ``var`` if present."""
    exact = datetime_era5_variable_path(root, t, var)
    if os.path.exists(exact):
        return exact
    base = _era5_day_dir(root, t)
    if not os.path.isdir(base):
        return None
    stamp = _era5_stamp(t)
    candidates = [
        f"{stamp}-{var}.npy",
        f"{stamp}_{var}.npy",
        f"{stamp}-{var.replace('-', '_')}.npy",
        f"{stamp}_{var.replace('-', '_')}.npy",
    ]
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    # Last-resort support for prefixes such as HH:MM:SS-era5-t-1000.npy.
    patterns = [
        os.path.join(base, f"{stamp}*{var}.npy"),
        os.path.join(base, f"{stamp}*{var.replace('-', '_')}.npy"),
    ]
    for pattern in patterns:
        matches = sorted(glob(pattern))
        if matches:
            return matches[0]
    return None


def era5_time_has_targets(root: str | os.PathLike[str], t: datetime, target_vars: Sequence[str]) -> bool:
    """True when either a full-state file or all per-variable target files exist."""
    if find_era5_fullstate_file(root, t) is not None:
        return True
    return all(find_era5_variable_file(root, t, var) is not None for var in target_vars)


def _era5_variable_name_from_file(path: str | os.PathLike[str]) -> Optional[str]:
    """Extract ``t-1000`` from ``HH:MM:SS-t-1000.npy`` style names."""
    name = Path(path).name
    m = re.match(r"^\d{2}:\d{2}:\d{2}[-_](?P<var>.+)\.npy$", name)
    if not m:
        return None
    var = m.group("var")
    # Undo the common underscore variant only for variable prefixes.
    if re.match(r"^[a-zA-Z]+_\d+$", var):
        head, lev = var.rsplit("_", 1)
        var = f"{head}-{lev}"
    return var


def collect_era5_target_times(
    root: str | os.PathLike[str],
    target_vars: Sequence[str],
    start_year: int,
    end_year: int,
    dt_data: int = 1,
) -> List[datetime]:
    """Scan nested ERA5 labels and return times with complete T/Q targets.

    The implementation walks ``YYYY/YYYY-MM-DD`` directories with ``os.scandir``
    instead of materialising a multi-million-file recursive glob.  This is much
    friendlier to shared HPC filesystems when ``dt_data=1`` and the dataset has
    one ``*.npy`` per variable per hour.
    """
    root = str(root)
    if not os.path.isdir(root):
        return []
    target_set = set(map(str, target_vars))
    full_times: set[datetime] = set()
    var_times: Dict[datetime, set[str]] = {}
    for year in range(int(start_year), int(end_year)):
        year_dir = os.path.join(root, f"{year:04d}")
        if not os.path.isdir(year_dir):
            continue
        try:
            day_entries = sorted((e for e in os.scandir(year_dir) if e.is_dir()), key=lambda e: e.name)
        except OSError:
            continue
        for day_entry in day_entries:
            try:
                day = datetime.strptime(day_entry.name, "%Y-%m-%d")
            except ValueError:
                continue
            try:
                file_entries = sorted((e for e in os.scandir(day_entry.path) if e.is_file() and e.name.endswith(".npy")), key=lambda e: e.name)
            except OSError:
                continue
            for entry in file_entries:
                stamp_match = re.match(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<rest>.*)\.npy$", entry.name)
                if not stamp_match:
                    continue
                hour = int(stamp_match.group("h"))
                minute = int(stamp_match.group("m"))
                second = int(stamp_match.group("s"))
                if dt_data > 1 and hour % int(dt_data) != 0:
                    continue
                t = day.replace(hour=hour, minute=minute, second=second)
                rest = stamp_match.group("rest")
                if rest in {"", "-era5", "-state", "-all"}:
                    full_times.add(t)
                    continue
                var = _era5_variable_name_from_file(entry.path)
                if var in target_set:
                    var_times.setdefault(t, set()).add(var)
                    continue
                for candidate in target_set:
                    if rest.endswith("-" + candidate) or rest.endswith("_" + candidate.replace("-", "_")):
                        var_times.setdefault(t, set()).add(candidate)
                        break
    complete = set(full_times)
    complete.update(t for t, vars_present in var_times.items() if target_set.issubset(vars_present))
    return sorted(complete)


def _safe_np_load(path: str | os.PathLike[str]) -> Any:
    """Load NPY/NPZ safely; plain NPY arrays use mmap to lower host-memory pressure."""
    path = str(path)
    mmap_mode = "r" if path.endswith(".npy") else None
    data = np.load(path, allow_pickle=True, mmap_mode=mmap_mode)
    if isinstance(data, np.lib.npyio.NpzFile):
        with data as npz:
            return {k: npz[k] for k in npz.files}
    if getattr(data, "shape", None) == () and data.dtype == object:
        obj = data.item()
        if isinstance(obj, Mapping):
            return dict(obj)
    return data


def _as_channel_first(arr: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    """Convert [H,W], [C,H,W], or [H,W,C] arrays to [C,H,W]."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        h, w = grid_shape
        if arr.shape[-2:] == (h, w):
            return arr
        if arr.shape[:2] == (h, w):
            return np.moveaxis(arr, -1, 0)
    if arr.ndim == 4 and arr.shape[0] == 2 and arr.shape[1] == len(PRESSURE_LEVELS):
        return arr.reshape(2 * len(PRESSURE_LEVELS), arr.shape[-2], arr.shape[-1])
    raise ValueError(f"Cannot interpret array shape {arr.shape} as channel-first grid {grid_shape}")


def _to_float_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _empty_obs() -> Dict[str, torch.Tensor]:
    f = torch.empty(0, dtype=torch.float32)
    l = torch.empty(0, dtype=torch.long)
    return {
        "measurement": f,
        "lat": f,
        "lon": f,
        "relative_time": f,
        "channel": l,
        "platform": l,
        "scan_angle": f,
        "sat_zenith_angle": f,
        "solar_zenith_angle": f,
        "pressure": f,
        "height": f,
        "variable_type": l,
        "report_type": l,
        "station_type": l,
        "quality_flag": f,
        "mask": f,
    }


@dataclass(frozen=True)
class SensorFiles:
    measurement_suffixes: Tuple[str, ...]
    aux_suffixes: Tuple[str, ...]
    mask_suffixes: Tuple[str, ...]


SAT_FILES = SensorFiles(
    measurement_suffixes=(
        "brightness_temperature_value.npy",
        "tmbrs_value.npy",
        "obs_value.npy",
        "measurement.npy",
        "value.npy",
    ),
    aux_suffixes=("auxiliary_value.npy", "metadata_value.npy", "aux_value.npy"),
    mask_suffixes=("mask.npy", "obs_mask.npy"),
)
CONV_FILES = SensorFiles(
    measurement_suffixes=("obs_value.npy", "observation_value.npy", "measurement.npy", "value.npy"),
    aux_suffixes=("auxiliary_value.npy", "metadata_value.npy", "pressure_value.npy", "height_value.npy"),
    mask_suffixes=("mask.npy", "obs_mask.npy"),
)


class HealDARetrievalDataset(Dataset):
    """Variable-length point-cloud observation dataset for retrieval.

    Parameters mirror the Hydra datamodule config.  The loader never reads ERA5
    targets as model input; ERA5 is used only for ``target``.
    """

    def __init__(
        self,
        obs_dir: str,
        era5_dir: str,
        scale_dir: str,
        mode: str,
        sensors: Sequence[str],
        target_variables: Sequence[str] = TARGET_VARS,
        pressure_levels: Sequence[int] = PRESSURE_LEVELS,
        era5_all_vars: Sequence[str] = XICHEN_ERA5_ALL_VARS,
        grid_shape: Sequence[int] = (181, 360),
        obs_window: Mapping[str, int] | None = None,
        no_lookahead: bool = False,
        no_lookahead_window: Mapping[str, int] | None = None,
        dt_data: int = 6,
        dt_obs: int = 3,
        start_year: int = 2016,
        end_year: int = 2022,
        debug: bool = False,
        max_debug_samples: int = 8,
        max_points_per_sensor: int = 250_000,
        strict_time_index: bool = False,
        target_cache_size: int = 16,
        normalize_target: bool = True,
        normalize_obs: bool = True,
        require_obs_stats: bool = False,
        require_sensor_dirs: bool = True,
        filter_empty_observation_windows: bool = True,
        min_required_sensors: int = 1,
        require_complete_obs_window: bool = False,
        qc: Optional[Mapping[str, Sequence[float]]] = None,
        obs_default_normalization: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> None:
        super().__init__()
        self.obs_dir = os.path.abspath(os.path.expanduser(obs_dir))
        self.era5_dir = os.path.abspath(os.path.expanduser(era5_dir))
        self.scale_dir = os.path.abspath(os.path.expanduser(scale_dir))
        self.mode = mode
        self.sensors = [canonical_sensor(s) for s in sensors]
        if any(s in {"satwnd", "ascat"} for s in self.sensors):
            raise ValueError("Retrieval task must not read satwnd/ascat unless explicitly reconfigured.")
        self.target_variables = list(target_variables)
        self.pressure_levels = list(pressure_levels)
        self.era5_all_vars = list(era5_all_vars)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.dt_data = int(dt_data)
        self.dt_obs = int(dt_obs)
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.debug = bool(debug)
        self.max_debug_samples = int(max_debug_samples)
        self.max_points_per_sensor = int(max_points_per_sensor)
        self.strict_time_index = bool(strict_time_index)
        self.target_cache_size = max(int(target_cache_size), 0)
        self._target_cache: "OrderedDict[datetime, torch.Tensor]" = OrderedDict()
        self.normalize_target = bool(normalize_target)
        self.normalize_obs = bool(normalize_obs)
        self.require_obs_stats = bool(require_obs_stats)
        self.require_sensor_dirs = bool(require_sensor_dirs)
        self.filter_empty_observation_windows = bool(filter_empty_observation_windows)
        self.min_required_sensors = max(int(min_required_sensors), 0)
        self.require_complete_obs_window = bool(require_complete_obs_window)
        self.qc = dict(qc or {})
        self.obs_default_normalization = dict(obs_default_normalization or {})

        window = no_lookahead_window if no_lookahead else obs_window
        window = window or {"start_hours": -21, "end_hours": 3}
        self.window_start = int(window.get("start_hours", -21))
        self.window_end = int(window.get("end_hours", 3))

        if not os.path.isdir(self.era5_dir):
            raise FileNotFoundError(
                f"ERA5 directory does not exist: {self.era5_dir}. Expected nested folders like "
                "YYYY/YYYY-MM-DD/HH:MM:SS-t-1000.npy. Override paths.era5_dir if needed."
            )
        self.sensor_dirs = {sensor: self._find_sensor_dir(sensor) for sensor in self.sensors}
        self.sensor_schema = {sensor: self._load_sensor_schema(sensor) for sensor in self.sensors}
        self.sensor_stats = {sensor: self._load_obs_stats(sensor) for sensor in self.sensors}
        self.target_mean, self.target_std = self._load_target_stats() if self.normalize_target else (None, None)
        self.target_times = self._build_target_times()
        if self.filter_empty_observation_windows and self.min_required_sensors > 0:
            before = len(self.target_times)
            self.target_times = [t for t in self.target_times if self._time_has_required_observations(t)]
            after = len(self.target_times)
            if before and after < before and int(os.environ.get("RANK", "0")) == 0:
                warnings.warn(
                    f"Filtered ERA5 target times by observation-window availability: {before} -> {after}. "
                    "This follows the working XiChen retrieval pattern and avoids all-empty observation batches. "
                    "Set datamodule.data.filter_empty_observation_windows=false to disable.",
                    RuntimeWarning,
                )
        if self.debug:
            self.target_times = self.target_times[: self.max_debug_samples]
        if not self.target_times:
            raise FileNotFoundError(
                f"No usable retrieval samples found for split={mode} under ERA5={self.era5_dir} and OBS={self.obs_dir}. "
                "Expected ERA5 labels as YYYY/YYYY-MM-DD/HH:MM:SS.npy or split files such as "
                "HH:MM:SS-t-1000.npy plus at least one configured observation sensor in the HealDA window. "
                "Check paths, start/end years, dt_data, strict_time_index, and filter_empty_observation_windows."
            )

    def __len__(self) -> int:
        return len(self.target_times)

    def _find_sensor_dir(self, sensor: str) -> str:
        for cand in SENSOR_DIR_CANDIDATES[sensor]:
            p = os.path.join(self.obs_dir, cand)
            if os.path.isdir(p):
                return p
        expected = [os.path.join(self.obs_dir, cand) for cand in SENSOR_DIR_CANDIDATES[sensor]]
        if self.require_sensor_dirs:
            raise FileNotFoundError(
                f"Observation directory for sensor {sensor!r} was not found. Checked:\n  - "
                + "\n  - ".join(expected)
                + "\nUse paths.obs_dir=/public02/data/Observation/observation_npy/ or disable "
                "datamodule.data.require_sensor_dirs only for synthetic smoke tests."
            )
        return expected[0]

    def _load_sensor_schema(self, sensor: str) -> Mapping[str, Any]:
        root = self.sensor_dirs[sensor]
        schema_alias = "prepbufr" if sensor == "gdas_prebufr" else sensor
        patterns = [
            os.path.join(root, f"{schema_alias}_1.0deg_schema.json"),
            os.path.join(root, f"{sensor}_1.0deg_schema.json"),
            os.path.join(root, "schema.json"),
            os.path.join(root, "*_schema.json"),
        ]
        for pattern in patterns:
            for path in sorted(glob(pattern)):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    warnings.warn(f"Failed to read schema {path}: {exc}")
        return {}

    def _load_obs_stats(self, sensor: str) -> Optional[Dict[str, np.ndarray]]:
        candidates = [
            os.path.join(self.scale_dir, "retrieval_obs_stats", f"{sensor}.npz"),
            os.path.join(self.sensor_dirs[sensor], "retrieval_obs_stats.npz"),
            os.path.join(self.sensor_dirs[sensor], "sensor_stats.npz"),
        ]
        for path in candidates:
            if os.path.exists(path):
                with np.load(path) as npz:
                    if "mean" in npz and "std" in npz:
                        return {"mean": np.asarray(npz["mean"], dtype=np.float32), "std": np.asarray(npz["std"], dtype=np.float32)}
                    return {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}

        # Reuse the existing XiChen normalization files.  This is the same data
        # source used by the working reference retrieval project, but converted
        # into dense channel-indexed arrays for the point-cloud loader.
        sensor_stats = self._load_channel_stats_from_npz(self.sensor_dirs[sensor], SENSOR_CHANNEL_VARS.get(sensor, []))
        if sensor_stats is not None:
            return sensor_stats
        if sensor == "gdas_prebufr":
            conventional_stats = self._load_channel_stats_from_npz(self.scale_dir, SENSOR_CHANNEL_VARS["gdas_prebufr"])
            if conventional_stats is not None:
                return conventional_stats

        if self.require_obs_stats:
            raise FileNotFoundError(
                f"Observation normalization stats for sensor {sensor!r} were not found. Checked retrieval_obs_stats, "
                "sensor_stats, the sensor directory normalize_mean/std files, and ERA5 scale_dir normalize_mean/std for GDAS. "
                f"Run: python tools/generate_retrieval_mean_std.py --obs_dir {self.obs_dir} "
                f"--era5_dir {self.era5_dir} --scale_dir {self.scale_dir} --include_obs_stats"
            )
        return None

    @staticmethod
    def _load_channel_stats_from_npz(directory: str, variables: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
        if not variables:
            return None
        mean_path = os.path.join(directory, "normalize_mean.npz")
        std_path = os.path.join(directory, "normalize_std.npz")
        if not (os.path.exists(mean_path) and os.path.exists(std_path)):
            return None
        try:
            with np.load(mean_path) as mean_npz, np.load(std_path) as std_npz:
                if not all(var in mean_npz and var in std_npz for var in variables):
                    return None
                mean = np.concatenate([np.asarray(mean_npz[var], dtype=np.float32).reshape(1) for var in variables])
                std = np.concatenate([np.asarray(std_npz[var], dtype=np.float32).reshape(1) for var in variables])
        except Exception as exc:
            warnings.warn(f"Failed to read channel normalization from {directory}: {exc}")
            return None
        return {"mean": mean.astype(np.float32), "std": np.maximum(std.astype(np.float32), 1.0e-6)}

    def _load_target_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        mean_path = os.path.join(self.scale_dir, "normalize_mean.npz")
        std_path = os.path.join(self.scale_dir, "normalize_std.npz")
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(
                "ERA5 target mean/std files are missing. Expected both files:\n"
                f"  {mean_path}\n  {std_path}\n"
                "Generate them with:\n"
                f"  python tools/generate_retrieval_mean_std.py --era5_dir {self.era5_dir} "
                f"--scale_dir {self.scale_dir} --target_vars {' '.join(self.target_variables)}"
            )
        mean_npz = dict(np.load(mean_path))
        std_npz = dict(np.load(std_path))
        missing = [v for v in self.target_variables if v not in mean_npz or v not in std_npz]
        if missing:
            raise KeyError(
                f"Missing target variable stats in {self.scale_dir}: {missing}. "
                "Regenerate normalize_mean.npz / normalize_std.npz for the T/Q-13 target list."
            )
        mean = np.concatenate([np.asarray(mean_npz[v]).reshape(1) for v in self.target_variables]).astype(np.float32)
        std = np.concatenate([np.asarray(std_npz[v]).reshape(1) for v in self.target_variables]).astype(np.float32)
        std = np.where(std == 0, 1.0, std)
        return mean, std

    def _build_target_times(self) -> List[datetime]:
        times: List[datetime] = []
        if self.strict_time_index:
            return collect_era5_target_times(
                self.era5_dir, self.target_variables, self.start_year, self.end_year, self.dt_data
            )

        start = datetime(self.start_year, 1, 1)
        end = datetime(self.end_year, 1, 1)
        t = start
        while t < end:
            if era5_time_has_targets(self.era5_dir, t, self.target_variables):
                times.append(t)
            elif not os.path.exists(self.era5_dir):
                break
            t += timedelta(hours=self.dt_data)
            if self.debug and len(times) >= self.max_debug_samples:
                break

        # If the data are real but not aligned to the configured dt_data grid
        # (for example files named 22:00:00-t-1000.npy), fall back to a scan of
        # the actual nested ERA5 tree instead of reporting an empty dataset.
        if not times and os.path.isdir(self.era5_dir):
            times = collect_era5_target_times(
                self.era5_dir, self.target_variables, self.start_year, self.end_year, self.dt_data
            )
        if not times and os.path.isdir(self.era5_dir) and self.dt_data != 1:
            scanned = collect_era5_target_times(
                self.era5_dir, self.target_variables, self.start_year, self.end_year, dt_data=1
            )
            if scanned:
                warnings.warn(
                    f"No complete ERA5 targets were aligned to dt_data={self.dt_data} hours. "
                    "Falling back to all complete target times found in the nested per-variable ERA5 layout. "
                    "Set datamodule.data.strict_time_index=true or datamodule.data.dt_data=1 to make this explicit.",
                    RuntimeWarning,
                )
                times = scanned
        return times

    def _sensor_has_measurement_file(self, sensor: str, obs_time: datetime) -> bool:
        files = SAT_FILES if sensor in SATELLITE_SENSORS else CONV_FILES
        return self._find_time_file(self.sensor_dirs[sensor], obs_time, files.measurement_suffixes) is not None

    def _sensor_window_file_count(self, sensor: str, target_time: datetime) -> int:
        count = 0
        for hour in range(self.window_start, self.window_end + 1, self.dt_obs):
            obs_time = target_time + timedelta(hours=hour)
            if self._sensor_has_measurement_file(sensor, obs_time):
                count += 1
        return count

    def _time_has_required_observations(self, target_time: datetime) -> bool:
        present_sensors = 0
        for sensor in self.sensors:
            n_files = self._sensor_window_file_count(sensor, target_time)
            if self.require_complete_obs_window:
                expected = len(range(self.window_start, self.window_end + 1, self.dt_obs))
                ok = n_files >= expected
            else:
                ok = n_files > 0
            present_sensors += int(ok)
        return present_sensors >= self.min_required_sensors

    def _find_time_file(self, root: str, t: datetime, suffixes: Iterable[str]) -> Optional[str]:
        for suffix in suffixes:
            path = datetime_path(root, t, suffix)
            if os.path.exists(path):
                return path
        # fallback for slightly different naming conventions
        base = os.path.join(root, f"{t.year:04d}", f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
        if os.path.isdir(base):
            stamp = f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
            for suffix in suffixes:
                matches = sorted(glob(os.path.join(base, f"{stamp}*{suffix}")))
                if matches:
                    return matches[0]
        return None

    def _load_target_variable_file(self, path: str, var: str) -> np.ndarray:
        data = _safe_np_load(path)
        if isinstance(data, Mapping):
            if var in data:
                arr = np.asarray(data[var]).squeeze()
            elif len(data) == 1:
                arr = np.asarray(next(iter(data.values()))).squeeze()
            else:
                raise KeyError(f"ERA5 variable file {path} does not contain {var!r}; keys={sorted(data.keys())}")
        else:
            arr = np.asarray(data).squeeze()
        if arr.ndim == 3:
            cf = _as_channel_first(arr, self.grid_shape)
            if cf.shape[0] != 1:
                raise ValueError(f"ERA5 variable file {path} should contain one channel, got {cf.shape}")
            arr = cf[0]
        if arr.shape != self.grid_shape:
            raise ValueError(f"ERA5 variable file {path} has shape {arr.shape}, expected {self.grid_shape}")
        return arr.astype(np.float32)

    def _load_target(self, t: datetime) -> torch.Tensor:
        """Load and optionally normalize one ERA5 T/Q target, with a small per-worker LRU cache."""
        if self.target_cache_size > 0 and t in self._target_cache:
            cached = self._target_cache.pop(t)
            self._target_cache[t] = cached
            return cached.clone()
        path = find_era5_fullstate_file(self.era5_dir, t)
        if path is not None:
            data = _safe_np_load(path)
            if isinstance(data, Mapping):
                missing = [v for v in self.target_variables if v not in data]
                if missing:
                    raise KeyError(f"ERA5 file {path} does not contain target variables {missing}")
                arr = np.stack([np.asarray(data[v]).squeeze() for v in self.target_variables], axis=0)
            else:
                arr = _as_channel_first(np.asarray(data), self.grid_shape)
                if arr.shape[0] == len(self.target_variables):
                    pass
                elif set(self.target_variables).issubset(self.era5_all_vars) and arr.shape[0] >= len(self.era5_all_vars):
                    idx = [self.era5_all_vars.index(v) for v in self.target_variables]
                    arr = arr[idx]
                else:
                    raise ValueError(
                        f"ERA5 target {path} has {arr.shape[0]} channels. Provide era5_all_vars in the "
                        "Hydra config or pre-extract the 26 T/Q channels."
                    )
        else:
            paths = {var: find_era5_variable_file(self.era5_dir, t, var) for var in self.target_variables}
            missing = [var for var, var_path in paths.items() if var_path is None]
            if missing:
                day = _era5_day_dir(self.era5_dir, t)
                stamp = _era5_stamp(t)
                raise FileNotFoundError(
                    "ERA5 target files were not found for "
                    f"{t.isoformat()} under {day}. Expected either {stamp}.npy or per-variable files "
                    f"like {stamp}-t-1000.npy. Missing target variables: {missing[:8]}"
                    + (" ..." if len(missing) > 8 else "")
                )
            arr = np.stack(
                [self._load_target_variable_file(paths[var], var) for var in self.target_variables], axis=0
            )
        arr = arr.astype(np.float32)
        if self.normalize_target and self.target_mean is not None and self.target_std is not None:
            arr = (arr - self.target_mean[:, None, None]) / self.target_std[:, None, None]
        tensor = _to_float_tensor(arr)
        if self.target_cache_size > 0:
            self._target_cache[t] = tensor
            while len(self._target_cache) > self.target_cache_size:
                self._target_cache.popitem(last=False)
        return tensor.clone()

    def _field_index(self, sensor: str, field_names: Sequence[str]) -> Optional[int]:
        schema = self.sensor_schema.get(sensor, {})
        candidates: List[str] = []
        for block in ("auxiliary_value", "metadata_value", "obs_value"):
            fields = schema.get(block, {}).get("fields_in_order") if isinstance(schema.get(block), Mapping) else None
            if fields:
                candidates = list(fields)
                break
        if not candidates and isinstance(schema.get("fields_in_order"), Sequence):
            candidates = list(schema["fields_in_order"])
        lower = {str(v).lower(): i for i, v in enumerate(candidates)}
        for name in field_names:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    @staticmethod
    @lru_cache(maxsize=8)
    def _lat_lon_grid(grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = grid_shape
        lat = np.linspace(90.0, -90.0, h, dtype=np.float32)
        lon = np.linspace(0.0, 360.0, w, endpoint=False, dtype=np.float32)
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        return lat2d, lon2d

    def _normalize_measurements(self, sensor: str, measurement: np.ndarray, channel: np.ndarray) -> np.ndarray:
        if not self.normalize_obs:
            return measurement.astype(np.float32)
        stats = self.sensor_stats.get(sensor)
        if stats is not None and "mean" in stats and "std" in stats:
            mean = stats["mean"]
            std = np.where(stats["std"] == 0, 1.0, stats["std"])
            ch = np.clip(channel.astype(int), 0, len(mean) - 1)
            return ((measurement - mean[ch]) / std[ch]).astype(np.float32)
        # Configurable safe defaults; these are not used for field discovery.
        key = "satellite" if sensor in SATELLITE_SENSORS else "conventional"
        default = self.obs_default_normalization.get(key, None)
        if default is not None and len(default) == 2:
            mean, std = float(default[0]), max(float(default[1]), 1e-6)
            return ((measurement - mean) / std).astype(np.float32)
        return measurement.astype(np.float32)

    def _load_aux_array(self, sensor: str, t: datetime, files: SensorFiles) -> Optional[np.ndarray]:
        root = self.sensor_dirs[sensor]
        aux_path = self._find_time_file(root, t, files.aux_suffixes)
        if aux_path is None:
            return None
        aux = _safe_np_load(aux_path)
        if isinstance(aux, Mapping):
            # For NPZ/dict auxiliary files, preserve the schema field order instead
            # of sorting keys alphabetically; otherwise indices such as fovn/saza
            # point to the wrong metadata channel.
            schema = self.sensor_schema.get(sensor, {})
            field_order = []
            if isinstance(schema.get("auxiliary_value"), Mapping):
                field_order = list(schema["auxiliary_value"].get("fields_in_order", []))
            elif isinstance(schema.get("fields_in_order"), Sequence):
                field_order = list(schema.get("fields_in_order", []))
            lower = {str(k).lower(): k for k in aux.keys()}
            fields = []
            for name in field_order:
                key = lower.get(str(name).lower())
                if key is not None:
                    val = np.asarray(aux[key])
                    if val.ndim >= 2:
                        fields.append(np.squeeze(val))
            if not fields:
                return None
            aux = np.stack(fields, axis=0)
        try:
            return _as_channel_first(np.asarray(aux), self.grid_shape)
        except Exception:
            return None

    def _extract_aux_field(
        self,
        sensor: str,
        aux: Optional[np.ndarray],
        names: Sequence[str],
        flat_idx: np.ndarray,
        default: float = np.nan,
    ) -> np.ndarray:
        if aux is None:
            return np.full(len(flat_idx), default, dtype=np.float32)
        idx = self._field_index(sensor, names)
        if idx is None or idx >= aux.shape[0]:
            return np.full(len(flat_idx), default, dtype=np.float32)
        return aux[idx].reshape(-1)[flat_idx].astype(np.float32)

    def _load_sensor_at_time(self, sensor: str, obs_time: datetime, target_time: datetime) -> Dict[str, torch.Tensor]:
        root = self.sensor_dirs[sensor]
        files = SAT_FILES if sensor in SATELLITE_SENSORS else CONV_FILES
        measurement_path = self._find_time_file(root, obs_time, files.measurement_suffixes)
        if measurement_path is None or not os.path.exists(measurement_path):
            return _empty_obs()

        raw = _safe_np_load(measurement_path)
        if isinstance(raw, Mapping):
            # Point-cloud npz/dict path.  We only consume fields actually present.
            return self._point_mapping_to_obs(sensor, raw, obs_time, target_time)

        arr = _as_channel_first(np.asarray(raw), self.grid_shape)
        drop_tail = int(SENSOR_DROP_TRAILING_CHANNELS.get(sensor, 0))
        expected_channels = len(SENSOR_CHANNEL_VARS.get(sensor, []))
        if drop_tail > 0 and expected_channels > 0 and arr.shape[0] == expected_channels + drop_tail:
            arr = arr[:-drop_tail]
        c, h, w = arr.shape
        if (h, w) != self.grid_shape:
            raise ValueError(f"{measurement_path} has grid {(h, w)}, expected {self.grid_shape}")

        mask_path = self._find_time_file(root, obs_time, files.mask_suffixes)
        if mask_path and os.path.exists(mask_path):
            mask_arr = np.asarray(_safe_np_load(mask_path)).squeeze()
            if mask_arr.ndim == 2:
                valid_mask = np.broadcast_to(mask_arr[None, :, :] > 0, (c, h, w))
            else:
                valid_mask = _as_channel_first(mask_arr, self.grid_shape) > 0
                if valid_mask.shape[0] == 1:
                    valid_mask = np.broadcast_to(valid_mask, (c, h, w))
                elif valid_mask.shape[0] == c + int(SENSOR_DROP_TRAILING_CHANNELS.get(sensor, 0)):
                    valid_mask = valid_mask[:c]
        else:
            valid_mask = np.isfinite(arr)

        valid = np.isfinite(arr) & valid_mask
        if sensor in SATELLITE_SENSORS:
            rng = self.qc.get("infrared_bt_range" if sensor == "hrs4" else "microwave_bt_range", (0.0, 400.0))
            valid &= (arr >= float(rng[0])) & (arr <= float(rng[1]))
        ch_idx, ij = np.nonzero(valid.reshape(c, -1))
        if len(ch_idx) == 0:
            return _empty_obs()

        if self.max_points_per_sensor > 0 and len(ch_idx) > self.max_points_per_sensor:
            # Deterministic thinning preserves global coverage and avoids loader OOM.
            keep = np.linspace(0, len(ch_idx) - 1, self.max_points_per_sensor, dtype=np.int64)
            ch_idx = ch_idx[keep]
            ij = ij[keep]

        measurement = arr.reshape(c, -1)[ch_idx, ij].astype(np.float32)
        lat_grid, lon_grid = self._lat_lon_grid(self.grid_shape)
        lat = lat_grid.reshape(-1)[ij].astype(np.float32)
        lon = lon_grid.reshape(-1)[ij].astype(np.float32)
        rel_time = np.full(len(ch_idx), (obs_time - target_time).total_seconds() / 3600.0, dtype=np.float32)
        aux = self._load_aux_array(sensor, obs_time, files)

        pressure = self._extract_aux_field(sensor, aux, ("pressure", "pres", "prs", "p"), ij)
        height = self._extract_aux_field(sensor, aux, ("height", "hgt", "elev", "elevation", "hmsl", "hols"), ij)
        if sensor in CONVENTIONAL_SENSORS:
            p_rng = self.qc.get("pressure_range_hpa", (0.5, 1100.0))
            h_rng = self.qc.get("height_range_m", (0.0, 60000.0))
            p_ok = ~np.isfinite(pressure) | ((pressure >= float(p_rng[0])) & (pressure <= float(p_rng[1])))
            h_ok = ~np.isfinite(height) | ((height >= float(h_rng[0])) & (height <= float(h_rng[1])))
            keep = p_ok & h_ok
            measurement, lat, lon, rel_time, ch_idx, ij, pressure, height = (
                measurement[keep], lat[keep], lon[keep], rel_time[keep], ch_idx[keep], ij[keep], pressure[keep], height[keep]
            )
            if len(measurement) == 0:
                return _empty_obs()

        platform = self._extract_aux_field(sensor, aux, ("platform", "platform_id", "sat_id", "satellite_id", "said", "siid"), ij, default=0.0)
        scan = self._extract_aux_field(sensor, aux, ("scan_angle", "scanline", "fov", "fovn"), ij)
        satza = self._extract_aux_field(sensor, aux, ("satellite_zenith_angle", "satellite_za", "saza"), ij)
        solza = self._extract_aux_field(sensor, aux, ("solar_zenith_angle", "solza", "soza"), ij)
        report_type = self._extract_aux_field(sensor, aux, ("report_type", "report", "type"), ij, default=0.0)
        station_type = self._extract_aux_field(sensor, aux, ("station_type", "station", "stype"), ij, default=0.0)
        quality = self._extract_aux_field(sensor, aux, ("quality_flag", "quality", "qc", "lsql", "scan_quality_flags"), ij, default=0.0)
        measurement = self._normalize_measurements(sensor, measurement, ch_idx)

        obs = {
            "measurement": _to_float_tensor(measurement),
            "lat": _to_float_tensor(lat),
            "lon": _to_float_tensor(lon),
            "relative_time": _to_float_tensor(rel_time),
            "channel": torch.from_numpy(ch_idx.astype(np.int64)),
            "platform": torch.from_numpy(np.nan_to_num(platform, nan=0.0).astype(np.int64)),
            "scan_angle": _to_float_tensor(scan),
            "sat_zenith_angle": _to_float_tensor(satza),
            "solar_zenith_angle": _to_float_tensor(solza),
            "pressure": _to_float_tensor(pressure),
            "height": _to_float_tensor(height),
            "variable_type": torch.from_numpy(ch_idx.astype(np.int64)),
            "report_type": torch.from_numpy(np.nan_to_num(report_type, nan=0.0).astype(np.int64)),
            "station_type": torch.from_numpy(np.nan_to_num(station_type, nan=0.0).astype(np.int64)),
            "quality_flag": _to_float_tensor(quality),
            "mask": torch.ones(len(measurement), dtype=torch.float32),
        }
        return obs

    def _point_mapping_to_obs(
        self,
        sensor: str,
        data: Mapping[str, np.ndarray],
        obs_time: datetime,
        target_time: datetime,
    ) -> Dict[str, torch.Tensor]:
        lower = {str(k).lower(): k for k in data}

        def _infer_point_count() -> int:
            for value in data.values():
                arr = np.asarray(value)
                if arr.size > 1:
                    return int(arr.reshape(-1).shape[0])
            return 1

        def get_any(names: Sequence[str], default: float = np.nan) -> np.ndarray:
            for name in names:
                if name.lower() in lower:
                    arr = np.asarray(data[lower[name.lower()]]).reshape(-1)
                    return arr
            return np.full(_infer_point_count(), default, dtype=np.float32)

        measurement = get_any(("observation", "measurement", "obs", "value", "brightness_temperature"))
        n = len(measurement)
        lat = get_any(("lat", "latitude", "obs_latitude"))
        lon = get_any(("lon", "longitude", "obs_longitude"))
        rel_time = get_any(("relative_time", "dt", "time_offset"), default=(obs_time - target_time).total_seconds() / 3600.0)
        channel = get_any(("channel", "channel_index", "sensor_index", "variable_type"), default=0.0).astype(np.int64)
        valid = np.isfinite(measurement) & np.isfinite(lat) & np.isfinite(lon)
        valid &= (lat >= -90.0) & (lat <= 90.0) & (lon >= -360.0) & (lon <= 720.0)
        if sensor in SATELLITE_SENSORS:
            rng = self.qc.get("infrared_bt_range" if sensor == "hrs4" else "microwave_bt_range", (0.0, 400.0))
            valid &= (measurement >= float(rng[0])) & (measurement <= float(rng[1]))
        if sensor in CONVENTIONAL_SENSORS:
            pressure_all = get_any(("pressure", "pres", "prs", "p"))
            height_all = get_any(("height", "hgt", "elev", "elevation", "hmsl", "hols"))
            p_rng = self.qc.get("pressure_range_hpa", (0.5, 1100.0))
            h_rng = self.qc.get("height_range_m", (0.0, 60000.0))
            valid &= ~np.isfinite(pressure_all) | ((pressure_all >= float(p_rng[0])) & (pressure_all <= float(p_rng[1])))
            valid &= ~np.isfinite(height_all) | ((height_all >= float(h_rng[0])) & (height_all <= float(h_rng[1])))
        idx = np.nonzero(valid)[0]
        if self.max_points_per_sensor > 0 and len(idx) > self.max_points_per_sensor:
            idx = idx[np.linspace(0, len(idx) - 1, self.max_points_per_sensor, dtype=np.int64)]
        if len(idx) == 0:
            return _empty_obs()
        measurement = self._normalize_measurements(sensor, measurement[idx].astype(np.float32), channel[idx])
        return {
            "measurement": _to_float_tensor(measurement),
            "lat": _to_float_tensor(lat[idx].astype(np.float32)),
            "lon": _to_float_tensor(lon[idx].astype(np.float32)),
            "relative_time": _to_float_tensor(rel_time[idx].astype(np.float32)),
            "channel": torch.from_numpy(channel[idx].astype(np.int64)),
            "platform": torch.from_numpy(np.nan_to_num(get_any(("platform", "satellite", "sat_id", "said"), 0.0)[idx], nan=0.0).astype(np.int64)),
            "scan_angle": _to_float_tensor(get_any(("scan_angle", "fov", "fovn"))[idx].astype(np.float32)),
            "sat_zenith_angle": _to_float_tensor(get_any(("satellite_zenith_angle", "satellite_za", "saza"))[idx].astype(np.float32)),
            "solar_zenith_angle": _to_float_tensor(get_any(("solar_zenith_angle", "solza", "soza"))[idx].astype(np.float32)),
            "pressure": _to_float_tensor(get_any(("pressure", "pres"))[idx].astype(np.float32)),
            "height": _to_float_tensor(get_any(("height", "elev", "hmsl", "hols"))[idx].astype(np.float32)),
            "variable_type": torch.from_numpy(np.nan_to_num(get_any(("variable_type", "variable", "channel"), 0.0)[idx], nan=0.0).astype(np.int64)),
            "report_type": torch.from_numpy(np.nan_to_num(get_any(("report_type", "type"), 0.0)[idx], nan=0.0).astype(np.int64)),
            "station_type": torch.from_numpy(np.nan_to_num(get_any(("station_type", "station"), 0.0)[idx], nan=0.0).astype(np.int64)),
            "quality_flag": _to_float_tensor(get_any(("quality_flag", "quality", "qc"), 0.0)[idx].astype(np.float32)),
            "mask": torch.ones(len(idx), dtype=torch.float32),
        }

    def _thin_obs_dict(self, obs: Dict[str, torch.Tensor], sensor: str, target_time: datetime) -> Dict[str, torch.Tensor]:
        """Apply the point cap to the *whole observation window* for one sensor.

        Earlier versions capped each hourly/3-hourly file independently and then
        concatenated the full HealDA window, so a configured 250k cap could become
        2.25M points for a 9-file [-21,+3] window.  That creates very large nested
        CPU batches and, under DDP, can exhaust GPU memory before forward() starts.
        This helper enforces the cap after window concatenation.
        """
        n = int(obs.get("measurement", torch.empty(0)).shape[0])
        if self.max_points_per_sensor <= 0 or n <= self.max_points_per_sensor:
            return obs
        keep = torch.linspace(0, n - 1, self.max_points_per_sensor, dtype=torch.long)
        thinned: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == n:
                thinned[key] = value.index_select(0, keep)
            else:
                thinned[key] = value
        return thinned

    def _load_sensor_window(self, sensor: str, target_time: datetime) -> Dict[str, torch.Tensor]:
        parts: List[Dict[str, torch.Tensor]] = []
        for hour in range(self.window_start, self.window_end + 1, self.dt_obs):
            obs_time = target_time + timedelta(hours=hour)
            part = self._load_sensor_at_time(sensor, obs_time, target_time)
            if part["measurement"].numel() > 0:
                parts.append(part)
        if not parts:
            return _empty_obs()
        out: Dict[str, torch.Tensor] = {}
        for key in parts[0]:
            out[key] = torch.cat([p[key] for p in parts], dim=0)
        return self._thin_obs_dict(out, sensor=sensor, target_time=target_time)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        target_time = self.target_times[index]
        target = self._load_target(target_time)
        obs = {sensor: self._load_sensor_window(sensor, target_time) for sensor in self.sensors}
        return {
            "target": target,
            "target_time": target_time.isoformat(),
            "target_time_epoch": int(target_time.replace(tzinfo=timezone.utc).timestamp()),
            "observations": obs,
            "target_variables": self.target_variables,
            "pressure_levels": self.pressure_levels,
        }


def collate_retrieval_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate retrieval samples without padding observation point clouds."""
    target = torch.stack([item["target"] for item in batch], dim=0)
    sensors = list(batch[0]["observations"].keys())
    observations: Dict[str, List[Dict[str, torch.Tensor]]] = {sensor: [] for sensor in sensors}
    for item in batch:
        for sensor in sensors:
            observations[sensor].append(item["observations"][sensor])
    return {
        "target": target,
        "observations": observations,
        "target_time": [item["target_time"] for item in batch],
        "target_time_epoch": torch.tensor([item["target_time_epoch"] for item in batch], dtype=torch.long),
        "target_variables": batch[0]["target_variables"],
        "pressure_levels": batch[0]["pressure_levels"],
    }
