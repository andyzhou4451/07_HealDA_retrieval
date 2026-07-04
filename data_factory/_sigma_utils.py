"""Shared helpers for per-observation sigma quality-control scripts.

This module extracts the duplicated `compute_sigma` helper from
`data_factory/npy_prepbufr_qc.py` and `data_factory/npy_satwnd_qc.py`.
Both call sites share the same signature and behavior; the only practical
difference between the original copies was that the satwnd version guarded
against missing obs files, while the prepbufr version assumed files
existed. The `skip_missing` flag preserves the original behavior of each
caller.
"""

import logging
import os
from datetime import datetime

import numpy as np
from dateutil.relativedelta import relativedelta


def compute_sigma(
    era5_path,
    obs_path,
    variables,
    year,
    save_dir,
    resolution,
    dt,
    np_devaition,
    obs_sigma,
    skip_missing=False,
):
    """Iterate over a year of 3-hourly (configurable) timesteps and accumulate
    per-variable RMSE between obs and ERA5 background.

    Parameters
    ----------
    era5_path : str
        Root directory containing per-timestep ERA5 background ``.npy`` files.
    obs_path : str
        Root directory containing per-timestep observation ``-obs_value.npy`` files.
    variables : list[str]
        Variable names to aggregate RMSE for.
    year : int
        Year to process (Jan 1 00:00 through Jan 1 of the following year).
    save_dir : str
        Output directory; created if missing.
    resolution : float
        Grid resolution (degrees). Retained for signature compatibility with
        the original scripts; not used in the computation.
    dt : int
        Timestep in hours between successive samples. Retained for signature
        compatibility; not used in the computation.
    np_devaition : np.ndarray
        Per-variable ERA5 deviation used for the O-B outlier filter.
    obs_sigma : dict[str, list[float]]
        Mutable accumulator; each variable's RMSE samples are appended.
    skip_missing : bool, default False
        When True, silently skip timesteps whose obs file does not exist
        (preserves the original satwnd behavior). When False, raise
        ``FileNotFoundError`` via ``np.load`` (preserves the original
        prepbufr behavior).

    Returns
    -------
    dict[str, list[float]]
        The same ``obs_sigma`` dict passed in, with per-variable RMSE lists.
    """
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    os.makedirs(os.path.join(save_dir, f"{year:04d}"), exist_ok=True)

    start_time = datetime(year, 1, 1, 0, 0)
    end_time = datetime(year + 1, 1, 1, 0, 0)

    current_time = start_time

    while current_time < end_time:
        logging.info(f"Start calculate observation error at {current_time}")
        # non-constant fields
        obs_file_path = os.path.join(
            obs_path,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}-obs_value.npy",
        )
        if skip_missing and not os.path.exists(obs_file_path):
            current_time = current_time + relativedelta(hours=3)
            continue
        np_obs = np.load(obs_file_path)
        era5_file_path = os.path.join(
            era5_path,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )
        np_era5 = np.load(era5_file_path)[17:43]

        omb = np.abs(np_obs - np_era5)
        obs_mask = ~np.isnan(np_obs) * 1
        obs_mask = np.where(omb > 0.5 * np_devaition, 0, obs_mask)

        for i, k in enumerate(variables):
            mask_ = obs_mask[0, i]
            obs_ = np_obs[0, i]
            era5_ = np_era5[i]
            # 添加分母安全检查
            sum_mask = np.nansum(mask_)
            if sum_mask > 0:
                rmse = np.sqrt(np.nansum(mask_ * (obs_ - era5_) ** 2) / sum_mask)
            else:
                rmse = np.nan  # 无有效数据时设为nan
            if k not in obs_sigma:
                obs_sigma[k] = [rmse]
            else:
                obs_sigma[k].append(rmse)

        current_time = current_time + relativedelta(hours=3)

    return obs_sigma
