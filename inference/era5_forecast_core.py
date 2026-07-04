"""Shared AR-forecast evaluation core.

Extracted from the near-duplicate pair `era5_lr_forecast.py` and
`era5_interp_forecast.py`. The two scripts differed only in how they
loaded the evaluation-time ERA5 truth + climatology (and how the
forecast slice was denormalised back to physical units). That step is
factored out behind a small `loader` protocol; everything else — model
construction, checkpoint loading, normalisation, the multi-resolution
AR rollout, the metric computation, plotting, and `*.npz` writing —
lives here.

Public API
----------
eval_forecast(loader, ckpt_path, config, output_dir) -> dict

* loader : Callable[[datetime], tuple(era5, clim)]
    Called once per evaluation lead-time. Receives the `eval_time`
    datetime for the current lead, and returns a 2-tuple:
        era5 : np.ndarray   shape (1, V, H, W), physical units
        clim : np.ndarray   same shape, physical units
* ckpt_path : str
    Directory passed to `load_forecast_ckpt`.
* config : dict
    Required keys: era5_lr_dir, era5_hr_dir, forecast_hours,
    start_year, end_year, decorrelation_hours, dt, forecast_name,
    device.
* output_dir : str
    Where the `*_metrics.npz` is written.

The wrapper scripts pass `resolution_tag` via `config["resolution_tag"]`
so figure / `.npz` filenames match the prior per-script conventions.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Callable

import numpy as np
import torch
from dateutil.relativedelta import relativedelta

# Make `inference` importable when invoked as a script from repo root.
sys.path.append(".")


from src.models.forecast.arch import XiChenForecast
from src.metrics.weighted_acc_rmse import weighted_rmse, weighted_acc, weighted_activity
from inference.utils.data_utils import VARIABLES, get_era5, get_normalize
from inference.utils.model_utils import load_forecast_ckpt
from plots.plot_forecast_metrics import (
    plot_forecast_metrics,
    save_forecast_plots,
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("inference.era5_forecast_core")


_CONFIG_KEYS = (
    "era5_lr_dir",
    "era5_hr_dir",
    "forecast_hours",
    "start_year",
    "end_year",
    "decorrelation_hours",
    "dt",
    "forecast_name",
    "device",
)


def _build_model() -> XiChenForecast:
    """Construct the XiChenForecast with the project's canonical arch."""
    return XiChenForecast(
        default_vars=VARIABLES,
        img_size=[181, 360],
        window_size=[6, 12],
        patch_size=[6, 5],
        patch_stride=[5, 5],
        embed_dim=768,
        num_heads=12,
        encoder_depths=[2, 2, 2],
        latent_depths=[4, 4, 4],
        decoder_depths=[2, 2, 2],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0,
    )


def _autoregressive_rollout(
    forecast_model: XiChenForecast,
    era5_init: torch.Tensor,
    forecast_hours: int,
    dt: int,
    device,
):
    """Run the multi-resolution AR rollout; return (forecast, log_var)."""
    horizon = forecast_hours // dt + 1
    seq_forecast = np.zeros((horizon, len(VARIABLES), 181, 360))
    seq_log_var = np.zeros((horizon, len(VARIABLES), 181, 360))

    with torch.no_grad():
        for i in range(horizon):
            if i == 0:
                seq_forecast[0] = era5_init.numpy()
                seq_log_var[0] = 0 * era5_init.numpy()
                continue
            # Cascade through discrete sub-model horizons {24,12,6,3,1}h.
            for step in (24, 12, 6, 3, 1):
                if (step // dt) > 0 and (i % (step // dt)) == 0:
                    pred, log_var = forecast_model(
                        torch.from_numpy(
                            seq_forecast[i - step // dt : i - step // dt + 1]
                        ).to(device, dtype=torch.float32),
                        torch.from_numpy(
                            step * np.ones((1, 1))
                        ).to(device, dtype=torch.float32) / 100,
                        VARIABLES,
                        use_checkpoint=True,
                    )
                    seq_forecast[i : i + 1] = pred.detach().cpu().numpy()
                    seq_log_var[i : i + 1] = log_var.detach().cpu().numpy()
                    break
    return seq_forecast, seq_log_var


def _denorm(forecast_slice: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse normalise a (1, V, H, W) forecast slice with (V,) mean/std."""
    return forecast_slice * std + mean


def _eval_one_init(
    loader: Callable[[datetime], tuple],
    denorm: Callable[[np.ndarray], np.ndarray],
    seq_forecast: np.ndarray,
    seq_log_var: np.ndarray,
    current_time: datetime,
    forecast_hours: int,
    dt: int,
    pred_rmse_scale: np.ndarray,
):
    """Compute per-lead-time RMSE / ACC / activity / pred-RMSE for one init.

    `loader(eval_time)` returns `(era5, clim)` truth + climatology at the
    resolution used for the metric comparison. `denorm(forecast_slice)`
    takes the raw normalised forecast slice (1, V, H, W) and returns the
    physical-units slice at the metric-comparison resolution — e.g.
    identity for the LR (181x360) comparison, or
    `geographic_interpolate(lr2hr)` followed by HR denormalisation for
    the 0.25deg comparison. `pred_rmse_scale` is the high-res std used
    to scale the predicted standard deviation.
    """
    horizon = forecast_hours // dt + 1
    rmse_xf, acc_xf, activity_xf, pred_rmse_xf = [], [], [], []

    for i in range(horizon):
        eval_time = current_time + relativedelta(hours=i * dt)
        era5, clim = loader(eval_time)
        forecast_eval = denorm(seq_forecast[i : i + 1])

        rmse_xf.append(weighted_rmse(era5, forecast_eval))
        acc_xf.append(weighted_acc(era5 - clim, forecast_eval - clim))
        activity_xf.append(weighted_activity(forecast_eval, clim))
        pred_rmse = np.sqrt(np.exp(seq_log_var[i : i + 1])) * pred_rmse_scale
        pred_rmse_xf.append(np.sqrt(np.mean(pred_rmse ** 2, axis=(0, -2, -1))))

    return (
        np.stack(rmse_xf).squeeze(),
        np.stack(acc_xf).squeeze(),
        np.stack(activity_xf).squeeze(),
        np.stack(pred_rmse_xf).squeeze(),
    )


def save_metrics_and_plots(
    seq_rmse_xf,
    seq_acc_xf,
    seq_activity_xf,
    seq_pred_rmse_xf,
    forecast_name: str,
    output_dir: str,
    resolution_tag: str,
) -> dict:
    """Stack, plot, save .npz; return the metrics dict (trainer-validate compatible).

    `resolution_tag` distinguishes the two CLI variants, e.g. "1p0deg"
    or "interp_0p25deg"; it suffixes figure and `.npz` filenames.
    """
    seq_rmse_xf = np.stack(seq_rmse_xf, axis=0)
    seq_rmse_xf[:, :, -13:] *= 1000
    seq_acc_xf = np.stack(seq_acc_xf, axis=0)
    seq_activity_xf = np.stack(seq_activity_xf, axis=0)
    seq_activity_xf[:, :, -13:] *= 1000
    seq_pred_rmse_xf = np.stack(seq_pred_rmse_xf, axis=0)
    seq_pred_rmse_xf[:, :, -13:] *= 1000

    figures = plot_forecast_metrics(
        np.mean(seq_rmse_xf, axis=0),
        np.mean(seq_acc_xf, axis=0),
        np.mean(seq_activity_xf, axis=0),
        VARIABLES,
    )
    figures_dir = f"figures/{forecast_name}_{resolution_tag}"
    os.makedirs("figures", exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    save_forecast_plots(figures, output_dir=figures_dir)

    npz_path = f"{output_dir}/{forecast_name}_{resolution_tag}_metrics.npz"
    np.savez(
        npz_path,
        variables=VARIABLES,
        rmse=np.mean(seq_rmse_xf, axis=0),
        acc=np.mean(seq_acc_xf, axis=0),
        activity=np.mean(seq_activity_xf, axis=0),
    )

    return {
        "variables": np.array(VARIABLES),
        "rmse": np.mean(seq_rmse_xf, axis=0),
        "acc": np.mean(seq_acc_xf, axis=0),
        "activity": np.mean(seq_activity_xf, axis=0),
        "pred_rmse": np.mean(seq_pred_rmse_xf, axis=0),
        "metrics_npz": npz_path,
        "figures_dir": figures_dir,
    }


def eval_forecast(
    loader: Callable[[datetime], tuple],
    ckpt_path: str,
    config: dict,
    output_dir: str,
) -> dict:
    """Run an AR forecast evaluation against ERA5 truth and write metrics/plots.

    Parameters
    ----------
    loader : Callable[[datetime], (era5, clim)]
        Per-lead-time data loader. Receives the evaluation `datetime` for
        the current lead, returns `(era5, clim)` arrays of shape
        `(1, V, H, W)` in physical units.
    ckpt_path : str
        Directory containing the forecast checkpoint passed to
        `load_forecast_ckpt`.
    config : dict
        Must contain: era5_lr_dir, era5_hr_dir, forecast_hours,
        start_year, end_year, decorrelation_hours, dt, forecast_name,
        device. Optional: resolution_tag (default "1p0deg"), and either
        `forecast_mean_key`/`forecast_std_key` ("lr" or "hr") to pick
        which normalisation pair denormalises the metric comparison
        (default "lr").
    output_dir : str
        Directory in which the `*_metrics.npz` is written.

    Returns
    -------
    dict
        Metrics dict with keys: variables, rmse, acc, activity,
        pred_rmse, metrics_npz, figures_dir.
    """
    missing = [k for k in _CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f"eval_forecast config missing keys: {missing}")

    os.makedirs(output_dir, exist_ok=True)

    forecast_model = _build_model()
    forecast_model = load_forecast_ckpt(
        ckpt_path, config["forecast_name"], forecast_model
    )
    forecast_model.to(config["device"], dtype=torch.float32)
    forecast_model.eval()

    era5_lr_mean, era5_lr_std = get_normalize(
        f"{config['era5_lr_dir']}/normalized_mean_std", VARIABLES
    )
    era5_hr_mean, era5_hr_std = get_normalize(
        f"{config['era5_hr_dir']}/normalized_mean_std", VARIABLES
    )

    forecast_pair = config.get("forecast_pair", "lr")
    if forecast_pair == "lr":
        forecast_mean, forecast_std = era5_lr_mean, era5_lr_std
    elif forecast_pair == "hr":
        forecast_mean, forecast_std = era5_hr_mean, era5_hr_std
    else:
        raise ValueError(f"forecast_pair must be 'lr' or 'hr', got {forecast_pair!r}")

    # Default denorm: simple inverse normalisation on the model's LR grid.
    # Wrappers may pass a richer callable (e.g. geographic interpolation +
    # HR denormalisation) via config["denorm_fn"] to match their grid.
    base_denorm = lambda s: _denorm(s, forecast_mean, forecast_std)
    denorm = config.get("denorm_fn", base_denorm)

    start_time = datetime(config["start_year"], 1, 1, 0, 0)
    end_time = datetime(config["start_year"], 12, 21, 0, 0)
    current_time = start_time

    seq_rmse_xf, seq_acc_xf, seq_activity_xf, seq_pred_rmse_xf = [], [], [], []

    while current_time < end_time:
        init_file = os.path.join(
            config["era5_lr_dir"],
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )
        if os.path.exists(init_file):
            era5 = get_era5(init_file, (-1, 181, 360))
        era5_init = torch.from_numpy((era5 - era5_lr_mean) / era5_lr_std)

        seq_forecast, seq_log_var = _autoregressive_rollout(
            forecast_model,
            era5_init,
            config["forecast_hours"],
            config["dt"],
            config["device"],
        )

        rmse_xf, acc_xf, activity_xf, pred_rmse_xf = _eval_one_init(
            loader,
            denorm,
            seq_forecast,
            seq_log_var,
            current_time,
            config["forecast_hours"],
            config["dt"],
            era5_hr_std,
        )

        seq_rmse_xf.append(rmse_xf)
        seq_acc_xf.append(acc_xf)
        seq_activity_xf.append(activity_xf)
        seq_pred_rmse_xf.append(pred_rmse_xf)

        log.info("Forecast Z500 RMSE Xf (ERA5): %s at %s", rmse_xf[:, 11], current_time)
        log.info("Forecast Z500 ACC Xf (ERA5): %s at %s", acc_xf[:, 11], current_time)
        log.info("Forecast Z500 Activity Xf (ERA5): %s at %s", activity_xf[:, 11], current_time)
        log.info("Forecast Z500 Pred RMSE Xf (ERA5): %s at %s", pred_rmse_xf[:, 11], current_time)

        current_time = current_time + relativedelta(hours=config["decorrelation_hours"])

    return save_metrics_and_plots(
        seq_rmse_xf,
        seq_acc_xf,
        seq_activity_xf,
        seq_pred_rmse_xf,
        config["forecast_name"],
        output_dir,
        config.get("resolution_tag", "1p0deg"),
    )
