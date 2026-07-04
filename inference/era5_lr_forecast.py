import sys
sys.path.append(".")
import os
from datetime import datetime

import click

from inference.utils.data_utils import VARIABLES, get_climatology, get_era5
from inference.era5_forecast_core import eval_forecast
from src.utils.device import get_device


def _make_lr_loader(era5_lr_dir: str):
    """Loader for the 1.0deg grid: truth + climatology at LR resolution."""
    def _loader(eval_time: datetime):
        file_path = os.path.join(
            era5_lr_dir,
            f"{eval_time.year:04d}",
            f"{eval_time.year:04d}-{eval_time.month:02d}-{eval_time.day:02d}",
            f"{eval_time.hour:02d}:{eval_time.minute:02d}:{eval_time.second:02d}.npy",
        )
        era5 = get_era5(file_path, (-1, 181, 360))
        clim = get_climatology(
            f"{era5_lr_dir}/climatology_np181x360_2010_2021",
            (-1, 181, 360), eval_time, VARIABLES,
        )
        return era5, clim
    return _loader


@click.command()
@click.option("--ckpt_dir", type=str, default="./logs")
@click.option("--era5_lr_dir", type=click.Path(exists=True), default=os.environ.get("ERA5_LR_DIR", "/public02/data/era5_np181x360_level13"))
@click.option("--era5_hr_dir", type=click.Path(exists=True), default=os.environ.get("ERA5_HR_DIR", "/public02/data/era5_np721x1440_level13_merged"))
@click.option("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "outputs/pretrain_xichen_forecast"))
@click.option("--forecast_hours", type=int, default=240)
@click.option("--start_year", type=int, default=2023)
@click.option("--end_year", type=int, default=2024)
@click.option("--decorrelation_hours", type=int, default=6)
@click.option("--dt", type=int, default=6)
@click.option("--forecast_name", type=str, default="pretrain_xichen_forecast")
@click.option("--device", type=str, default="cuda")
def main(
    ckpt_dir,
    era5_lr_dir,
    era5_hr_dir,
    output_dir,
    forecast_hours,
    start_year,
    end_year,
    decorrelation_hours,
    dt,
    forecast_name,
    device,
):
    """AR rollout against ERA5 truth on the 1.0deg (LR) grid."""
    config = {
        "era5_lr_dir": era5_lr_dir,
        "era5_hr_dir": era5_hr_dir,
        "forecast_hours": forecast_hours,
        "start_year": start_year,
        "end_year": end_year,
        "decorrelation_hours": decorrelation_hours,
        "dt": dt,
        "forecast_name": forecast_name,
        "device": get_device(device, 0),
        "forecast_pair": "lr",
        "resolution_tag": "1p0deg",
    }
    eval_forecast(
        _make_lr_loader(era5_lr_dir),
        ckpt_dir,
        config,
        output_dir,
    )


if __name__ == "__main__":
    main()
