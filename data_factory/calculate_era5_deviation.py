import glob
import os
import sys
sys.path.append(".")
import click
import numpy as np
import datetime
from datetime import datetime, timedelta
from utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')

# 检查是否为闰年
def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def get_hours_in_month(year, month):    
    # 月份的天数，2月为闰年和平年的区分
    days_in_month = [31, 29 if is_leap_year(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 确保月份在1到12之间
    if 1 <= month <= 12:
        # 计算总小时数
        hours_this_month = days_in_month[month - 1] * 24
        if month > 1:
            total_before_hours = sum(days_in_month[:month-1]) * 24
        else:
            total_before_hours = 0
        return hours_this_month, total_before_hours
    else:
        raise ValueError("月份必须在1到12之间")

def every_hour(year: int, dt: int):
    start = datetime(year, 1, 1)
    total_hours = 8784 if is_leap_year(year) else 8760

    return [start + timedelta(hours=i) for i in range(0, total_hours, dt)]

@click.command()
@click.option("--era5_dir", type=click.Path(exists=True), default=os.environ.get("ERA5_LR_DIR", "/public02/data/era5_np181x360_level13"))
@click.option("--save_dir", type=click.Path(exists=True), default=os.environ.get("ERA5_LR_DIR", "/public02/data/era5_np181x360_level13"))
@click.option(
    "--surface_variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ],
)
@click.option(
    "--pressure_variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
    ],
)
@click.option("--dt", type=int, default=1)
def main(
    era5_dir,
    save_dir,
    surface_variables,
    pressure_variables,
    dt,
):
    os.makedirs(save_dir, exist_ok=True)
    era5_sigma = {}
    times = every_hour(2022, 6)

    for var in surface_variables:
        code = NAME_TO_VAR[var]
        sigma = []
        for time in times:
            data1 = np.load(
                os.path.join(
                    era5_dir,
                    f"{time.year:04d}",
                    f"{time.year:04d}-{time.month:02d}-{time.day:02d}",
                    f"{time.hour:02d}:{time.minute:02d}:{time.second:02d}-{code}.npy"
                )
            )
            time_ = time + timedelta(hours=24)
            # logging.info(time, time_)
            data2 = np.load(
                os.path.join(
                    era5_dir,
                    f"{time_.year:04d}",
                    f"{time_.year:04d}-{time_.month:02d}-{time_.day:02d}",
                    f"{time_.hour:02d}:{time_.minute:02d}:{time_.second:02d}-{code}.npy"
                )
            )
            sigma.append(np.abs(data2 - data1))
            # logging.info(f"{code} error: {np.sqrt(np.nanmean((data2 - data1)**2))}")

        era5_sigma[code] = np.nanmean(sigma, axis=0)

    for var in pressure_variables:
        code = NAME_TO_VAR[var]
        sigma = []
        for level in DEFAULT_PRESSURE_LEVELS:
            for time in times:
                data1 = np.load(
                    os.path.join(
                        era5_dir,
                        f"{time.year:04d}",
                        f"{time.year:04d}-{time.month:02d}-{time.day:02d}",
                        f"{time.hour:02d}:{time.minute:02d}:{time.second:02d}-{code}-{level}.npy"
                    )
                )
                time_ = time + timedelta(hours=48)
                data2 = np.load(
                    os.path.join(
                        era5_dir,
                        f"{time_.year:04d}",
                        f"{time_.year:04d}-{time_.month:02d}-{time_.day:02d}",
                        f"{time_.hour:02d}:{time_.minute:02d}:{time_.second:02d}-{code}-{level}.npy"
                    )
                )
                sigma.append(np.abs(data2 - data1))
                # logging.info(f"{code}-{level} error: {np.sqrt(np.nanmean((data2 - data1)**2))}")

            era5_sigma[f"{code}-{level}"] = np.nanmean(sigma, axis=0)

    logging.info(era5_sigma)
    np.savez(os.path.join(save_dir, "era5_48h_deviation.npz"), **era5_sigma)

if __name__ == "__main__":
    main()
