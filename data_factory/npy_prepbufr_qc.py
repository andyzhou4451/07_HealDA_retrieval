import glob
import os
import sys
sys.path.append(".")
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import interp2d
import datetime
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')

ERA5_DEVIATION = np.load(os.environ.get("ERA5_LR_DIR", "/public02/data/era5_np181x360_level13") + "/era5_24h_deviation.npz")

from data_factory._sigma_utils import compute_sigma

@click.command()
@click.option("--era5_dir", type=click.Path(exists=True), default=os.environ.get("ERA5_LR_DIR", "/public02/data/era5_np181x360_level13"))
@click.option("--obs_dir", type=click.Path(exists=True), default=os.environ.get("OBS_DIR", "/public02/data/Observation/observation_npy/GDAS_prebufr_corrected_npy_1.0deg"))
@click.option("--save_dir", type=str, default=os.environ.get("OBS_DIR", "/public02/data/Observation/observation_npy/GDAS_prebufr_corrected_npy_1.0deg"))
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
    "t2m", "u10", "v10", "msl",
    "z-50", "z-100", "z-150", "z-200", "z-250", "z-300", "z-400", "z-500", "z-600", "z-700", "z-850", "z-925", "z-1000",
    "u-50", "u-100", "u-150", "u-200", "u-250", "u-300", "u-400", "u-500", "u-600", "u-700", "u-850", "u-925", "u-1000",
    "v-50", "v-100", "v-150", "v-200", "v-250", "v-300", "v-400", "v-500", "v-600", "v-700", "v-850", "v-925", "v-1000",
    "t-50", "t-100", "t-150", "t-200", "t-250", "t-300", "t-400", "t-500", "t-600", "t-700", "t-850", "t-925", "t-1000",
    "q-50", "q-100", "q-150", "q-200", "q-250", "q-300", "q-400", "q-500", "q-600", "q-700", "q-850", "q-925", "q-1000",
    ],
)
@click.option("--resolution", type=float, default=1.0)
@click.option("--year", type=int, default=2022)
@click.option("--dt", type=int, default=3)
def main(
    era5_dir,
    obs_dir,
    save_dir,
    variables,
    resolution,
    year,
    dt,
):
    os.makedirs(save_dir, exist_ok=True)
    
    np_deviation = []
    for var in variables:
        np_deviation.append(ERA5_DEVIATION[var])
    np_devaition = np.stack(np_deviation)
    obs_sigma = {}
    obs_sigma = compute_sigma(era5_dir, obs_dir, variables, year, save_dir, resolution, dt, np_devaition, obs_sigma)
 
    for var in obs_sigma.keys():
        # if var not in constant_fields:
        obs_sigma[var] = np.stack(obs_sigma[var], axis=0)
 
    for var in obs_sigma.keys():  # aggregate over the years
        # if var not in constant_fields:
        sigma = obs_sigma[var]
        # var(X) = E[var(X|Y)] + var(E[X|Y])
        variance = np.nanmean(sigma**2, axis=0)
        sigma = np.sqrt(variance)
        obs_sigma[var] = sigma
 
    logging.info(obs_sigma)
    np.savez(os.path.join(save_dir, f"obs_sigma_qc_{year:04d}.npz"), **obs_sigma)

    variable_id = {}
    for i, k in enumerate(obs_sigma.keys()):
        variable_id[k] = int(i)
    np.savez(os.path.join(save_dir, "variable_id.npz"), **variable_id)

if __name__ == "__main__":
    main()
