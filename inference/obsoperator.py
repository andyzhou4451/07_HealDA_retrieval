import sys
sys.path.append(".")
import os
from pathlib import Path
import pickle
import numpy as np
import torch
import glob
import json
import re
import click
import dask
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.models.obsoperator.arch import XiChenObsOp
from src.utils.device import get_device
from inference.utils.data_utils import *
from inference.utils.model_utils import (
    load_obsop_ckpt,
)
from plots.plot_obsop_metrics import plot_obsop_omb
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')

def eval_obsoperator(
    era5_dir,
    obs_name,
    obs_dir,
    save_dir,
    start_year,
    end_year,
    model_name,
    debug,
    device,
):

    with open(f"inference/configs/{obs_name}_obsop.json") as f:
        model_params = json.load(f)

    obsop_model = XiChenObsOp(**model_params)
    obsop_model = load_obsop_ckpt(
        "logs", 
        model_name, 
        obsop_model
    )
    obsop_model.to(device, dtype=torch.float32)

    obs_dict = prepare_sat[obs_name](
        obs_dir,
        obsop_model.out_sat_vars,
        sat_tmbrs_vars[obs_name]
    )

    era5_mean, era5_std = get_normalize(f"{era5_dir}/normalized_mean_std", VARIABLES)
    tmbrs_mean, tmbrs_std = get_normalize(f"{obs_dir}/1b{obs_name}_merged_npy_1.0deg", sat_tmbrs_vars[obs_name])

    if debug:
        start_time = datetime(start_year, 1, 1, 0, 0)
        end_time = datetime(start_year, 1, 3, 0, 0)
    else:
        start_time = datetime(start_year, 1, 1, 0, 0)
        end_time = datetime(end_year, 1, 1, 0, 0)

    # Initialize dictionaries to store all data for each channel
    tgt_tmbrs_data = {var: [] for var in obsop_model.out_sat_vars}
    out_tmbrs_data = {var: [] for var in obsop_model.out_sat_vars}
    val_obserr_data = {var: [] for var in obsop_model.out_sat_vars}
    total_mse, total_var = 0, 0

    current_time = start_time
    num_samples = 0

    while current_time < end_time:
        era5_path = os.path.join(
            era5_dir,
            f"{current_time.year:04d}",
            f"{current_time.year:04d}-{current_time.month:02d}-{current_time.day:02d}",
            f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.npy",
        )
        if not os.path.exists(era5_path):
            logging.warning(f"Skip missing ERA5 file: {era5_path}")
            current_time = current_time + relativedelta(hours=3)
            continue
        era5 = get_era5(era5_path, (-1, 181, 360))

        era5 = torch.from_numpy((era5 - era5_mean) / era5_std)
        
        np_tmbrs_data, np_auxiliary_data, np_mask = get_sat[obs_name](
            obs_dir=obs_dir, 
            obs_time=current_time, 
            auxiliary_vars=sat_auxiliary_vars[obs_name],
            tmbrs_vars=sat_tmbrs_vars[obs_name],
            obs_dict=obs_dict,
            num_lat=181,
            num_lon=360,
        )            

        tmbrs_tensor = torch.as_tensor((np_tmbrs_data - obs_dict["tmbrs_mean"]) / obs_dict["tmbrs_std"])
        auxiliary_tensor = torch.as_tensor(np_auxiliary_data).unsqueeze(0)   # (1, 4, 68, 64, 128)
        mask_tensor = torch.as_tensor(np_mask).unsqueeze(-3).unsqueeze(0)
        sat_tensor = torch.concat([auxiliary_tensor, tmbrs_tensor], dim=1) * mask_tensor

        with torch.no_grad():
            out_tmbrs, log_var, tgt_tmbrs = obsop_model(
                era5.to(device, dtype=torch.float32), 
                sat_tensor.to(device, dtype=torch.float32), 
                mask_tensor.to(device, dtype=torch.float32), 
                use_checkpoint=False
            )

        out_tmbrs = mask_tensor.detach().cpu().numpy() * out_tmbrs.detach().cpu().numpy()
        log_var = log_var.detach().cpu().numpy()
        tgt_tmbrs = mask_tensor.detach().cpu().numpy() * tgt_tmbrs.detach().cpu().numpy()
        sat_mask = mask_tensor.detach().cpu().numpy()
        var = np.exp(log_var) * sat_mask
        tgt_sat_var_ids = np.array([sat_tmbrs_vars[obs_name].index(item) for item in obsop_model.out_sat_vars])
        
        out_tmbrs = (obs_dict["tmbrs_std"][:, tgt_sat_var_ids] * out_tmbrs + obs_dict["tmbrs_mean"][:, tgt_sat_var_ids]) * sat_mask
        tgt_tmbrs = (obs_dict["tmbrs_std"][:, tgt_sat_var_ids] * tgt_tmbrs + obs_dict["tmbrs_mean"][:, tgt_sat_var_ids]) * sat_mask
        var = sat_mask * (obs_dict["tmbrs_std"][:, tgt_sat_var_ids] * np.sqrt(var)) ** 2

        val_rmse = sat_mask * np.sqrt((out_tmbrs - tgt_tmbrs) ** 2)
        val_obserr = sat_mask * np.sqrt(var)

        # Process each channel (level) separately
        for channel in range(len(obsop_model.out_sat_vars)):
            # Extract data for current channel
            tgt_tmbrs_channel = tgt_tmbrs[0, channel, :, :]
            out_tmbrs_channel = out_tmbrs[0, channel, :, :]
            val_obserr_channel = val_obserr[0, channel, :, :]
            mask_channel = np_mask

            # Get channel name
            channel_name = obsop_model.out_sat_vars[channel]

            # Get indices where mask is 1
            valid_indices = np.where(mask_channel == 1)

            # Extract masked data
            tgt_tmbrs_masked = tgt_tmbrs_channel[valid_indices]
            out_tmbrs_masked = out_tmbrs_channel[valid_indices]
            val_obserr_masked = val_obserr_channel[valid_indices]

            # Store data for this channel
            tgt_tmbrs_data[channel_name].extend(tgt_tmbrs_masked)
            out_tmbrs_data[channel_name].extend(out_tmbrs_masked)
            val_obserr_data[channel_name].extend(val_obserr_masked)

        total_mse += np.sum(sat_mask * ((out_tmbrs - tgt_tmbrs) ** 2), axis=(0, -2, -1)) / (sat_mask.sum(axis=(0, -2, -1)) + 1e-6)
        total_var += np.sum(sat_mask * var, axis=(0, -2, -1)) / (sat_mask.sum(axis=(0, -2, -1)) + 1e-6)
        num_samples += 1

        current_time = current_time + relativedelta(hours=3)

    if num_samples == 0:
        raise FileNotFoundError(f"No valid ERA5/observation pairs were found in {era5_dir} for {start_year}-{end_year}")

    total_rmse = (total_mse / num_samples) ** 0.5
    total_obserr = (total_var / num_samples) ** 0.5

    obs_sigma = {var: [] for var in obsop_model.out_sat_vars}
    for j in range(val_rmse.shape[1]):
        logging.info(f"{obs_name} ObsOp RMSE of {obsop_model.out_sat_vars[j]} is: {total_rmse[j]}")
        logging.info(f"{obs_name} ObsOp predict error of {obsop_model.out_sat_vars[j]} is: {total_obserr[j]}")
        obs_sigma[obsop_model.out_sat_vars[j]].append(total_rmse[j])

    np.savez(
        f"{save_dir}/{obs_name}/avg_obs_error.npz",
        **obs_sigma,
    )

    # After processing all time steps, create plots for each variable
    for channel_name in obsop_model.out_sat_vars:
        tgt_tmbrs_values = np.array(tgt_tmbrs_data[channel_name])
        out_tmbrs_values = np.array(out_tmbrs_data[channel_name])

        if len(tgt_tmbrs_values) > 0:
            logging.info(f"Creating plot for {channel_name} with {len(tgt_tmbrs_values)} total data points")
            # Create a mask of all ones since we've already filtered the data
            mask_ = np.ones_like(tgt_tmbrs_values)
            plot_obsop_omb(
                tgt_tmbrs_values=tgt_tmbrs_values,
                out_tmbrs_values=out_tmbrs_values,
                mask=mask_,
                variable_name=channel_name,
                plot_dir=f"{save_dir}/{obs_name}",
            )
        else:
            logging.info(f"Warning: No valid data for {channel_name} across all time steps")

@click.command()
@click.option("--era5_dir", type=click.Path(exists=True), default=os.environ.get("ERA5_LR_DIR", "/public02/data/era5_np181x360_level13/"))
@click.option("--obs_name", type=str, default="atms")
@click.option("--obs_dir", type=click.Path(exists=True), default=os.environ.get("OBS_DIR", "/public02/data/Observation/observation_npy"))
@click.option("--save_dir", type=str, default="/public/home/wangwuxing01/research/XiChen/data/xichen_results/obsop")
@click.option("--start_year", type=int, default=2023)
@click.option("--end_year", type=int, default=2024)
@click.option("--model_name", type=str, default="train_atms_obsop_20260608")
@click.option("--debug", type=bool, default=False)
@click.option("--device", type=str, default="cuda")
def main(
    era5_dir,
    obs_name,
    obs_dir,
    save_dir,
    start_year,
    end_year,
    model_name,
    debug,
    device,
):
    device = get_device(device, 0)
    os.makedirs(f"{save_dir}", exist_ok=True)
    os.makedirs(f"{save_dir}/{obs_name}", exist_ok=True)
    eval_obsoperator(
        era5_dir,
        obs_name,
        obs_dir,
        save_dir,
        start_year,
        end_year,
        model_name,
        debug,
        device,
    )

if __name__ == "__main__":
    main()

