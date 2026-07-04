import sys
import os
from pathlib import Path
import pickle
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import json
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(name)s - %(levelname)s - %(message)s')

def plot_obsop_omb(
    tgt_tmbrs_values: np.ndarray,
    out_tmbrs_values: np.ndarray,
    mask: np.ndarray,
    variable_name: str,
    plot_dir: str,
) -> None:
    """
    Plot histogram of differences (prepbufr - ERA5) multiplied by mask with probability density.
    Args:
    obs_data: 1D numpy array containing original observations
    era5_data: 1D numpy array containing ERA5 data
    mask: 1D numpy array containing mask (0 or 1)
    variable_name: Name of the variable being compared
    Raises:
    ValueError: If input arrays have different lengths
    """
    if len(tgt_tmbrs_values) != len(out_tmbrs_values) or len(tgt_tmbrs_values) != len(mask):
        raise ValueError("All input arrays must have the same length")

    # Calculate differences multiplied by mask
    original_differences = (tgt_tmbrs_values - out_tmbrs_values) * mask

    # Filter out NaN values (where mask is 0)
    valid_mask = mask == 1
    diff_values = original_differences[valid_mask]

    if len(original_differences) == 0:
        logging.info(f"No valid data points for {variable_name}")
        return

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create histogram with probability density
    # sns.histplot(diff_values, bins=50, kde=True, color='blue', alpha=0.3, stat='density')
    plt.hist(diff_values, bins=50, density=True, alpha=0.3, color='blue', label='OMB')

    # 👇 手动添加 KDE 曲线 (使用 scipy，完全不依赖 pandas/seaborn)
    # 过滤掉 inf/nan 防止 scipy 报错
    finite_values = diff_values[np.isfinite(diff_values)]
    if len(finite_values) > 1:
        kde = stats.gaussian_kde(finite_values)
        x_range = np.linspace(np.min(finite_values), np.max(finite_values), 200)
        plt.plot(x_range, kde(x_range), color='blue', linewidth=2)

    # Add vertical line at zero (no difference)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No difference')

    # Add labels and title
    plt.xlabel(f'OMB/Error ({variable_name})')
    plt.ylabel('Probability Density')
    plt.legend(loc='best')

    # Add statistics
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    plt.text(
        0.02, 0.95, 
        f'Mean: {mean_diff:.3f}\nStd: {std_diff:.3f}\n',
        transform=plt.gca().transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/obsop_omb_{variable_name}.jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{plot_dir}/obsop_omb_{variable_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()