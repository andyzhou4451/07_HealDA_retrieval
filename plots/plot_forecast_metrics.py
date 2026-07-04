import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def plot_forecast_metrics(
    rmse: np.ndarray,
    acc: np.ndarray,
    activity: np.ndarray,
    variables: list,
    title: str = "Forecast Metrics",
    cmap: str = "viridis",
    figsize: tuple = (20, 16),
    dpi: int = 300
) -> List[plt.Figure]:
    """Generate forecast metric line plots for RMSE, ACC, and activity in a 4x5 grid."""

    figures = []

    # Define the variables we want to plot (300hPa level)
    target_vars = [
        'z-300', 't-300', 'u-300', 'v-300', 'q-300',
        'z-500', 't-500', 'u-500', 'v-500', 'q-500',
        'z-850', 't-850', 'u-850', 'v-850', 'q-850',
        't2m', 'u10', 'v10', 'msl'
    ]

    # Get the indices of these variables in the variables list
    var_indices = [variables.index(var) for var in target_vars if var in variables]

    # Metrics to plot
    metrics = {
        'RMSE': rmse,
        'ACC': acc,
        'Activity': activity
    }

    for metric_name, metric_data in metrics.items():
        # Select only the data for our target variables
        selected_data = metric_data[:, var_indices]

        # Create figure with 4x5 grid of subplots
        fig, axes = plt.subplots(4, 5, figsize=figsize, dpi=dpi)
        fig.suptitle(f"{title} - {metric_name}", fontsize=16)

        # Flatten axes array for easy iteration
        axes = axes.ravel()

        # Create x-axis (lead times in hours)
        lead_times = np.arange(0, selected_data.shape[0] * 6, 6)

        # Plot each variable in its own subplot
        for i, (var, ax) in enumerate(zip(target_vars, axes)):
            if var in variables:  # Only plot if variable exists in data
                ax.plot(lead_times, selected_data[:, i], label=var, marker='o', markersize=4)
                ax.set_xlabel('Lead Time (hours)', fontsize=10)
                ax.set_ylabel(metric_name, fontsize=10)
                ax.set_title(var, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            else:
                ax.axis('off')  # Hide subplot if variable not in data

        # Adjust layout
        plt.tight_layout()
        figures.append(fig)

    return figures

def save_forecast_plots(
    figures: List[plt.Figure],
    output_dir: str,
    prefix: str = ["rmse", "acc", "activity"]
) -> None:
    """
    Save the forecast metric plots to files.

    Args:
        figures: List of matplotlib figures to save
        output_dir: Directory to save the plots
        prefix: Prefix for output filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, fig in enumerate(figures):
        filename = f"{prefix[i]}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=fig.dpi)
        plt.close(fig)