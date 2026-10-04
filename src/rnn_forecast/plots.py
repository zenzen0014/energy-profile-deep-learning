"""Learning and forecast visualizations for the RNN notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def plot_learning_curves(history, save_path):
    """Plot normalized training loss and original-scale validation MAE."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Training loss")
    axes[1].plot(epochs, [row["val_mae"] for row in history], color="tab:orange")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Validation forecast error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend() if axis is axes[0] else None
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_forecast_examples(sequences, hourly, actual, predicted, save_path, n_examples=3):
    """Compare the last observed day, actual next day, and forecast for test samples."""
    indices = np.linspace(0, len(sequences) - 1, num=min(n_examples, len(sequences)), dtype=int)
    fig, axes = plt.subplots(len(indices), 1, figsize=(11, 3.5 * len(indices)), squeeze=False)
    for axis, index in zip(axes.flat, indices):
        sequence = sequences[index]
        previous_day = hourly[sequence["input_sample_ids"][-1]]
        hours = range(24)
        axis.plot(hours, previous_day, color="0.55", linestyle="--", label="Previous day")
        axis.plot(hours, actual[index], color="tab:green", label="Actual next day")
        axis.plot(hours, predicted[index], color="tab:orange", label="RNN forecast")
        axis.set(
            title=f"{sequence['building_id']} | target {sequence['target_date']}",
            xlabel="Hour",
            ylabel="Electricity",
            xlim=(0, 23),
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
