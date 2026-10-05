"""Learning and forecast visualizations for the RNN notebook."""

from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def plot_learning_curves(history, save_path):
    """Plot normalized training/validation MSE and validation MAE."""
    epochs = [row["epoch"] for row in history]
    last = history[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history],
                 label=f"Train (last: {last['train_loss']:.5f})")
    axes[0].plot(epochs, [row["val_loss"] for row in history],
                 label=f"Validation (last: {last['val_loss']:.5f})")
    axes[0].set(xlabel="Epoch", ylabel="Normalized MSE", title="Training loss", ylim=(0, None))
    axes[1].plot(epochs, [row["val_mae"] for row in history], color="tab:orange",
                 label=f"Validation MAE (last: {last['val_mae']:.2f})")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Validation forecast error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_three_dataset_samples(building_rows, hourly, save_path):
    """Show three labeled daily load profiles from evenly spaced buildings."""
    building_ids = sorted(building_rows)
    if len(building_ids) < 3:
        raise ValueError("At least three buildings are required")
    selected = np.linspace(0, len(building_ids) - 1, num=3, dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for number, (axis, index) in enumerate(zip(axes, selected), start=1):
        row = building_rows[building_ids[index]][len(building_rows[building_ids[index]]) // 2]
        axis.plot(range(24), hourly[row["sample_id"]], color="tab:blue", marker="o", markersize=3)
        axis.set(title=f"Sample {number}: {row['building_id']} | {row['date']}",
                 xlabel="Hour", xlim=(0, 23))
        axis.set_xticks((0, 6, 12, 18, 23))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Electricity")
    fig.suptitle("Simple dataset representation: 3 daily load profiles", y=1.03)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_forecast_examples(sequences, hourly, actual, predicted, save_path, n_examples=3,
                           selection="even"):
    """Show the previous day beside actual, forecast, and hourly error."""
    count = min(n_examples, len(sequences))
    if selection == "best":
        daily_mae = np.abs(predicted - actual).mean(axis=1)
        daily_range = np.ptp(actual, axis=1)
        ranked = np.argsort(daily_mae / np.maximum(daily_range, 1), kind="stable")
        indices, buildings = [], set()
        for index in ranked:
            building = sequences[index]["building_id"]
            if building not in buildings:
                indices.append(int(index))
                buildings.add(building)
            if len(indices) == count:
                break
        if len(indices) < count:
            indices.extend(int(index) for index in ranked if int(index) not in indices)
        indices = indices[:count]
    elif selection == "even":
        indices = np.linspace(0, len(sequences) - 1, num=count, dtype=int)
    else:
        raise ValueError("selection must be 'even' or 'best'")
    fig, axes = plt.subplots(len(indices), 2, figsize=(13, 3.4 * len(indices)),
                             squeeze=False, sharey='row')
    for (previous_axis, target_axis), index in zip(axes, indices):
        sequence = sequences[index]
        previous_day = hourly[sequence["input_sample_ids"][-1]]
        hours = range(24)
        previous_axis.plot(hours, previous_day, color="0.55", label="Previous day")
        actual_line, = target_axis.plot(hours, actual[index], color="tab:green", label="Actual")
        forecast_line, = target_axis.plot(hours, predicted[index], color="tab:orange", label="GRU forecast")
        hourly_error = np.abs(predicted[index] - actual[index])
        error_axis = target_axis.twinx()
        error_bars = error_axis.bar(hours, hourly_error, width=0.7, color="tab:red", alpha=0.25,
                                    label="Absolute error")
        error_axis.set(ylabel="Absolute error", ylim=(0, max(float(hourly_error.max()) * 1.5, 1)))
        error_axis.tick_params(axis="y", colors="tab:red")
        error_axis.yaxis.label.set_color("tab:red")
        error_axis.set_zorder(target_axis.get_zorder() - 1)
        target_axis.patch.set_visible(False)
        previous_date = date.fromisoformat(sequence["target_date"]) - timedelta(days=1)
        previous_axis.set(title=f"{sequence['building_id']} | Previous day {previous_date}",
                          xlabel="Hour", ylabel="Electricity", xlim=(0, 23))
        sample_mae = hourly_error.mean()
        target_axis.set(title=f"Current day {sequence['target_date']} | MAE {sample_mae:.1f}",
                        xlabel="Hour", xlim=(0, 23))
        for axis in (previous_axis, target_axis):
            axis.set_xticks((0, 6, 12, 18, 23))
            axis.grid(alpha=0.25)
        previous_axis.legend(fontsize=8)
        target_axis.legend([actual_line, forecast_line, error_bars],
                           ["Actual", "GRU forecast", "Absolute error"], fontsize=8)
    if selection == "best":
        fig.suptitle("Best test examples: lowest MAE / daily load range; one per building",
                     fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985 if selection == "best" else 1))
    _save(fig, save_path)
    return fig
