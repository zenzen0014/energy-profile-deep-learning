"""Preview and prediction charts for 24-hour sequences."""

from pathlib import Path

import matplotlib.pyplot as plt

from . import CLASS_NAMES
from .architecture import plot_architecture


def plot_profile_preview(rows, hourly, save_path):
    """Show one unmodified 24-hour profile from each class."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for class_id, ax in enumerate(axes):
        row = next(row for row in rows if int(row["class_id"]) == class_id)
        ax.plot(range(24), hourly[row["sample_id"]], color="tab:blue")
        ax.set(title=CLASS_NAMES[class_id], xlabel="Hour", ylabel="Electricity", xlim=(0, 23))
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_test_examples(rows, hourly, predicted, save_path):
    """Show a correct and incorrect test prediction per class when possible."""
    selected = []
    for class_id in range(len(CLASS_NAMES)):
        indices = [i for i, row in enumerate(rows) if int(row["class_id"]) == class_id]
        correct = next((i for i in indices if predicted[i] == class_id), None)
        incorrect = next((i for i in indices if predicted[i] != class_id), None)
        examples = [index for index in (correct, incorrect) if index is not None]
        examples.extend(index for index in indices if index not in examples)
        selected.extend(examples[:2])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, index in zip(axes.flat, selected):
        row = rows[index]
        actual = CLASS_NAMES[int(row["class_id"])]
        guess = CLASS_NAMES[int(predicted[index])]
        ax.plot(range(24), hourly[row["sample_id"]], color="tab:blue")
        ax.set(title=f"Actual: {actual}\nPredicted: {guess}", xlabel="Hour",
               ylabel="Electricity", xlim=(0, 23))
        ax.title.set_color("tab:green" if actual == guess else "tab:red")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _save(fig, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
