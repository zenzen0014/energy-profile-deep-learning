"""Figures used by the student-facing CNN notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.architecture_volume import plot_architecture
from . import CLASS_NAMES


def plot_dataset_preview(rows, hourly, reference, data_dir, save_path):
    """Show one 24-hour curve and Pearson image from each class."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for class_id in range(3):
        row = next(row for row in rows if int(row["class_id"]) == class_id)
        profile = hourly[row["sample_id"]]
        baseline = reference[row["building_id"]]
        axes[class_id, 0].plot(range(24), profile, label="Target day")
        axes[class_id, 0].plot(range(24), baseline, "--", label="30-day reference")
        axes[class_id, 0].set(title=f"{CLASS_NAMES[class_id]} | {row['date']}",
                              xlabel="Hour", ylabel="Electricity")
        axes[class_id, 0].grid(alpha=0.2)
        if class_id == 0:
            axes[class_id, 0].legend(fontsize=8)
        with Image.open(Path(data_dir) / row["image_path"]) as image:
            axes[class_id, 1].imshow(image)
        axes[class_id, 1].set(title="Pearson 24×24", xticks=[], yticks=[])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_learning_curves(history, save_path):
    """Compare train and validation loss, accuracy, and macro-F1."""
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ("loss", "accuracy", "macro_f1"),
                                  ("Loss", "Accuracy", "Macro-F1")):
        for prefix, label, color in (("train", "Train", "tab:blue"),
                                     ("val", "Validation", "tab:orange")):
            values = [row[f"{prefix}_{metric}"] for row in history]
            ax.plot(epochs, values, label=f"{label}: {values[-1]:.3f}", color=color)
        ax.legend(title=f"Final epoch: {epochs[-1]}")
        ax.set(xlabel="Epoch", ylabel=title, title=title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(matrix, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual",
           title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for actual in range(3):
        for predicted in range(3):
            color = "white" if matrix[actual, predicted] > matrix.max() / 2 else "black"
            ax.text(predicted, actual, str(matrix[actual, predicted]),
                    ha="center", va="center", color=color)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_test_examples(rows, predicted, data_dir, save_path):
    """Show one correct and one mistaken test image per actual class when possible."""
    selected = []
    for class_id in range(3):
        indices = [i for i, row in enumerate(rows) if int(row["class_id"]) == class_id]
        correct = next((i for i in indices if predicted[i] == class_id), None)
        mistake = next((i for i in indices if predicted[i] != class_id), None)
        examples = [index for index in (correct, mistake) if index is not None]
        examples.extend(index for index in indices if index not in examples)
        selected.extend(examples[:2])
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, index in zip(axes.flat, selected):
        row = rows[index]
        with Image.open(Path(data_dir) / row["image_path"]) as image:
            ax.imshow(image, interpolation="nearest")
        true_name = CLASS_NAMES[int(row["class_id"])]
        predicted_name = CLASS_NAMES[int(predicted[index])]
        ax.set(title=f"Actual: {true_name}\nPredicted: {predicted_name}", xticks=[], yticks=[])
        ax.title.set_color("tab:green" if true_name == predicted_name else "tab:red")
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_model_comparison(main_metrics, baseline_metrics, save_path):
    """Optional comparison with the Basic CNN baseline."""
    names = ("Three-channel CNN", "Basic Pearson CNN")
    models = (main_metrics, baseline_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x, width = np.arange(2), 0.35
    for index, (name, metrics) in enumerate(zip(names, models)):
        axes[0].bar(x + (index - 0.5) * width,
                    [metrics["test_accuracy"], metrics["test_macro_f1"]], width, label=name)
        axes[1].bar(np.arange(3) + (index - 0.5) * width,
                    [metrics["per_class"][class_name]["f1"] for class_name in CLASS_NAMES],
                    width, label=name)
    axes[0].set(xticks=x, xticklabels=["Accuracy", "Macro-F1"], ylim=(0, 1.05), title="Test scores")
    axes[1].set(xticks=np.arange(3), xticklabels=CLASS_NAMES, ylim=(0, 1.05), title="F1 by class")
    for ax in axes:
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _save(fig, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
