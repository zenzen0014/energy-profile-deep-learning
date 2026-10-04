"""Test-sample utilities for TensorFlow image CNN notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_test_examples(rows, predicted, data_dir: Path, save_path: Path, class_names=None):
    """Plot a correct and an incorrect image prediction for every actual class."""
    predicted = np.asarray(predicted)
    if len(rows) != len(predicted):
        raise ValueError("rows and predicted must contain the same number of samples")
    if class_names is None:
        names_by_id = {int(row["class_id"]): row["class_name"] for row in rows}
        class_names = tuple(names_by_id[class_id] for class_id in range(len(names_by_id)))

    selected = []
    for class_id in range(len(class_names)):
        indices = [i for i, row in enumerate(rows) if int(row["class_id"]) == class_id]
        correct = next((i for i in indices if predicted[i] == class_id), None)
        mistake = next((i for i in indices if predicted[i] != class_id), None)
        examples = [index for index in (correct, mistake) if index is not None]
        examples.extend(index for index in indices if index not in examples)
        selected.extend(examples[:2])

    fig, axes = plt.subplots(2, len(class_names), figsize=(10, 7), squeeze=False)
    for ax, index in zip(axes.flat, selected):
        row = rows[index]
        with Image.open(Path(data_dir) / row["image_path"]) as image:
            ax.imshow(image, interpolation="nearest")
        actual_name = class_names[int(row["class_id"])]
        predicted_name = class_names[int(predicted[index])]
        ax.set(title=f"Actual: {actual_name}\nPredicted: {predicted_name}", xticks=[], yticks=[])
        ax.title.set_color("tab:green" if actual_name == predicted_name else "tab:red")
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def predict_one(model, images: np.ndarray, labels: np.ndarray, index: int = 0):
    """Predict one TensorFlow/Keras image and return actual class, prediction, probabilities."""
    if not 0 <= index < len(images) or len(images) != len(labels):
        raise IndexError("index must refer to one image and label in the test split")

    probabilities = np.asarray(model.predict(images[index:index + 1], verbose=0))[0]
    return int(labels[index]), int(probabilities.argmax()), probabilities
