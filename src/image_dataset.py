"""Dataset loading helpers for the TensorFlow Basic CNN notebook."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_image_splits(
    data_dir: Path,
    class_names: tuple[str, ...],
    image_size: tuple[int, int],
    seed: int,
    expected_records: int | None = 6720,
    return_rows: bool = False,
):
    """Load RGB images and create reproducible 60/20/20 class-stratified splits."""
    with (data_dir / "metadata/image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    if expected_records is not None:
        assert len(records) == expected_records
    assert all(row["class_name"] == class_names[int(row["class_id"])] for row in records)
    assert all((data_dir / row["image_path"]).is_file() for row in records)

    rng = np.random.default_rng(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for class_id in range(len(class_names)):
        rows = [row for row in records if int(row["class_id"]) == class_id]
        order = rng.permutation(len(rows))
        train_end, val_end = round(0.6 * len(rows)), round(0.8 * len(rows))
        train_rows += [rows[i] for i in order[:train_end]]
        val_rows += [rows[i] for i in order[train_end:val_end]]
        test_rows += [rows[i] for i in order[val_end:]]

    for rows in (train_rows, val_rows, test_rows):
        rng.shuffle(rows)

    def load_split(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        images = np.empty((len(rows), *image_size, 3), dtype=np.float32)
        labels = np.empty(len(rows), dtype=np.int32)
        for i, row in enumerate(rows):
            with Image.open(data_dir / row["image_path"]) as image:
                images[i] = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            labels[i] = int(row["class_id"])
        return images, labels

    splits = load_split(train_rows), load_split(val_rows), load_split(test_rows)
    if return_rows:
        return *splits, (train_rows, val_rows, test_rows)
    return splits
