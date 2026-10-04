"""Read the daily profile dataset and prepare CNN inputs."""

from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import CLASS_NAMES
from .splits import class_counts, split_profiles


def load_records(data_dir):
    """Return image records, 24-hour profiles, and calibration profiles."""
    data_dir = Path(data_dir)
    manifest = data_dir / "metadata/image_manifest.csv"
    samples = data_dir / "samples.csv"
    references = data_dir / "metadata/building_references.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    with samples.open(newline="", encoding="utf-8") as handle:
        sample_rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with references.open(newline="", encoding="utf-8") as handle:
        reference_rows = {row["building_id"]: row for row in csv.DictReader(handle)}

    if not records or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("The manifest is empty or contains duplicate sample_id values")
    hourly = {
        sample_id: np.array([float(row[f"h{hour:02d}"]) for hour in range(24)], dtype=np.float32)
        for sample_id, row in sample_rows.items()
    }
    reference = {
        building_id: np.array([float(row[f"h{hour:02d}"]) for hour in range(24)], dtype=np.float32)
        for building_id, row in reference_rows.items()
    }
    for row in records:
        if row["class_name"] != CLASS_NAMES[int(row["class_id"])]:
            raise ValueError(f"Class label mismatch: {row['sample_id']}")
        if row["sample_id"] not in hourly or row["building_id"] not in reference:
            raise ValueError(f"Missing profile or reference: {row['sample_id']}")
        if not (data_dir / row["image_path"]).is_file():
            raise FileNotFoundError(data_dir / row["image_path"])
    return records, hourly, reference


def fit_normalization(train_rows, hourly, reference):
    """Calculate image-channel scaling from training profiles only."""
    train_hourly = np.concatenate([hourly[row["sample_id"]] for row in train_rows])
    train_log = np.log1p(train_hourly)
    center = float(np.median(train_log))
    scale = float(np.percentile(train_log, 95) - np.percentile(train_log, 5))
    deviation = np.concatenate([
        np.log1p(hourly[row["sample_id"]]) - np.log1p(reference[row["building_id"]])
        for row in train_rows
    ])
    deviation_scale = float(np.percentile(np.abs(deviation), 90))
    if scale <= 0 or deviation_scale <= 0:
        raise ValueError("Normalization scales must be positive")
    return {"level_center": center, "level_scale": scale, "deviation_scale": deviation_scale}


def make_input_maps(image_path, profile, reference, normalization):
    """Return Pearson, load-level, and reference-deviation maps (3, 24, 24)."""
    with Image.open(image_path) as image:
        if image.size != (24, 24):
            raise ValueError(f"Image must be 24x24: {image_path}")
        pearson = np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 0].copy() / 255.0
    level = np.clip(
        (np.log1p(profile) - normalization["level_center"]) / normalization["level_scale"] + 0.5,
        0.0, 1.0,
    )
    deviation = 0.5 + 0.5 * np.tanh(
        (np.log1p(profile) - np.log1p(reference)) / normalization["deviation_scale"]
    )
    return np.stack([
        pearson,
        np.tile(level, (24, 1)),
        np.tile(deviation, (24, 1)),
    ]).astype(np.float32)


class LoadProfileDataset(Dataset):
    """One item is a three-channel image and its integer class label."""

    def __init__(self, data_dir, rows, hourly, reference, normalization):
        self.data_dir = Path(data_dir)
        self.rows = list(rows)
        self.hourly = hourly
        self.reference = reference
        self.normalization = normalization

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        maps = make_input_maps(
            self.data_dir / row["image_path"],
            self.hourly[row["sample_id"]],
            self.reference[row["building_id"]],
            self.normalization,
        )
        return torch.from_numpy(maps), torch.tensor(int(row["class_id"]), dtype=torch.long)
