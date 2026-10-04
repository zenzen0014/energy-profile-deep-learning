"""Load 24 hourly values directly, without Pearson images or building references."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import CLASS_NAMES


def load_profiles(data_dir):
    """Return profile labels and h00–h23 values from samples.csv."""
    path = Path(data_dir) / "samples.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"No profiles found in {path}")

    records, hourly = [], {}
    for source in source_rows:
        sample_id = source["sample_id"]
        class_id = int(source["class_id"])
        if sample_id in hourly:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if class_id not in range(len(CLASS_NAMES)) or source["class_name"] != CLASS_NAMES[class_id]:
            raise ValueError(f"Class label mismatch: {sample_id}")
        values = np.array([float(source[f"h{hour:02d}"]) for hour in range(24)], dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid hourly values: {sample_id}")
        records.append({key: source[key] for key in
                        ("sample_id", "building_id", "date", "class_id", "class_name")})
        hourly[sample_id] = values
    return records, hourly


def fit_normalization(train_rows, hourly):
    """Fit log-scale normalization on training profiles only."""
    values = np.concatenate([hourly[row["sample_id"]] for row in train_rows])
    log_values = np.log1p(values)
    center = float(np.median(log_values))
    scale = float(np.percentile(log_values, 95) - np.percentile(log_values, 5))
    if scale <= 0:
        raise ValueError("Normalization scale must be positive")
    return {"log_center": center, "log_scale": scale}


class HourWindowDataset(Dataset):
    """Return one normalized 24-hour sequence with shape (1, 24)."""

    def __init__(self, rows, hourly, normalization):
        self.rows = list(rows)
        profiles = np.stack([hourly[row["sample_id"]] for row in self.rows])
        values = (np.log1p(profiles) - normalization["log_center"]) / normalization["log_scale"]
        self.inputs = torch.from_numpy(values[:, None, :].astype(np.float32))
        self.labels = torch.tensor([int(row["class_id"]) for row in self.rows], dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
