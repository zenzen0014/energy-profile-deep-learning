"""Chronological daily-profile loading and sequence preparation."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import HOURS_PER_DAY


def load_daily_profiles(data_dir):
    """Load daily 24-hour profiles, grouped and ordered by building."""
    path = Path(data_dir) / "samples.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"No profiles found in {path}")

    grouped, hourly = defaultdict(list), {}
    for source in source_rows:
        sample_id = source["sample_id"]
        if sample_id in hourly:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        values = np.array(
            [float(source[f"h{hour:02d}"]) for hour in range(HOURS_PER_DAY)], dtype=np.float32
        )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid hourly values: {sample_id}")
        row = {
            "sample_id": sample_id,
            "building_id": source["building_id"],
            "date": source["date"],
        }
        grouped[row["building_id"]].append(row)
        hourly[sample_id] = values

    for building_id, rows in grouped.items():
        rows.sort(key=lambda row: row["date"])
        dates = [date.fromisoformat(row["date"]) for row in rows]
        if any((later - earlier).days != 1 for earlier, later in zip(dates, dates[1:])):
            raise ValueError(f"Missing daily observations for {building_id}")
    return dict(grouped), hourly


def make_forecast_splits(building_rows, lookback_days=7, train_fraction=0.6, val_fraction=0.2):
    """Create chronological next-day sequences independently for every building."""
    if lookback_days < 1 or not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("lookback and split fractions must be positive")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one")

    splits = [[], [], []]
    for building_id, rows in sorted(building_rows.items()):
        if len(rows) <= lookback_days:
            raise ValueError(f"Not enough rows for {building_id}")
        sequences = [
            {
                "building_id": building_id,
                "input_sample_ids": [row["sample_id"] for row in rows[target - lookback_days:target]],
                "target_sample_id": rows[target]["sample_id"],
                "target_date": rows[target]["date"],
            }
            for target in range(lookback_days, len(rows))
        ]
        train_end = round(train_fraction * len(sequences))
        val_end = round((train_fraction + val_fraction) * len(sequences))
        splits[0].extend(sequences[:train_end])
        splits[1].extend(sequences[train_end:val_end])
        splits[2].extend(sequences[val_end:])
    return tuple(splits)


def fit_normalization(sequences, hourly):
    """Fit log-scale normalization using training observations only."""
    sample_ids = {
        sample_id
        for sequence in sequences
        for sample_id in [*sequence["input_sample_ids"], sequence["target_sample_id"]]
    }
    values = np.concatenate([hourly[sample_id] for sample_id in sorted(sample_ids)])
    log_values = np.log1p(values)
    center = float(np.median(log_values))
    scale = float(np.percentile(log_values, 95) - np.percentile(log_values, 5))
    if scale <= 0:
        raise ValueError("Normalization scale must be positive")
    return {"method": "log1p_median_iqr90", "log_center": center, "log_scale": scale}


def inverse_normalize(values, normalization):
    """Convert normalized log-scale profiles back to the original load scale."""
    restored = np.expm1(values * normalization["log_scale"] + normalization["log_center"])
    return np.maximum(restored, 0.0)


class DailyForecastDataset(Dataset):
    """Return a normalized lookback sequence and its next-day target profile."""

    def __init__(self, sequences, hourly, normalization):
        self.sequences = list(sequences)
        self.normalization = normalization
        inputs = np.stack([
            np.stack([hourly[sample_id] for sample_id in sequence["input_sample_ids"]])
            for sequence in self.sequences
        ])
        targets = np.stack([hourly[sequence["target_sample_id"]] for sequence in self.sequences])
        self.inputs = torch.from_numpy(self._normalize(inputs).astype(np.float32))
        self.targets = torch.from_numpy(self._normalize(targets).astype(np.float32))

    def _normalize(self, values):
        return (np.log1p(values) - self.normalization["log_center"]) / self.normalization["log_scale"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
