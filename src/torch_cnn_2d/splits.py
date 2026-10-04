"""Reusable class-stratified splits for daily load profiles."""

from collections import Counter

import numpy as np

from . import CLASS_NAMES


def split_profiles(records, seed=42, train_fraction=0.6, val_fraction=0.2):
    """Split individual profiles, regardless of building ID."""
    rng = np.random.default_rng(seed)
    train_rows, val_rows, test_rows = [], [], []
    for class_id in range(len(CLASS_NAMES)):
        class_rows = [row for row in records if int(row["class_id"]) == class_id]
        order = rng.permutation(len(class_rows))
        n_train = round(train_fraction * len(class_rows))
        n_val = round(val_fraction * len(class_rows))
        train_rows.extend(class_rows[index] for index in order[:n_train])
        val_rows.extend(class_rows[index] for index in order[n_train:n_train + n_val])
        test_rows.extend(class_rows[index] for index in order[n_train + n_val:])
    for rows in (train_rows, val_rows, test_rows):
        rng.shuffle(rows)
    all_ids = [row["sample_id"] for rows in (train_rows, val_rows, test_rows) for row in rows]
    if len(all_ids) != len(records) or len(set(all_ids)) != len(records):
        raise ValueError("The split lost or duplicated samples")
    return train_rows, val_rows, test_rows


def class_counts(rows):
    counts = Counter(row["class_name"] for row in rows)
    return {name: counts[name] for name in CLASS_NAMES}
