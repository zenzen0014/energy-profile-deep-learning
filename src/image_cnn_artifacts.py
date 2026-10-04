"""Reproducible artifact saving for TensorFlow image CNN runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def save_run(output_dir, model, history, metrics, splits, normalization, seed, best_epoch):
    """Save a Keras model, history, split manifest, and test report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = splits

    model.save(output_dir / "basic_cnn.keras")

    history_keys = list(history.history)
    history_rows = [
        {key: float(history.history[key][epoch]) for key in history_keys}
        for epoch in range(len(history.history[history_keys[0]]))
    ]
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_keys)
        writer.writeheader()
        writer.writerows(history_rows)

    manifest_fields = list(train_rows[0]) if train_rows else []
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*manifest_fields, "split"])
        writer.writeheader()
        for split_name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
            for row in rows:
                writer.writerow({key: row[key] for key in manifest_fields} | {"split": split_name})

    report = {
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "normalization": normalization,
        "split_unit": "profile",
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "test_count": len(test_rows),
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
