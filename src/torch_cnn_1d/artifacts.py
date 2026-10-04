"""Save a reproducible 1D-CNN run."""

import csv
import json
from pathlib import Path

import torch

from . import CLASS_NAMES


def save_run(output_dir, model, history, metrics, splits, normalization, seed, best_epoch):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = splits
    torch.save({
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": type(model).__name__,
        "input_shape": (1, 24),
        "input_fields": [f"h{hour:02d}" for hour in range(24)],
        "class_names": CLASS_NAMES,
        "normalization": normalization,
        "seed": seed,
        "best_epoch": best_epoch,
        "split_unit": "profile",
    }, output_dir / "best_model.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "building_id", "class_id", "class_name", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
            writer.writerows({**{key: row[key] for key in fields if key != "split"},
                              "split": split_name} for row in rows)
    report = {"best_epoch": best_epoch, "split_unit": "profile", "input_shape": [1, 24], **metrics}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
