"""Save a reproducible CNN run without cluttering the teaching notebook."""

from pathlib import Path
import csv
import json

import torch

from . import CLASS_NAMES


def save_run(output_dir, model, history, metrics, splits, normalization,
             decision_bias, seed, best_epoch, validation_macro_f1, calibrated_macro_f1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, test_rows = splits
    torch.save({
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "class_names": CLASS_NAMES,
        "architecture": type(model).__name__,
        "input_shape": (3, 24, 24),
        "seed": seed,
        "best_epoch": best_epoch,
        "normalization": normalization,
        "decision_bias": decision_bias.tolist(),
        "validation_macro_f1_calibrated": calibrated_macro_f1,
        "split_unit": "profile",
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
    }, output_dir / "best_model.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "building_id", "class_id", "class_name", "split", "image_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
            for row in rows:
                writer.writerow({key: row[key] for key in fields if key != "split"} | {"split": split_name})
    report = {"best_epoch": best_epoch, "validation_macro_f1": validation_macro_f1,
              "split_unit": "profile", **metrics}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
