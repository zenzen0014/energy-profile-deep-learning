"""Reproducible artifact saving for RNN forecasting runs."""

import csv
import json
from pathlib import Path

import torch


def save_run(output_dir, model, history, metrics, splits, normalization, seed, best_epoch, lookback_days):
    """Save model weights, learning history, chronological split manifest, and report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_sequences, val_sequences, test_sequences = splits
    torch.save({
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "architecture": type(model).__name__,
        "recent_weight": getattr(model, "recent_weight", None),
        "seasonal_lag": getattr(model, "seasonal_lag", None),
        "input_shape": [lookback_days, 24],
        "forecast_horizon_hours": 24,
        "normalization": normalization,
        "seed": seed,
        "best_epoch": best_epoch,
        "split_unit": "chronological_target_date_per_building",
    }, output_dir / "best_model.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fields = ["building_id", "target_date", "target_sample_id", "input_sample_ids", "split"]
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name, sequences in (("train", train_sequences), ("validation", val_sequences), ("test", test_sequences)):
            for sequence in sequences:
                writer.writerow({
                    "building_id": sequence["building_id"],
                    "target_date": sequence["target_date"],
                    "target_sample_id": sequence["target_sample_id"],
                    "input_sample_ids": "|".join(sequence["input_sample_ids"]),
                    "split": split_name,
                })
    report = {
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "lookback_days": int(lookback_days),
        "forecast_horizon_hours": 24,
        "recent_weight": getattr(model, "recent_weight", None),
        "seasonal_lag": getattr(model, "seasonal_lag", None),
        "normalization": normalization,
        "split_unit": "chronological_target_date_per_building",
        "train_count": len(train_sequences),
        "validation_count": len(val_sequences),
        "test_count": len(test_sequences),
        **metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
