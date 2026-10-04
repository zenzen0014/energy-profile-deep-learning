"""Training and evaluation routines for hourly load-profile forecasting."""

from __future__ import annotations

import copy

import numpy as np
import torch

from .data import inverse_normalize


def evaluate(model, loader, criterion, device, normalization):
    """Evaluate normalized MSE and original-scale forecasting metrics."""
    model.eval()
    loss_sum, actual, predicted = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            forecast = model(xb)
            loss_sum += criterion(forecast, yb).item() * len(xb)
            actual.append(yb.cpu().numpy())
            predicted.append(forecast.cpu().numpy())
    actual_raw = inverse_normalize(np.vstack(actual), normalization)
    predicted_raw = inverse_normalize(np.vstack(predicted), normalization)
    error = predicted_raw - actual_raw
    return {
        "loss": loss_sum / len(loader.dataset),
        "actual": actual_raw,
        "predicted": predicted_raw,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / np.maximum(actual_raw.sum(), 1e-6) * 100),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, normalization,
                max_epochs=30, patience=6):
    """Train with early stopping on original-scale validation MAE."""
    history, best_state = [], None
    best_mae, best_epoch, no_improvement = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            forecast = model(xb)
            loss = criterion(forecast, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
        validation = evaluate(model, val_loader, criterion, device, normalization)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader.dataset),
            "val_loss": validation["loss"],
            "val_mae": validation["mae"],
            "val_rmse": validation["rmse"],
        }
        history.append(row)
        print(f"Epoch {epoch:02d} | train MSE {row['train_loss']:.4f} | "
              f"val MAE {row['val_mae']:.3f} | val RMSE {row['val_rmse']:.3f}")
        if row["val_mae"] < best_mae - 1e-6:
            best_mae, best_epoch, no_improvement = row["val_mae"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print("Early stopping at epoch", epoch)
            break
    model.load_state_dict(best_state)
    return history, best_epoch, best_mae


def summarize_test(result):
    """Return JSON-ready metrics for a next-day hourly profile forecast."""
    return {
        "test_normalized_mse": float(result["loss"]),
        "test_mae": float(result["mae"]),
        "test_rmse": float(result["rmse"]),
        "test_wape_percent": float(result["wape"]),
    }
