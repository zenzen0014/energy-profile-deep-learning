"""Training and evaluation routines for the three-class CNN."""

import copy

import numpy as np
import torch

from . import CLASS_NAMES


def confusion_matrix(actual, predicted, n_classes=3):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual_class, predicted_class in zip(actual, predicted):
        matrix[int(actual_class), int(predicted_class)] += 1
    return matrix


def per_class_scores(matrix):
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def _batch_sem(losses):
    return float(np.std(losses, ddof=1) / np.sqrt(len(losses))) if len(losses) > 1 else 0.0


def evaluate(model, loader, criterion, device, decision_bias=None):
    """Evaluate one loader; rows stay aligned with a non-shuffled loader."""
    model.eval()
    total_loss, batch_losses, actual, predicted = 0.0, [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            batch_loss = criterion(logits, yb).item()
            total_loss += batch_loss * len(xb)
            batch_losses.append(batch_loss)

            if decision_bias is not None:
                logits = logits + torch.as_tensor(decision_bias, dtype=logits.dtype, device=device)

            actual.extend(yb.cpu().tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
            
    matrix = confusion_matrix(actual, predicted)
    return {
        "loss": total_loss / len(loader.dataset),
        "loss_sem": _batch_sem(batch_losses),
        "matrix": matrix,
        "actual": np.array(actual),
        "predicted": np.array(predicted),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                max_epochs=30, patience=6):
    """Train with early stopping on validation macro-F1."""
    history = []
    best_f1, best_state, best_epoch, no_improvement = -1.0, None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, batch_losses, actual, predicted = 0.0, [], [], []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            batch_losses.append(loss.item())
            actual.extend(yb.cpu().tolist())
            predicted.extend(logits.argmax(1).detach().cpu().tolist())

        train_matrix = confusion_matrix(actual, predicted)
        validation = evaluate(model, val_loader, criterion, device)
        train_f1 = float(per_class_scores(train_matrix)[2].mean())
        val_f1 = float(per_class_scores(validation["matrix"])[2].mean())

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / len(train_loader.dataset),
            "train_loss_sem": _batch_sem(batch_losses),
            "val_loss": validation["loss"],
            "val_loss_sem": validation["loss_sem"],
            "train_macro_f1": train_f1,
            "val_macro_f1": val_f1,
            "train_accuracy": np.trace(train_matrix) / train_matrix.sum(),
            "val_accuracy": np.trace(validation["matrix"]) / validation["matrix"].sum(),
        })
        print(f"Epoch {epoch:02d} | train loss {history[-1]['train_loss']:.4f} | "
              f"val loss {validation['loss']:.4f} | val macro-F1 {val_f1:.3f}")
        
        if val_f1 > best_f1 + 1e-4:
            best_f1, best_epoch, no_improvement = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print("Early stopping at epoch", epoch)
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_f1


def calibrate_peak_bias(model, val_loader, device):
    """Tune only the peak decision threshold on validation logits."""
    model.eval()
    actual, all_logits = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            all_logits.append(model(xb.to(device)).cpu().numpy())
            actual.extend(yb.tolist())
    actual = np.array(actual)
    logits = np.vstack(all_logits)
    best_score, peak_bias = -1.0, 0.0
    for candidate in np.linspace(-2.0, 1.0, 121):
        predicted = (logits + np.array([0.0, 0.0, candidate])).argmax(axis=1)
        score = float(per_class_scores(confusion_matrix(actual, predicted))[2].mean())
        if score > best_score:
            best_score, peak_bias = score, float(candidate)
    return np.array([0.0, 0.0, peak_bias], dtype=np.float32), best_score


def summarize_test(result, train_rows):
    """Return a JSON-ready test summary and majority-class baseline."""
    matrix = result["matrix"]
    precision, recall, f1 = per_class_scores(matrix)
    train_counts = np.bincount([int(row["class_id"]) for row in train_rows], minlength=3)
    majority = int(train_counts.argmax())
    baseline = confusion_matrix(result["actual"], np.full_like(result["actual"], majority))
    return {
        "test_loss": float(result["loss"]),
        "test_accuracy": float(np.trace(matrix) / matrix.sum()),
        "test_macro_f1": float(f1.mean()),
        "majority_baseline_macro_f1": float(per_class_scores(baseline)[2].mean()),
        "class_names": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(matrix[i].sum())}
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def predict_one(model, dataset, index, device, decision_bias=None):
    """Predict one dataset item and return its class and probabilities."""
    x, actual = dataset[index]
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
        if decision_bias is not None:
            logits = logits + torch.as_tensor(decision_bias, dtype=logits.dtype, device=device)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(actual), int(probabilities.argmax()), probabilities
