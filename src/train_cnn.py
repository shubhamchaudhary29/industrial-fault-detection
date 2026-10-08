"""
Industrial Fault Detection - Lightweight CNN Classifier Training
=================================================================
Trains an end-to-end Convolutional Neural Network on preprocessed surface images.

Architecture:
- ConvBlock 1: Conv2d(1->32, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2)
- ConvBlock 2: Conv2d(32->64, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2)
- ConvBlock 3: Conv2d(64->128, 3x3) + BatchNorm + ReLU + MaxPool2d(2x2)
- Global Average Pooling (AdaptiveAvgPool2d(1, 1))
- Dense Head: Linear(128->64) + ReLU + Dropout(0.4) + Linear(64->1)
- Loss: Binary Cross-Entropy with Logits
- Checkpointing & Early Stopping on Validation Loss
"""

import os
import copy
import time
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class LightweightFaultCNN(nn.Module):
    """
    Lightweight 3-stage CNN tailored for industrial defect classification.
    """

    def __init__(self, in_channels: int = 1, dropout_rate: float = 0.4):
        super(LightweightFaultCNN, self).__init__()

        # Conv Block 1: (B, 1, 224, 224) -> (B, 32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Block 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling: (B, 128, 28, 28) -> (B, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        logits = self.classifier(x)
        return logits.squeeze(-1)


def prepare_tensors(X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts NumPy arrays (N, H, W, C) to PyTorch tensors (N, C, H, W) and (N,).
    """
    # Transpose (N, H, W, C) -> (N, C, H, W)
    if X.ndim == 4:
        X_transposed = np.transpose(X, (0, 3, 1, 2))
    elif X.ndim == 3:
        X_transposed = np.expand_dims(X, axis=1)
    else:
        raise ValueError(f"Unexpected shape for X: {X.shape}")

    tensor_x = torch.from_numpy(X_transposed.astype(np.float32))
    tensor_y = torch.from_numpy(y.astype(np.float32))
    return tensor_x, tensor_y


def plot_training_curves(history: Dict[str, List[float]], output_path: str = "outputs/training_curves.png"):
    """
    Plots training & validation loss and accuracy curves over epochs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1e1e24")

    # 1. Loss Subplot
    ax1.set_facecolor("#2b2b36")
    ax1.plot(epochs, history["train_loss"], color="#00e5ff", lw=2, marker="o", ms=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], color="#ff5252", lw=2, marker="s", ms=4, label="Val Loss")
    ax1.set_title("Binary Cross-Entropy Loss", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white", fontsize=10)
    ax1.set_ylabel("Loss", color="white", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3, color="#888888")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#555555")
    ax1.legend(facecolor="#1e1e24", edgecolor="#555555", labelcolor="white")

    # 2. Accuracy Subplot
    ax2.set_facecolor("#2b2b36")
    train_acc_pct = [acc * 100 for acc in history["train_acc"]]
    val_acc_pct = [acc * 100 for acc in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, color="#00e5ff", lw=2, marker="o", ms=4, label="Train Accuracy")
    ax2.plot(epochs, val_acc_pct, color="#69f0ae", lw=2, marker="s", ms=4, label="Val Accuracy")
    ax2.set_title("Classification Accuracy (%)", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", color="white", fontsize=10)
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle="--", alpha=0.3, color="#888888")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#555555")
    ax2.legend(facecolor="#1e1e24", edgecolor="#555555", labelcolor="white")

    plt.suptitle("Lightweight CNN Fault Detector - Training Progress", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[CNN] Training curves saved successfully to: {output_path}")


def train_cnn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_save_path: str = "models/cnn_detector.pth",
    plot_save_path: str = "outputs/training_curves.png",
    num_epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 8,
    seed: int = 42,
) -> Tuple[LightweightFaultCNN, Dict[str, float]]:
    """
    Trains and checkpoints the lightweight CNN model on surface images.

    Returns
    -------
    best_model : LightweightFaultCNN
        The best model based on validation loss.
    summary_metrics : Dict[str, float]
        Dictionary of final training metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    # Prepare datasets & dataloaders
    x_tr_tensor, y_tr_tensor = prepare_tensors(X_train, y_train)
    x_va_tensor, y_va_tensor = prepare_tensors(X_val, y_val)

    train_dataset = TensorDataset(x_tr_tensor, y_tr_tensor)
    val_dataset = TensorDataset(x_va_tensor, y_va_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = LightweightFaultCNN(in_channels=1, dropout_rate=0.4).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_model_weights = None
    best_epoch = 0
    patience_counter = 0

    print(f"\n[CNN] Training LightweightFaultCNN on {len(X_train)} samples ({num_epochs} max epochs, batch={batch_size})...")
    start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                running_val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        # Checkpoint if validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved_tag = "*"
        else:
            patience_counter += 1
            improved_tag = ""

        if epoch % 5 == 0 or epoch == 1 or improved_tag == "*":
            print(
                f"  Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.1f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.1f}% {improved_tag}"
            )

        # Early Stopping
        if patience_counter >= patience:
            print(f"[CNN] Early stopping triggered at epoch {epoch} (no val loss improvement for {patience} epochs).")
            break

    total_training_time = time.perf_counter() - start_time

    # Load best weights into model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save model weights & architecture metadata
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "val_acc": history["val_acc"][best_epoch - 1],
            "train_acc": history["train_acc"][best_epoch - 1],
            "history": history,
        },
        model_save_path,
    )
    print(f"[CNN] Best model weights saved to: {model_save_path} (from epoch {best_epoch})")

    # Plot training curves
    plot_training_curves(history, plot_save_path)

    final_metrics = {
        "training_time_sec": total_training_time,
        "best_epoch": float(best_epoch),
        "train_accuracy": history["train_acc"][best_epoch - 1],
        "val_accuracy": history["val_acc"][best_epoch - 1],
        "best_val_loss": best_val_loss,
    }

    print(f"[CNN] Training finished in {total_training_time:.2f}s")
    print(f"      Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"      Train Acc: {final_metrics['train_accuracy']*100:.2f}% | Val Acc: {final_metrics['val_accuracy']*100:.2f}%")

    return model, final_metrics
