"""
Industrial Fault Detection - Model Evaluation & Diagnostic Visualizations
==========================================================================
Evaluates the baseline SVM and lightweight CNN models on held-out test data.

Computes:
- Accuracy, Precision, Recall, F1-Score, Specificity
- ROC-AUC and Precision-Recall AUC (Average Precision)
- Confusion Matrix (TP, FP, TN, FN)
- Comparative CSV summary export
- Diagnostic plots: Normalized Confusion Matrices, ROC Curves, and PR Curves
"""

import os
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.dataset import load_processed_tensors
from src.feature_extraction import FeatureExtractor, load_scaler
from src.train_svm import load_baseline_svm
from src.train_cnn import LightweightFaultCNN, prepare_tensors


def evaluate_svm_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    svm_path: str = "models/baseline_svm.joblib",
    scaler_path: str = "models/scaler.joblib",
    saved_features_path: str = "models/X_test_features.npy",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluates baseline SVM model on test set.
    """
    svm = load_baseline_svm(svm_path)
    scaler = load_scaler(scaler_path)

    # Use cached standardized test features if available, else extract
    if os.path.exists(saved_features_path):
        X_test_scaled = np.load(saved_features_path)
    else:
        extractor = FeatureExtractor()
        feats = extractor.extract_dataset_features(X_test, desc="Test Features (SVM)")
        X_test_scaled = scaler.transform(feats)

    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc(*roc_curve(y_test, y_prob)[:2]),
        "pr_auc": average_precision_score(y_test, y_prob),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return y_pred, y_prob, metrics


def evaluate_cnn_model(
    X_test: np.ndarray,
    y_test: np.ndarray,
    cnn_path: str = "models/cnn_detector.pth",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluates lightweight CNN model on test set.
    """
    device = torch.device("cpu")
    model = LightweightFaultCNN(in_channels=1).to(device)

    checkpoint = torch.load(cnn_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    X_test_t, _ = prepare_tensors(X_test, y_test)
    X_test_t = X_test_t.to(device)

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": auc(*roc_curve(y_test, probs)[:2]),
        "pr_auc": average_precision_score(y_test, probs),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return preds, probs, metrics


def export_evaluation_summary_csv(
    svm_metrics: Dict[str, float],
    cnn_metrics: Dict[str, float],
    output_path: str = "outputs/evaluation_summary.csv",
) -> pd.DataFrame:
    """
    Exports a structured side-by-side comparison table of test metrics to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = [
        ("Test Accuracy", f"{svm_metrics['accuracy'] * 100:.2f}%", f"{cnn_metrics['accuracy'] * 100:.2f}%", f"{(cnn_metrics['accuracy'] - svm_metrics['accuracy']) * 100:+.2f}%"),
        ("Precision", f"{svm_metrics['precision'] * 100:.2f}%", f"{cnn_metrics['precision'] * 100:.2f}%", f"{(cnn_metrics['precision'] - svm_metrics['precision']) * 100:+.2f}%"),
        ("Recall (Sensitivity)", f"{svm_metrics['recall'] * 100:.2f}%", f"{cnn_metrics['recall'] * 100:.2f}%", f"{(cnn_metrics['recall'] - svm_metrics['recall']) * 100:+.2f}%"),
        ("F1-Score", f"{svm_metrics['f1_score']:.4f}", f"{cnn_metrics['f1_score']:.4f}", f"{(cnn_metrics['f1_score'] - svm_metrics['f1_score']):+.4f}"),
        ("ROC-AUC", f"{svm_metrics['roc_auc']:.4f}", f"{cnn_metrics['roc_auc']:.4f}", f"{(cnn_metrics['roc_auc'] - svm_metrics['roc_auc']):+.4f}"),
        ("PR-AUC (Avg Precision)", f"{svm_metrics['pr_auc']:.4f}", f"{cnn_metrics['pr_auc']:.4f}", f"{(cnn_metrics['pr_auc'] - svm_metrics['pr_auc']):+.4f}"),
        ("True Positives (TP)", str(svm_metrics['tp']), str(cnn_metrics['tp']), f"{cnn_metrics['tp'] - svm_metrics['tp']:+d}"),
        ("False Positives (FP)", str(svm_metrics['fp']), str(cnn_metrics['fp']), f"{cnn_metrics['fp'] - svm_metrics['fp']:+d}"),
        ("True Negatives (TN)", str(svm_metrics['tn']), str(cnn_metrics['tn']), f"{cnn_metrics['tn'] - svm_metrics['tn']:+d}"),
        ("False Negatives (FN)", str(svm_metrics['fn']), str(cnn_metrics['fn']), f"{cnn_metrics['fn'] - svm_metrics['fn']:+d}"),
    ]

    df = pd.DataFrame(rows, columns=["Evaluation Metric", "Baseline SVM (RBF)", "Lightweight CNN", "CNN Advantage"])
    df.to_csv(output_path, index=False)
    print(f"[EVALUATION] Summary comparison saved to: {output_path}")
    return df


def plot_side_by_side_confusion_matrices(
    y_test: np.ndarray,
    y_pred_svm: np.ndarray,
    y_pred_cnn: np.ndarray,
    output_path: str = "outputs/confusion_matrices.png",
):
    """
    Plots normalized side-by-side Confusion Matrices for both models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    cm_svm_norm = cm_svm.astype(float) / cm_svm.sum(axis=1)[:, np.newaxis]
    cm_cnn_norm = cm_cnn.astype(float) / cm_cnn.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#1e1e24")

    labels = ["Normal", "Defective"]

    # Subplot 1: SVM
    ax1.set_facecolor("#2b2b36")
    sns.heatmap(
        cm_svm_norm,
        annot=False,
        cmap="Blues",
        cbar=True,
        ax=ax1,
        vmin=0.0,
        vmax=1.0,
        linewidths=1.5,
        linecolor="#1e1e24",
    )
    for i in range(2):
        for j in range(2):
            raw_c = cm_svm[i, j]
            pct = cm_svm_norm[i, j] * 100
            txt_color = "white" if pct > 50 else "#333333"
            ax1.text(j + 0.5, i + 0.5, f"{raw_c}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=12, fontweight="bold")

    ax1.set_title("Baseline SVM Confusion Matrix", color="white", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax1.set_ylabel("True Label", color="white", fontsize=11)
    ax1.set_xticklabels(labels, color="white", fontsize=10)
    ax1.set_yticklabels(labels, color="white", fontsize=10, rotation=0)

    # Subplot 2: CNN
    ax2.set_facecolor("#2b2b36")
    sns.heatmap(
        cm_cnn_norm,
        annot=False,
        cmap="Greens",
        cbar=True,
        ax=ax2,
        vmin=0.0,
        vmax=1.0,
        linewidths=1.5,
        linecolor="#1e1e24",
    )
    for i in range(2):
        for j in range(2):
            raw_c = cm_cnn[i, j]
            pct = cm_cnn_norm[i, j] * 100
            txt_color = "white" if pct > 50 else "#333333"
            ax2.text(j + 0.5, i + 0.5, f"{raw_c}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=12, fontweight="bold")

    ax2.set_title("Lightweight CNN Confusion Matrix", color="#69f0ae", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax2.set_ylabel("True Label", color="white", fontsize=11)
    ax2.set_xticklabels(labels, color="white", fontsize=10)
    ax2.set_yticklabels(labels, color="white", fontsize=10, rotation=0)

    plt.suptitle("Test Set Confusion Matrix Comparison (N = 30)", color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[DIAGNOSTICS] Confusion matrix plot saved: {output_path}")


def plot_roc_and_pr_curves(
    y_test: np.ndarray,
    y_prob_svm: np.ndarray,
    y_prob_cnn: np.ndarray,
    output_path: str = "outputs/roc_pr_curves.png",
):
    """
    Plots ROC Curves and Precision-Recall Curves side-by-side.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Compute ROC curves
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = auc(fpr_svm, tpr_svm)

    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_prob_cnn)
    auc_cnn = auc(fpr_cnn, tpr_cnn)

    # Compute PR curves
    prec_svm, rec_svm, _ = precision_recall_curve(y_test, y_prob_svm)
    ap_svm = average_precision_score(y_test, y_prob_svm)

    prec_cnn, rec_cnn, _ = precision_recall_curve(y_test, y_prob_cnn)
    ap_cnn = average_precision_score(y_test, y_prob_cnn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#1e1e24")

    # 1. ROC Curves
    ax1.set_facecolor("#2b2b36")
    ax1.plot(fpr_svm, tpr_svm, color="#00e5ff", lw=2.5, label=f"Baseline SVM (AUC = {auc_svm:.3f})")
    ax1.plot(fpr_cnn, tpr_cnn, color="#69f0ae", lw=2.5, label=f"Lightweight CNN (AUC = {auc_cnn:.3f})")
    ax1.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax1.set_title("Receiver Operating Characteristic (ROC)", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", color="white", fontsize=10)
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", color="white", fontsize=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.3, color="#888888")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#555555")
    ax1.legend(loc="lower right", facecolor="#1e1e24", edgecolor="#555555", labelcolor="white")

    # 2. PR Curves
    ax2.set_facecolor("#2b2b36")
    ax2.plot(rec_svm, prec_svm, color="#00e5ff", lw=2.5, label=f"Baseline SVM (AP = {ap_svm:.3f})")
    ax2.plot(rec_cnn, prec_cnn, color="#69f0ae", lw=2.5, label=f"Lightweight CNN (AP = {ap_cnn:.3f})")
    prevalence = np.mean(y_test)
    ax2.axhline(prevalence, color="#888888", linestyle="--", lw=1.5, label=f"Baseline Prevalence ({prevalence:.2f})")
    ax2.set_title("Precision-Recall (PR) Curve", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall (Sensitivity)", color="white", fontsize=10)
    ax2.set_ylabel("Precision (PPV)", color="white", fontsize=10)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0.4, 1.05])
    ax2.grid(True, linestyle="--", alpha=0.3, color="#888888")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#555555")
    ax2.legend(loc="lower left", facecolor="#1e1e24", edgecolor="#555555", labelcolor="white")

    plt.suptitle("Discriminative Diagnostic Curves (SVM vs CNN)", color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[DIAGNOSTICS] ROC & PR curves saved: {output_path}")
