"""
Industrial Fault Detection - Baseline Support Vector Machine (SVM) Training
===========================================================================
Trains a radial basis function (RBF) Support Vector Classifier on standardized
handcrafted features (edges, GLCM/LBP texture, morphological contours).
"""

import os
import time
from typing import Dict, Tuple, Optional
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_baseline_svm(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    model_save_path: str = "models/baseline_svm.joblib",
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    random_state: int = 42,
) -> Tuple[SVC, Dict[str, float]]:
    """
    Trains and validates an RBF SVM classifier on handcrafted features.

    Returns
    -------
    model : SVC
        Trained SVM model.
    metrics : Dict[str, float]
        Dictionary of performance metrics (train_acc, val_acc, precision, recall, f1, runtime).
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"\n[SVM] Initializing Support Vector Classifier (kernel='{kernel}', C={C}, gamma='{gamma}')...")
    svm_model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,
        random_state=random_state,
    )

    # Measure training time
    start_time = time.perf_counter()
    svm_model.fit(X_train_scaled, y_train)
    training_time = time.perf_counter() - start_time

    # Predictions
    y_train_pred = svm_model.predict(X_train_scaled)
    y_val_pred = svm_model.predict(X_val_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

    # Save trained model to disk
    joblib.dump(svm_model, model_save_path)
    print(f"[SVM] Model successfully saved to: {model_save_path}")

    metrics = {
        "training_time_sec": training_time,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "val_f1": val_f1,
    }

    print(f"[SVM] Training completed in {training_time:.3f}s")
    print(f"      Train Accuracy : {train_acc * 100:.2f}%")
    print(f"      Val Accuracy   : {val_acc * 100:.2f}%")
    print(f"      Val Precision  : {val_prec * 100:.2f}%")
    print(f"      Val Recall     : {val_rec * 100:.2f}%")
    print(f"      Val F1 Score   : {val_f1:.4f}")

    return svm_model, metrics


def load_baseline_svm(model_path: str = "models/baseline_svm.joblib") -> SVC:
    """Loads trained SVM model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
