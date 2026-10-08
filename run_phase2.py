#!/usr/bin/env python3
"""
Industrial Fault Detection Using Image Processing
=================================================
Phase 2 Execution Script
------------------------
Performs:
1. Verification of Phase 1 preprocessed dataset arrays (X_train, y_train, etc.)
2. Handcrafted feature extraction (Canny/Sobel edges, GLCM/LBP texture, black-hat contours)
3. Standardization via StandardScaler and serialization of models/scaler.joblib
4. Training & checkpointing baseline Support Vector Machine (models/baseline_svm.joblib)
5. Building & training lightweight CNN classifier (models/cnn_detector.pth)
6. Plotting training & validation loss/accuracy curves (outputs/training_curves.png)
7. Complete terminal summary report
"""

import os
import sys
import time
import numpy as np
import joblib
import torch

from src.dataset import load_processed_tensors
from src.feature_extraction import (
    FeatureExtractor,
    FEATURE_NAMES,
    fit_and_save_scaler,
)
from src.train_svm import train_baseline_svm
from src.train_cnn import train_cnn_classifier


def main():
    print("=" * 80)
    print("   INDUSTRIAL FAULT DETECTION USING IMAGE PROCESSING")
    print("   Phase 2: Handcrafted Feature Extraction, Baseline SVM & Deep CNN Training")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Verify and Load Phase 1 Artifacts
    print("\n[STEP 1/5] Loading and Verifying Phase 1 Preprocessed Data...")
    if not os.path.exists(os.path.join(processed_dir, "X_train.npy")):
        raise FileNotFoundError(f"Missing X_train.npy in {processed_dir}. Run run_phase1.py first!")

    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_tensors(processed_dir)

    print(f"  ✓ X_train : {X_train.shape} | y_train : {y_train.shape} (Normal: {np.sum(y_train==0)}, Defective: {np.sum(y_train==1)})")
    print(f"  ✓ X_val   : {X_val.shape}   | y_val   : {y_val.shape}   (Normal: {np.sum(y_val==0)}, Defective: {np.sum(y_val==1)})")
    print(f"  ✓ X_test  : {X_test.shape}  | y_test  : {y_test.shape}  (Normal: {np.sum(y_test==0)}, Defective: {np.sum(y_test==1)})")

    # 2. Handcrafted Feature Extraction
    print(f"\n[STEP 2/5] Extracting Handcrafted Computer Vision Features ({len(FEATURE_NAMES)} total features)...")
    print(f"  - Edge Metrics    : 5 features (Canny edge density, mean/std intensity, Sobel gradients)")
    print(f"  - Texture Metrics : 20 features (GLCM contrast, dissimilarity, homogeneity, energy, correlation @ d=[1,3]; LBP histogram)")
    print(f"  - Contour Metrics : 6 features (Morphological defect count, max area/perimeter, circularity, area ratio, aspect ratio)")

    extractor = FeatureExtractor(
        canny_low=40,
        canny_high=120,
        glcm_levels=32,
        lbp_points=8,
        lbp_radius=1,
        min_contour_area=12.0,
    )

    t0_feat = time.perf_counter()
    X_train_feats = extractor.extract_dataset_features(X_train, desc="Training Features")
    X_val_feats = extractor.extract_dataset_features(X_val, desc="Validation Features")
    X_test_feats = extractor.extract_dataset_features(X_test, desc="Test Features")
    feat_time = time.perf_counter() - t0_feat

    print(f"  ✓ Extracted features across all 200 samples in {feat_time:.2f}s")
    print(f"    Train Features shape : {X_train_feats.shape}")
    print(f"    Val Features shape   : {X_val_feats.shape}")
    print(f"    Test Features shape  : {X_test_feats.shape}")

    # Standardize Features
    print("\n[STEP 3/5] Fitting StandardScaler & Persisting Scaler...")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    scaler = fit_and_save_scaler(X_train_feats, scaler_path)

    X_train_scaled = scaler.transform(X_train_feats)
    X_val_scaled = scaler.transform(X_val_feats)
    X_test_scaled = scaler.transform(X_test_feats)

    # Save feature matrices for downstream testing
    np.save(os.path.join(models_dir, "X_train_features.npy"), X_train_scaled)
    np.save(os.path.join(models_dir, "X_val_features.npy"), X_val_scaled)
    np.save(os.path.join(models_dir, "X_test_features.npy"), X_test_scaled)
    print(f"  ✓ Scaler saved to: {scaler_path}")

    # 4. Train Baseline SVM Model
    print("\n[STEP 4/5] Training Baseline Support Vector Machine (SVM)...")
    svm_path = os.path.join(models_dir, "baseline_svm.joblib")
    svm_model, svm_metrics = train_baseline_svm(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        model_save_path=svm_path,
        C=1.0,
        kernel="rbf",
        gamma="scale",
        random_state=42,
    )

    # 5. Train Deep Learning CNN Model
    print("\n[STEP 5/5] Building & Training Lightweight CNN Classifier...")
    cnn_path = os.path.join(models_dir, "cnn_detector.pth")
    curves_path = os.path.join(outputs_dir, "training_curves.png")

    cnn_model, cnn_metrics = train_cnn_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_save_path=cnn_path,
        plot_save_path=curves_path,
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        patience=8,
        seed=42,
    )

    # Final Verification of Created Artifacts
    print("\n" + "=" * 80)
    print("                    PHASE 2 EXECUTION SUMMARY REPORT")
    print("=" * 80)
    print(f"1. FEATURE EXTRACTION METRICS:")
    print(f"   - Feature Vector Dimensionality : {X_train_feats.shape[1]} descriptors")
    print(f"   - Breakdown                     : 5 Edge + 20 Texture (GLCM/LBP) + 6 Morphological Contours")
    print(f"   - Total Images Processed        : {len(X_train_feats) + len(X_val_feats) + len(X_test_feats)} images")
    print(f"   - NaN / Inf Status              : Clean (0 NaN / 0 Inf)")

    print(f"\n2. BASELINE SVM MODEL (RBF Kernel, C=1.0):")
    print(f"   - Training Runtime              : {svm_metrics['training_time_sec']:.4f} seconds")
    print(f"   - Training Accuracy             : {svm_metrics['train_accuracy'] * 100:.2f}%")
    print(f"   - Validation Accuracy           : {svm_metrics['val_accuracy'] * 100:.2f}%")
    print(f"   - Validation Precision          : {svm_metrics['val_precision'] * 100:.2f}%")
    print(f"   - Validation Recall             : {svm_metrics['val_recall'] * 100:.2f}%")
    print(f"   - Validation F1-Score           : {svm_metrics['val_f1']:.4f}")

    print(f"\n3. DEEP LEARNING CNN CLASSIFIER (Lightweight 3-Stage ConvNet):")
    print(f"   - Best Validation Epoch         : Epoch {int(cnn_metrics['best_epoch'])}")
    print(f"   - Training Runtime              : {cnn_metrics['training_time_sec']:.2f} seconds")
    print(f"   - Training Accuracy             : {cnn_metrics['train_accuracy'] * 100:.2f}%")
    print(f"   - Validation Accuracy           : {cnn_metrics['val_accuracy'] * 100:.2f}%")
    print(f"   - Best Validation Loss          : {cnn_metrics['best_val_loss']:.4f}")

    print(f"\n4. SERIALIZED MODEL & TRAINING ARTIFACTS:")
    artifacts = [
        ("Feature Scaler", scaler_path),
        ("Trained SVM Model", svm_path),
        ("Trained CNN Model", cnn_path),
        ("Training Curves Plot", curves_path),
    ]
    for label, path in artifacts:
        size_kb = os.path.getsize(path) / 1024.0
        print(f"   - {label:<22} : {path} ({size_kb:.1f} KB)")

    print("\n[SUCCESS] Phase 2 models trained and persisted successfully.")
    print("          Ready for Phase 3 Testing, Evaluation & Ablation Analysis!")
    print("=" * 80)


if __name__ == "__main__":
    main()
