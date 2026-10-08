# Industrial Fault Detection Using Image Processing

An end-to-end, modular, and production-grade computer vision and machine learning system for automated industrial surface fault inspection and anomaly localization (Normal vs. Defective binary classification).

---

## 📌 Project Overview

In industrial manufacturing (metal rolling, sand casting, machining), automated optical inspection (AOI) prevents catastrophic component failure by detecting structural flaws before parts leave the factory line.

This project delivers a complete 3-phase automated inspection pipeline:
1. **Phase 1: Ingestion & Adaptive Preprocessing:** Standardized resizing, single-channel conversion, bilateral/Gaussian smoothing, and Contrast Limited Adaptive Histogram Equalization (**CLAHE**) to bring micro-fissures, scratches, and porous defects into high contrast.
2. **Phase 2: Hybrid Learning Pipeline:** Handcrafted feature engineering ($31$ edge, GLCM, LBP, and morphological contour descriptors) with an **RBF Support Vector Machine (SVM)** baseline paired against an end-to-end **Lightweight 3-Stage Convolutional Neural Network (CNN)**.
3. **Phase 3: Held-out Diagnostics & Anomaly Localization:** Comparative evaluation on unseen test data, ROC/PR analysis, and a standalone single-image CLI inference engine with visual anomaly bounding box overlays.

---

## 📂 Project Architecture

```
aryan project/
├── data/
│   ├── raw/
│   │   ├── normal/               # 100 high-res raw surface images (brushed & sand-cast)
│   │   └── defective/            # 100 high-res raw defect images (cracks, scratches, pits, patches)
│   └── processed/
│       ├── train/ {normal, defective}  # 140 train images (224x224 uint8)
│       ├── val/   {normal, defective}  # 30 validation images (224x224 uint8)
│       ├── test/  {normal, defective}  # 30 test images (224x224 uint8)
│       ├── X_train.npy, y_train.npy    # Preprocessed float32 tensors (140, 224, 224, 1)
│       ├── X_val.npy, y_val.npy        # Validation float32 tensors (30, 224, 224, 1)
│       └── X_test.npy, y_test.npy      # Test float32 tensors (30, 224, 224, 1)
├── models/
│   ├── baseline_svm.joblib       # Persisted RBF Support Vector Classifier
│   ├── scaler.joblib             # Fitted StandardScaler for 31D feature space
│   ├── cnn_detector.pth          # Best checkpointed PyTorch ConvNet weights
│   └── X_*_features.npy          # Standardized 31D feature matrices
├── outputs/
│   ├── confusion_matrices.png    # Side-by-side normalized test confusion matrices
│   ├── roc_pr_curves.png         # ROC Curves and Precision-Recall Curves
│   ├── evaluation_summary.csv    # Benchmark metric comparison table
│   ├── sample_inference_predictions.png # Batch test inference with localized flaw boxes
│   └── training_curves.png       # CNN training/validation loss & accuracy curves
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # OpenCV CLAHE, filtering, resizing & normalization
│   ├── dataset.py                # Stratified splitting & array loaders
│   ├── synthetic_generator.py    # Realistic industrial benchmark synthesis
│   ├── feature_extraction.py     # 31D Handcrafted feature extractor (Edge, GLCM, LBP, Contours)
│   ├── train_svm.py              # Baseline SVM trainer & validation
│   ├── train_cnn.py              # Lightweight 3-stage CNN classifier trainer
│   ├── evaluate.py               # Test evaluation & diagnostic plotting engine
│   └── infer.py                  # Standalone inference & anomaly localization engine
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py        # Unit tests for preprocessing pipeline
│   ├── test_features.py          # Unit tests for feature extraction
│   └── test_inference.py         # Integration tests for inference engine
├── phase1_sample_visualization.png # Phase 1 raw vs preprocessed verification plot
├── main.py                       # Master orchestrator script (Phases 1-3 & tests)
├── run_phase1.py                 # Standalone Phase 1 execution script
├── run_phase2.py                 # Standalone Phase 2 execution script
├── run_phase3.py                 # Standalone Phase 3 execution script
├── predict.py                    # Standalone CLI prediction tool with visual overlay
└── requirements.txt              # Project dependencies
```

---

## 🔬 Methodology & Technical Specifications

### 1. Preprocessing Pipeline (`src/preprocess.py`)
- **Spatial Normalization:** Resized to standardized $224 \times 224$ dimensions via area-downsampling (`INTER_AREA`) and cubic-interpolation (`INTER_CUBIC`).
- **Color Space:** Conversion to single-channel luminance intensity.
- **Denoising:** Multi-mode filtering (Gaussian blur with $\sigma=0$ or edge-preserving bilateral filtering).
- **Contrast Enhancement (CLAHE):** Contrast Limited Adaptive Histogram Equalization with `clipLimit=2.5` and `tileGridSize=(8, 8)` to maximize local gradients around fine cracks and pits while avoiding noise amplification in uniform metal regions.
- **Tensor Scaling:** Intensity normalization scaled to $[0.0, 1.0]$ `float32`.

### 2. Handcrafted Feature Extraction (`src/feature_extraction.py`)
Extracts a comprehensive **31-dimensional descriptor vector**:
- **Edge Descriptors (5):** Canny edge pixel density, mean edge intensity, edge standard deviation, Sobel gradient magnitude mean, and Sobel gradient standard deviation.
- **Texture Descriptors (20):**
  - **Gray-Level Co-occurrence Matrix (GLCM):** Contrast, Dissimilarity, Homogeneity, Energy, and Correlation evaluated at pixel displacements $d \in \{1, 3\}$ across $4$ orientations ($\theta \in \{0, \frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{4}\}$).
  - **Local Binary Patterns (LBP):** 10-bin normalized histogram of uniform patterns ($P=8, R=1$).
- **Morphological Contour Descriptors (6):** Black-Hat morphological filtering and contour detection (`cv2.findContours`) extracting defect-like contour count, maximum contour area, perimeter, circularity (isoperimetric quotient: $4\pi \frac{\text{Area}}{\text{Perimeter}^2}$), total defect area occupancy ratio, and aspect ratio.
- Standardized via `StandardScaler` fitted strictly on the training partition ($0$ NaN / $0$ Inf).

### 3. Model Architectures
- **Baseline SVM (`src/train_svm.py`):** Radial Basis Function (RBF) Support Vector Classifier (`C=1.0`, `gamma='scale'`, `probability=True`).
- **Deep CNN Classifier (`src/train_cnn.py`):** Lightweight 3-stage Convolutional Neural Network:
  $$\text{Input: } (B, 1, 224, 224)$$
  $$\text{Block 1: } \text{Conv2D}(1 \to 32, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \implies (B, 32, 112, 112)$$
  $$\text{Block 2: } \text{Conv2D}(32 \to 64, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \implies (B, 64, 56, 56)$$
  $$\text{Block 3: } \text{Conv2D}(64 \to 128, 3\times3) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times2) \implies (B, 128, 28, 28)$$
  $$\text{Global Average Pooling: } \text{AdaptiveAvgPool2D}(1, 1) \implies (B, 128)$$
  $$\text{Dense Head: } \text{Linear}(128 \to 64) \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(64 \to 1)$$
  $$\text{Loss: Binary Cross-Entropy with Logits, Adam Optimizer } (\text{lr}=10^{-3}, \text{weight decay}=10^{-4})$$

---

## 📊 Final Evaluation Results (Held-Out Test Set: $N = 30$)

| Metric | Baseline SVM (RBF) | Lightweight CNN | Notes |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **90.00%** | **90.00%** | $27/30$ correct classifications |
| **Precision** | **100.00%** | **100.00%** | **Zero False Positives ($FP=0$)** |
| **Recall (Sensitivity)** | **80.00%** | **80.00%** | $12/15$ true defects detected |
| **F1-Score** | **0.8889** | **0.8889** | Optimal precision-recall balance |
| **ROC-AUC** | **1.0000** | **0.9644** | Excellent ranking separation |
| **PR-AUC (Avg Precision)** | **1.0000** | **0.9694** | High precision retention |
| **True Positives (TP)** | 12 | 12 | Equal |
| **False Positives (FP)** | **0** | **0** | **0% False Alarm Rate** |
| **True Negatives (TN)** | 15 | 15 | $100\%$ normal pass-through |
| **False Negatives (FN)** | 3 | 3 | Equal |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Installation
Install all project dependencies into your Python environment:
```bash
pip install -r requirements.txt
```

### 2. Run All Phases (End-to-End Master Pipeline)
Run the entire pipeline (data generation, preprocessing, training both models, and running evaluation):
```bash
python main.py --all
```

Or execute individual phases:
```bash
# Phase 1: Ingestion, synthetic data generation & CLAHE preprocessing
python main.py --phase 1

# Phase 2: Handcrafted feature extraction, SVM & CNN training
python main.py --phase 2

# Phase 3: Test evaluation, diagnostic plots & batch inference
python main.py --phase 3
```

### 3. Run Automated Test Suite
Verify that all unit and integration tests pass ($100\%$ passing):
```bash
pytest -v tests/
# or
python main.py --test
```

### 4. Run Single-Image Inference with Anomaly Highlighting
Inspect any surface image using either the CNN or SVM model:
```bash
# Using Deep CNN (Default)
python predict.py --image data/raw/defective/defective_0001.png --model cnn

# Using Baseline SVM
python predict.py --image data/raw/normal/normal_0001.png --model svm
```

Sample CLI output:
```text
============================================================
        INDUSTRIAL FAULT INSPECTION RESULT
============================================================
  Input Image     : data/raw/defective/defective_0001.png
  Model Engine    : CNN
  Classification  : DEFECTIVE
  Confidence      : 86.30%
  Defect Score    : 86.30%
  Flaws Detected  : 1 region(s)
    - Flaw #1     : x=40, y=129, w=156, h=82
  Annotated Output: outputs/defective_0001_cnn_prediction.png
============================================================
```

---

## 📈 Visual Artifacts

- **Phase 1 Verification (`phase1_sample_visualization.png`):** High-resolution side-by-side comparison of raw vs. CLAHE-enhanced surfaces with difference heatmaps and intensity histograms.
- **Phase 2 Learning Curves (`outputs/training_curves.png`):** Loss and accuracy trajectories over epochs with early stopping checkpoint at epoch 23.
- **Phase 3 Confusion Matrices (`outputs/confusion_matrices.png`):** Normalized confusion matrix heatmaps showing zero false alarms.
- **Phase 3 ROC & PR Curves (`outputs/roc_pr_curves.png`):** Discriminative diagnostic curves demonstrating AUC $> 0.96$.
- **Phase 3 Batch Inspection (`outputs/sample_inference_predictions.png`):** Multi-panel visual breakdown with bounding boxes around localized defects.
