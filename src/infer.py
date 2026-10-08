"""
Industrial Fault Detection - Standalone Inference & Anomaly Localization Engine
================================================================================
Performs single-image inference using either the baseline SVM or deep CNN model.
Overlays predicted class, confidence score, and detected defect bounding boxes.
"""

import os
from typing import Dict, Tuple, List, Union, Optional
import cv2
import numpy as np
import torch

from src.preprocess import ImagePreprocessor
from src.feature_extraction import FeatureExtractor, load_scaler
from src.train_svm import load_baseline_svm
from src.train_cnn import LightweightFaultCNN, prepare_tensors


class FaultInferenceEngine:
    """
    Unified inference pipeline for industrial surface fault detection.
    """

    def __init__(
        self,
        model_type: str = "cnn",
        svm_path: str = "models/baseline_svm.joblib",
        scaler_path: str = "models/scaler.joblib",
        cnn_path: str = "models/cnn_detector.pth",
        min_defect_area: float = 12.0,
    ):
        self.model_type = model_type.lower()
        self.min_defect_area = min_defect_area
        self.preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            grayscale=True,
            noise_filter="gaussian",
            filter_kernel_size=3,
            use_clahe=True,
            clahe_clip_limit=2.5,
            clahe_tile_grid_size=(8, 8),
            normalization_method="minmax",
        )

        if self.model_type == "svm":
            print(f"[INFERENCE] Loading Baseline SVM ({svm_path}) & Scaler ({scaler_path})...")
            self.svm = load_baseline_svm(svm_path)
            self.scaler = load_scaler(scaler_path)
            self.feature_extractor = FeatureExtractor()
            self.cnn = None
        elif self.model_type == "cnn":
            print(f"[INFERENCE] Loading Lightweight CNN Classifier ({cnn_path})...")
            self.device = torch.device("cpu")
            self.cnn = LightweightFaultCNN(in_channels=1).to(self.device)
            ckpt = torch.load(cnn_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.cnn.load_state_dict(state_dict)
            self.cnn.eval()
            self.svm = None
            self.scaler = None
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'svm' or 'cnn'.")

    def detect_anomaly_boxes(self, uint8_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects significant potential defect regions via multi-scale morphological filtering
        and background deviation. Captures scratches, cracks, pitting, and large patches.
        Returns list of bounding boxes (x, y, w, h).
        """
        # 1. Black-hat filter for fine grooves, cracks, and pitting
        kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(uint8_image, cv2.MORPH_BLACKHAT, kernel_bh)
        _, thresh_bh = cv2.threshold(blackhat, 24, 255, cv2.THRESH_BINARY)

        # 2. Background deviation for larger oxidized patches / inclusions
        bg = cv2.GaussianBlur(uint8_image, (35, 35), 0)
        diff = cv2.subtract(bg, uint8_image)
        _, thresh_diff = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)

        # Combine defect response maps
        combined = cv2.bitwise_or(thresh_bh, thresh_diff)

        # Morphological closing to connect disjoint crack/scratch segments
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh_closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_defect_area:
                x, y, w, h = cv2.boundingRect(c)
                pad = 4
                x_p = max(0, x - pad)
                y_p = max(0, y - pad)
                w_p = min(uint8_image.shape[1] - x_p, w + 2 * pad)
                h_p = min(uint8_image.shape[0] - y_p, h + 2 * pad)
                boxes.append((x_p, y_p, w_p, h_p))

        # Sort boxes by area descending and keep top significant regions
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:8]
        return boxes

    def predict(self, image_or_path: Union[str, np.ndarray]) -> Dict:
        """
        Runs complete inference and visualization on a single image.
        """
        # Phase 1 Preprocessing
        uint8_proc, norm_tensor = self.preprocessor.process_pipeline(image_or_path)

        # Predict probability
        if self.model_type == "cnn":
            tensor_x = torch.from_numpy(np.expand_dims(np.transpose(norm_tensor, (2, 0, 1)), axis=0).astype(np.float32))
            with torch.no_grad():
                logit = self.cnn(tensor_x)
                prob_defective = float(torch.sigmoid(logit).item())
        else:
            feats = self.feature_extractor.extract_image_features(uint8_proc)
            feats_scaled = self.scaler.transform(feats.reshape(1, -1))
            prob_defective = float(self.svm.predict_proba(feats_scaled)[0, 1])

        # Classification decision
        is_defective = prob_defective >= 0.5
        label = "Defective" if is_defective else "Normal"
        confidence = prob_defective if is_defective else (1.0 - prob_defective)

        # Detect flaws only if classified as defective
        boxes = []
        if is_defective:
            boxes = self.detect_anomaly_boxes(uint8_proc)

        # Build annotated visualization
        annotated_bgr = cv2.cvtColor(uint8_proc, cv2.COLOR_GRAY2BGR)

        # Draw defect bounding boxes cleanly
        for i, (bx, by, bw, bh) in enumerate(boxes):
            cv2.rectangle(annotated_bgr, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            if i < 4:  # Only badge top 4 largest regions to avoid clutter
                cv2.putText(
                    annotated_bgr,
                    f"FLAW #{i+1}",
                    (bx, max(14, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Top banner overlay
        banner_h = 32
        banner = np.zeros((banner_h, annotated_bgr.shape[1], 3), dtype=np.uint8)
        if is_defective:
            banner[:] = (20, 20, 160)  # Dark red banner
            status_text = f"DEFECTIVE ({confidence * 100:.1f}%) | {len(boxes)} Flaws"
            text_color = (255, 255, 255)
        else:
            banner[:] = (20, 130, 40)  # Dark green banner
            status_text = f"NORMAL ({confidence * 100:.1f}%) | Passed"
            text_color = (255, 255, 255)

        cv2.putText(banner, status_text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv2.LINE_AA)

        # Combine banner and image
        annotated_bgr = np.vstack([banner, annotated_bgr])
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return {
            "label": label,
            "class_idx": 1 if is_defective else 0,
            "confidence": confidence,
            "prob_defective": prob_defective,
            "num_defects_found": len(boxes),
            "bounding_boxes": boxes,
            "annotated_bgr": annotated_bgr,
            "annotated_rgb": annotated_rgb,
            "preprocessed_uint8": uint8_proc,
        }
