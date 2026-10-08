"""
Industrial Fault Detection - Handcrafted Feature Extraction Module
===================================================================
Extracts edge, texture, and morphological contour features from preprocessed
surface inspection images for classical ML classification (SVM/Random Forest).

Features Extracted:
1. Edge Features:
   - Canny edge pixel density
   - Mean edge intensity & standard deviation
   - Sobel gradient magnitude mean & standard deviation
2. Texture Features:
   - Gray-Level Co-occurrence Matrix (GLCM) / Haralick metrics:
     Contrast, Dissimilarity, Homogeneity, Energy, Correlation
   - Local Binary Patterns (LBP) uniform histogram
3. Contour Features:
   - Morphological defect contour detection
   - Max contour area, perimeter, circularity (isoperimetric quotient)
   - Total defect-like contour count & area occupancy ratio
"""

import os
from typing import List, Tuple, Union, Optional
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm


FEATURE_NAMES = [
    # Edge features (5)
    "canny_edge_density",
    "canny_edge_mean_intensity",
    "canny_edge_std_intensity",
    "sobel_grad_mean",
    "sobel_grad_std",
    # GLCM / Haralick texture features (10: 5 props x 2 distances)
    "glcm_contrast_d1",
    "glcm_dissimilarity_d1",
    "glcm_homogeneity_d1",
    "glcm_energy_d1",
    "glcm_correlation_d1",
    "glcm_contrast_d3",
    "glcm_dissimilarity_d3",
    "glcm_homogeneity_d3",
    "glcm_energy_d3",
    "glcm_correlation_d3",
    # LBP texture histogram (10 uniform bins for P=8, R=1)
    "lbp_bin_0",
    "lbp_bin_1",
    "lbp_bin_2",
    "lbp_bin_3",
    "lbp_bin_4",
    "lbp_bin_5",
    "lbp_bin_6",
    "lbp_bin_7",
    "lbp_bin_8",
    "lbp_bin_9",
    # Contour / morphological features (6)
    "defect_contour_count",
    "max_contour_area",
    "max_contour_perimeter",
    "max_contour_circularity",
    "total_contour_area_ratio",
    "max_contour_aspect_ratio",
]


class FeatureExtractor:
    """
    Extracts handcrafted computer vision features from surface images.
    """

    def __init__(
        self,
        canny_low: int = 40,
        canny_high: int = 120,
        glcm_levels: int = 32,
        lbp_points: int = 8,
        lbp_radius: int = 1,
        min_contour_area: float = 12.0,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.glcm_levels = glcm_levels
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.min_contour_area = min_contour_area

    def _prepare_gray_uint8(self, img: Union[str, np.ndarray]) -> np.ndarray:
        """Converts input image to 2D uint8 grayscale array [0, 255]."""
        if isinstance(img, str):
            arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise ValueError(f"Could not load image: {img}")
            return arr

        arr = np.array(img)
        # Squeeze channel dimension if (H, W, 1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Scale float [0, 1] to uint8 [0, 255]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    def extract_edge_features(self, gray: np.ndarray) -> List[float]:
        """Calculates Canny edge metrics and Sobel gradient moments."""
        h, w = gray.shape
        total_pixels = float(h * w)

        # 1. Canny edge detector
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edge_mask = edges > 0
        edge_density = float(np.sum(edge_mask)) / total_pixels

        if np.any(edge_mask):
            edge_intensities = gray[edge_mask]
            edge_mean_intensity = float(np.mean(edge_intensities))
            edge_std_intensity = float(np.std(edge_intensities))
        else:
            edge_mean_intensity = 0.0
            edge_std_intensity = 0.0

        # 2. Sobel gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        sobel_mean = float(np.mean(mag))
        sobel_std = float(np.std(mag))

        return [
            edge_density,
            edge_mean_intensity,
            edge_std_intensity,
            sobel_mean,
            sobel_std,
        ]

    def extract_texture_features(self, gray: np.ndarray) -> List[float]:
        """Calculates GLCM Haralick texture properties and LBP histogram."""
        # 1. GLCM features (quantized to self.glcm_levels for stability and speed)
        quantized = (gray.astype(np.float32) / (256.0 / self.glcm_levels)).astype(np.uint8)
        distances = [1, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(
            quantized,
            distances=distances,
            angles=angles,
            levels=self.glcm_levels,
            symmetric=True,
            normed=True,
        )

        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        glcm_feats = []
        for d_idx in range(len(distances)):
            for prop in props:
                val = graycoprops(glcm, prop)[d_idx, :]
                glcm_feats.append(float(np.mean(val)))

        # 2. Local Binary Patterns (LBP) uniform histogram
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_points,
            R=self.lbp_radius,
            method="uniform",
        )
        # Uniform LBP with P=8 has P + 2 = 10 bins
        n_bins = self.lbp_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_feats = [float(b) for b in lbp_hist]

        return glcm_feats + lbp_feats

    def extract_contour_features(self, gray: np.ndarray) -> List[float]:
        """
        Calculates morphological contour statistics:
        Identifies defect regions (scratches, cracks, pitting, patches) via
        Black-Hat morphological filtering and thresholding.
        """
        h, w = gray.shape
        total_area = float(h * w)

        # Black-Hat filter highlights dark structures (grooves, cracks, pits) on brighter background
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold blackhat response
        _, thresh = cv2.threshold(blackhat, 22, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        count = float(len(significant_contours))

        if significant_contours:
            areas = [cv2.contourArea(c) for c in significant_contours]
            max_idx = int(np.argmax(areas))
            max_cnt = significant_contours[max_idx]

            max_area = float(areas[max_idx])
            max_perim = float(cv2.arcLength(max_cnt, closed=True))

            # Circularity: 4 * pi * Area / (Perimeter^2)  [1.0 for perfect circle]
            if max_perim > 1e-5:
                circularity = float(4.0 * np.pi * max_area / (max_perim**2))
            else:
                circularity = 0.0

            total_defect_area = float(sum(areas))
            area_ratio = total_defect_area / total_area

            # Aspect ratio of bounding box
            _, _, bw, bh = cv2.boundingRect(max_cnt)
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
        else:
            max_area = 0.0
            max_perim = 0.0
            circularity = 0.0
            area_ratio = 0.0
            aspect_ratio = 1.0

        return [
            count,
            max_area,
            max_perim,
            circularity,
            area_ratio,
            aspect_ratio,
        ]

    def extract_image_features(self, img: Union[str, np.ndarray]) -> np.ndarray:
        """
        Extracts a single 1D feature vector for an image.
        Length: 31 features (5 edge + 20 texture + 6 contour).
        """
        gray = self._prepare_gray_uint8(img)
        edge_feats = self.extract_edge_features(gray)
        texture_feats = self.extract_texture_features(gray)
        contour_feats = self.extract_contour_features(gray)

        vec = np.array(edge_feats + texture_feats + contour_feats, dtype=np.float64)
        # Replace any potential NaNs with zeros
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def extract_dataset_features(
        self,
        images_or_paths: Union[np.ndarray, List[Union[str, np.ndarray]]],
        desc: str = "Extracting Features",
    ) -> np.ndarray:
        """Extracts feature matrix (N, num_features) across a dataset."""
        features = []
        for img in tqdm(images_or_paths, desc=desc):
            vec = self.extract_image_features(img)
            features.append(vec)
        return np.array(features, dtype=np.float64)


def fit_and_save_scaler(
    X_train_feats: np.ndarray,
    scaler_path: str = "models/scaler.joblib",
) -> StandardScaler:
    """Fits StandardScaler on training features and saves to disk."""
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(X_train_feats)
    joblib.dump(scaler, scaler_path)
    return scaler


def load_scaler(scaler_path: str = "models/scaler.joblib") -> StandardScaler:
    """Loads fitted StandardScaler from disk."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(scaler_path)
