"""
Industrial Fault Detection - Preprocessing Pipeline
===================================================
Modular and extensible OpenCV/NumPy preprocessing module for industrial
surface defect inspection.

Features:
- Image Resizing (interpolation-aware)
- Color Space Management (Grayscale or RGB/BGR retention)
- Multi-mode Noise Filtering (Gaussian, Bilateral, Median)
- Contrast Enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Intensity Normalization (Min-Max [0, 1], Standard Score / Z-Score)
- Dual output: ML-ready float32 tensors and disk-serializable uint8 images
"""

import os
from typing import Optional, Tuple, Union, List
import cv2
import numpy as np


class ImagePreprocessor:
    """
    Extensible image preprocessing pipeline for industrial defect detection.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        grayscale: bool = True,
        noise_filter: str = "gaussian",
        filter_kernel_size: int = 3,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        normalization_method: str = "minmax",
    ):
        """
        Parameters
        ----------
        target_size : Tuple[int, int]
            Target dimensions (width, height) for image resizing. Default is (224, 224).
        grayscale : bool
            Whether to convert images to single-channel grayscale. Default is True.
        noise_filter : str
            Type of noise filter: 'gaussian', 'bilateral', 'median', or 'none'.
        filter_kernel_size : int
            Kernel size for Gaussian blur or Median blur (must be positive odd integer).
        bilateral_d : int
            Diameter of each pixel neighborhood used during bilateral filtering.
        bilateral_sigma_color : float
            Filter sigma in the color space for bilateral filtering.
        bilateral_sigma_space : float
            Filter sigma in the coordinate space for bilateral filtering.
        use_clahe : bool
            Whether to apply Contrast Limited Adaptive Histogram Equalization.
        clahe_clip_limit : float
            Threshold for contrast limiting in CLAHE. Default is 2.0.
        clahe_tile_grid_size : Tuple[int, int]
            Grid size for CLAHE histogram equalization tiles. Default is (8, 8).
        normalization_method : str
            Intensity scaling method: 'minmax' ([0, 1]), 'zscore' ((x - mu) / sigma), or 'none'.
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.noise_filter = noise_filter.lower() if noise_filter else "none"
        self.filter_kernel_size = filter_kernel_size if filter_kernel_size % 2 == 1 else filter_kernel_size + 1
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.normalization_method = normalization_method.lower() if normalization_method else "none"

        # Initialize CLAHE operator
        if self.use_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )
        else:
            self._clahe = None

    def read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Loads an image from file path or validates an existing NumPy array.
        """
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image file does not exist: {image_or_path}")
            # Read image using OpenCV
            img = cv2.imread(image_or_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {image_or_path}")
        elif isinstance(image_or_path, np.ndarray):
            img = image_or_path.copy()
        else:
            raise TypeError(f"Expected file path or numpy.ndarray, got {type(image_or_path)}")

        if img.size == 0:
            raise ValueError("Provided image array is empty.")

        return img

    def convert_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale or standard 3-channel RGB/BGR.
        """
        is_color = len(image.shape) == 3 and image.shape[2] in (3, 4)

        if self.grayscale:
            if is_color:
                if image.shape[2] == 4:
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        else:
            if not is_color:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the image to self.target_size (width, height) using area or linear interpolation.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        if (w, h) == (target_w, target_h):
            return image

        # Use INTER_AREA when downsampling, INTER_CUBIC / INTER_LINEAR when upsampling
        if target_w < w or target_h < h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    def apply_noise_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the configured noise filter.
        """
        if self.noise_filter == "none":
            return image
        elif self.noise_filter == "gaussian":
            k = self.filter_kernel_size
            return cv2.GaussianBlur(image, (k, k), sigmaX=0)
        elif self.noise_filter == "bilateral":
            # Bilateral filter preserves sharp defect edges while smoothing texture noise
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.bilateralFilter(
                image,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
        elif self.noise_filter == "median":
            k = self.filter_kernel_size
            return cv2.medianBlur(image, k)
        else:
            raise ValueError(f"Unsupported noise filter: '{self.noise_filter}'")

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Enhances subtle micro-cracks, scratches, and local surface anomalies.
        """
        if not self.use_clahe or self._clahe is None:
            return image

        # Ensure image is in uint8 range for OpenCV CLAHE
        if image.dtype != np.uint8:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image

        if len(img_uint8.shape) == 2 or (len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1):
            # 2D Grayscale
            enhanced = self._clahe.apply(img_uint8)
            return enhanced
        else:
            # Color image: Apply CLAHE to L-channel in LAB color space
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = self._clahe.apply(l)
            merged = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            return enhanced

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes pixel intensities to [0, 1] or via standard Z-score ((x - mu) / sigma).
        Returns float32 array.
        """
        img_float = image.astype(np.float32)

        if self.normalization_method == "minmax":
            # Scale to [0.0, 1.0]
            # Standard image range is 0 to 255
            return img_float / 255.0
        elif self.normalization_method == "zscore":
            mean = np.mean(img_float)
            std = np.std(img_float)
            if std < 1e-7:
                return img_float - mean
            return (img_float - mean) / std
        elif self.normalization_method == "none":
            return img_float
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

    def process_pipeline(
        self,
        image_or_path: Union[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the complete preprocessing pipeline on a single image.

        Returns
        -------
        preprocessed_uint8 : np.ndarray
            Enhanced uint8 image (ideal for saving to disk or visualization).
        normalized_float : np.ndarray
            Normalized float32 feature tensor (ideal for ML/CNN consumption).
        """
        # 1. Load image
        img = self.read_image(image_or_path)

        # 2. Resize
        resized = self.resize_image(img)

        # 3. Color conversion
        converted = self.convert_color_space(resized)

        # 4. Noise filtering
        denoised = self.apply_noise_filter(converted)

        # 5. CLAHE contrast enhancement
        enhanced = self.apply_clahe(denoised)

        # Ensure enhanced is uint8
        if enhanced.dtype != np.uint8:
            uint8_version = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            uint8_version = enhanced

        # 6. Normalization
        norm_tensor = self.normalize(uint8_version)

        # Add channel dimension if single-channel grayscale (H, W) -> (H, W, 1)
        if len(norm_tensor.shape) == 2:
            norm_tensor = np.expand_dims(norm_tensor, axis=-1)

        return uint8_version, norm_tensor

    def preprocess(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Convenience wrapper returning the normalized float32 feature array.
        """
        _, norm_tensor = self.process_pipeline(image_or_path)
        return norm_tensor

    def preprocess_batch(
        self,
        images_or_paths: List[Union[str, np.ndarray]],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Processes a batch of images.

        Returns
        -------
        uint8_images : List[np.ndarray]
            List of preprocessed uint8 images.
        feature_batch : np.ndarray
            Stacked NumPy float32 array of shape (N, H, W, C).
        """
        uint8_list = []
        norm_list = []

        for item in images_or_paths:
            u8, norm = self.process_pipeline(item)
            uint8_list.append(u8)
            norm_list.append(norm)

        stacked_features = np.stack(norm_list, axis=0)
        return uint8_list, stacked_features


def save_processed_image(image: np.ndarray, output_path: str) -> None:
    """
    Saves preprocessed uint8 image to disk. Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to write image to {output_path}")
