"""
Industrial Fault Detection - Dataset Ingestion & Processing Pipeline
====================================================================
Handles dataset acquisition, integrity verification, stratified splitting,
preprocessing execution, and serialization of both disk images and ML-ready
NumPy tensors (X_train, y_train, etc.).
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Generator
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocess import ImagePreprocessor, save_processed_image
from src.synthetic_generator import IndustrialSurfaceGenerator


LABEL_MAPPING = {
    "normal": 0,
    "defective": 1
}
INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def verify_image_integrity(filepath: str) -> bool:
    """
    Checks if an image file exists, can be decoded, and is non-empty.
    """
    try:
        img = cv2.imread(filepath)
        if img is None or img.size == 0:
            return False
        return True
    except Exception:
        return False


def ensure_dataset_present(
    raw_dir: str,
    min_samples_per_class: int = 100,
    generator_seed: int = 42,
) -> Tuple[int, int]:
    """
    Ensures that raw images exist in data/raw/{normal, defective}.
    If fewer than min_samples_per_class images exist, falls back to generating
    a realistic synthetic industrial surface defect dataset.
    """
    normal_dir = os.path.join(raw_dir, "normal")
    defective_dir = os.path.join(raw_dir, "defective")

    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(defective_dir, exist_ok=True)

    normal_files = [f for f in glob.glob(os.path.join(normal_dir, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    defective_files = [f for f in glob.glob(os.path.join(defective_dir, "*")) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(normal_files) >= min_samples_per_class and len(defective_files) >= min_samples_per_class:
        print(f"[DATASET] Found existing raw dataset: {len(normal_files)} normal, {len(defective_files)} defective.")
        return len(normal_files), len(defective_files)

    print(f"[DATASET] Insufficient raw samples ({len(normal_files)} normal, {len(defective_files)} defective).")
    print(f"[DATASET] Generating high-resolution synthetic industrial benchmark dataset...")

    generator = IndustrialSurfaceGenerator(image_size=(512, 512), seed=generator_seed)
    n_count, d_count = generator.generate_dataset(
        normal_dir=normal_dir,
        defective_dir=defective_dir,
        num_normal=min_samples_per_class,
        num_defective=min_samples_per_class,
    )
    print(f"[DATASET] Generation complete: {n_count} normal, {d_count} defective images created.")
    return n_count, d_count


class IndustrialDatasetPipeline:
    """
    Manages end-to-end dataset loading, stratified splitting (70/15/15),
    preprocessing, image export, and NumPy tensor serialization.
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        preprocessor: Optional[ImagePreprocessor] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    def collect_raw_records(self) -> List[Tuple[str, int, str]]:
        """
        Scans raw_dir and returns verified list of (filepath, class_idx, class_name).
        """
        records = []
        for class_name, class_idx in LABEL_MAPPING.items():
            class_folder = os.path.join(self.raw_dir, class_name)
            if not os.path.exists(class_folder):
                continue

            patterns = [os.path.join(class_folder, f"*.{ext}") for ext in ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]]
            files = []
            for p in patterns:
                files.extend(glob.glob(p))

            files = sorted(files)
            for fpath in files:
                if verify_image_integrity(fpath):
                    records.append((fpath, class_idx, class_name))
                else:
                    print(f"[WARNING] Skipping unreadable/corrupted raw image: {fpath}")

        return records

    def split_records(
        self,
        records: List[Tuple[str, int, str]],
    ) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
        """
        Splits dataset records into Train (70%), Validation (15%), and Test (15%)
        using stratified sampling for exact class balance preservation.
        """
        labels = [r[1] for r in records]

        # First split: Train (70%) vs Temp (30%)
        temp_ratio = self.val_ratio + self.test_ratio
        train_records, temp_records = train_test_split(
            records,
            test_size=temp_ratio,
            stratify=labels,
            random_state=self.random_seed,
        )

        # Second split: Val (50% of 30% = 15%) vs Test (50% of 30% = 15%)
        temp_labels = [r[1] for r in temp_records]
        val_sub_ratio = self.val_ratio / temp_ratio

        val_records, test_records = train_test_split(
            temp_records,
            test_size=(1.0 - val_sub_ratio),
            stratify=temp_labels,
            random_state=self.random_seed,
        )

        return train_records, val_records, test_records

    def process_split_partition(
        self,
        split_name: str,
        records: List[Tuple[str, int, str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes images in a split partition:
        1. Applies preprocessor pipeline.
        2. Saves processed uint8 image to data/processed/<split>/<class_name>/<filename>.
        3. Collects normalized float32 feature array and label array.
        """
        X_list = []
        y_list = []

        print(f"\n[PIPELINE] Processing '{split_name}' set ({len(records)} images)...")
        for fpath, label_idx, class_name in tqdm(records, desc=f"Split [{split_name}]"):
            # Execute preprocessing pipeline
            uint8_img, norm_tensor = self.preprocessor.process_pipeline(fpath)

            # Destination for preprocessed image file
            filename = os.path.basename(fpath)
            out_folder = os.path.join(self.processed_dir, split_name, class_name)
            out_path = os.path.join(out_folder, filename)
            save_processed_image(uint8_img, out_path)

            X_list.append(norm_tensor)
            y_list.append(label_idx)

        X_array = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.int64)

        return X_array, y_array

    def run(self) -> Dict:
        """
        Executes end-to-end pipeline:
        - Ingestion
        - Stratified Split
        - Image Processing & Disk Export
        - Feature Tensor Serialization (.npy)
        - Quality & Range Validation
        """
        # Ensure directories exist
        for split in ["train", "val", "test"]:
            for cls in ["normal", "defective"]:
                os.makedirs(os.path.join(self.processed_dir, split, cls), exist_ok=True)

        records = self.collect_raw_records()
        if not records:
            raise RuntimeError(f"No valid raw images found in {self.raw_dir}")

        total_normal = sum(1 for r in records if r[1] == LABEL_MAPPING["normal"])
        total_defective = sum(1 for r in records if r[1] == LABEL_MAPPING["defective"])

        # Perform stratified splitting
        train_rec, val_rec, test_rec = self.split_records(records)

        # Process each partition
        X_train, y_train = self.process_split_partition("train", train_rec)
        X_val, y_val = self.process_split_partition("val", val_rec)
        X_test, y_test = self.process_split_partition("test", test_rec)

        # Serialize numeric feature arrays to .npy
        print("\n[PIPELINE] Serializing feature arrays to .npy format...")
        npy_paths = {
            "X_train": os.path.join(self.processed_dir, "X_train.npy"),
            "y_train": os.path.join(self.processed_dir, "y_train.npy"),
            "X_val": os.path.join(self.processed_dir, "X_val.npy"),
            "y_val": os.path.join(self.processed_dir, "y_val.npy"),
            "X_test": os.path.join(self.processed_dir, "X_test.npy"),
            "y_test": os.path.join(self.processed_dir, "y_test.npy"),
        }

        np.save(npy_paths["X_train"], X_train)
        np.save(npy_paths["y_train"], y_train)
        np.save(npy_paths["X_val"], X_val)
        np.save(npy_paths["y_val"], y_val)
        np.save(npy_paths["X_test"], X_test)
        np.save(npy_paths["y_test"], y_test)

        # Integrity and sanity checks
        for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            if has_nan or has_inf:
                raise ValueError(f"Integrity check failed: {name} contains NaN or Inf!")

        stats = {
            "total_raw": len(records),
            "total_normal": total_normal,
            "total_defective": total_defective,
            "train_count": len(train_rec),
            "val_count": len(val_rec),
            "test_count": len(test_rec),
            "train_normal": int(np.sum(y_train == 0)),
            "train_defective": int(np.sum(y_train == 1)),
            "val_normal": int(np.sum(y_val == 0)),
            "val_defective": int(np.sum(y_val == 1)),
            "test_normal": int(np.sum(y_test == 0)),
            "test_defective": int(np.sum(y_test == 1)),
            "X_train_shape": X_train.shape,
            "X_train_dtype": str(X_train.dtype),
            "X_train_min": float(np.min(X_train)),
            "X_train_max": float(np.max(X_train)),
            "X_val_shape": X_val.shape,
            "X_val_dtype": str(X_val.dtype),
            "X_val_min": float(np.min(X_val)),
            "X_val_max": float(np.max(X_val)),
            "X_test_shape": X_test.shape,
            "X_test_dtype": str(X_test.dtype),
            "X_test_min": float(np.min(X_test)),
            "X_test_max": float(np.max(X_test)),
            "npy_paths": npy_paths,
        }

        return stats


def load_processed_tensors(
    processed_dir: str = "data/processed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct loader function for Phase 2 ML / CNN pipelines.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


class BatchLoader:
    """
    Mini-batch generator for training ML / CNN models in Phase 2.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = len(X)
        self.indices = np.arange(self.num_samples)

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(self.indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = self.indices[i : i + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
