"""
Unit tests for Handcrafted Feature Extraction pipeline.
"""

import unittest
import numpy as np

from src.feature_extraction import FeatureExtractor, FEATURE_NAMES


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        # Synthetic 224x224 grayscale image with a dark horizontal line (defect)
        self.test_img = np.full((224, 224), 150, dtype=np.uint8)
        self.test_img[100:105, 50:180] = 30  # Defect scratch

    def test_feature_vector_dimension(self):
        feats = self.extractor.extract_image_features(self.test_img)
        self.assertEqual(feats.ndim, 1)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertEqual(len(feats), 31)

    def test_no_nan_or_inf(self):
        feats = self.extractor.extract_image_features(self.test_img)
        self.assertFalse(np.isnan(feats).any(), "Extracted features contained NaN values!")
        self.assertFalse(np.isinf(feats).any(), "Extracted features contained Inf values!")

    def test_dataset_extraction(self):
        dataset = [self.test_img, self.test_img.copy()]
        feat_matrix = self.extractor.extract_dataset_features(dataset, desc="Testing")
        self.assertEqual(feat_matrix.shape, (2, 31))
        self.assertEqual(feat_matrix.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
