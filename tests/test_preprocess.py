"""
Unit tests for ImagePreprocessor pipeline.
"""

import unittest
import numpy as np
import cv2

from src.preprocess import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
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
        # Create a synthetic 512x512 dummy test image
        self.dummy_img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

    def test_resize(self):
        resized = self.preprocessor.resize_image(self.dummy_img)
        self.assertEqual(resized.shape[:2], (224, 224))

    def test_color_conversion_grayscale(self):
        gray = self.preprocessor.convert_color_space(self.dummy_img)
        self.assertEqual(gray.ndim, 2)

    def test_clahe_enhancement(self):
        gray = self.preprocessor.convert_color_space(self.dummy_img)
        enhanced = self.preprocessor.apply_clahe(gray)
        self.assertEqual(enhanced.shape, gray.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_normalization_minmax(self):
        gray = self.preprocessor.convert_color_space(self.dummy_img)
        norm = self.preprocessor.normalize(gray)
        self.assertEqual(norm.dtype, np.float32)
        self.assertGreaterEqual(norm.min(), 0.0)
        self.assertLessEqual(norm.max(), 1.0)

    def test_process_pipeline_output_shapes(self):
        u8_img, norm_tensor = self.preprocessor.process_pipeline(self.dummy_img)
        self.assertEqual(u8_img.shape, (224, 224))
        self.assertEqual(u8_img.dtype, np.uint8)
        self.assertEqual(norm_tensor.shape, (224, 224, 1))
        self.assertEqual(norm_tensor.dtype, np.float32)
        self.assertFalse(np.isnan(norm_tensor).any())
        self.assertFalse(np.isinf(norm_tensor).any())

    def test_batch_processing(self):
        batch = [self.dummy_img, self.dummy_img.copy()]
        u8_list, tensor_batch = self.preprocessor.preprocess_batch(batch)
        self.assertEqual(len(u8_list), 2)
        self.assertEqual(tensor_batch.shape, (2, 224, 224, 1))


if __name__ == "__main__":
    unittest.main()
