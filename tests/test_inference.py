"""
Integration tests for FaultInferenceEngine and prediction pipeline.
"""

import os
import unittest
import numpy as np

from src.infer import FaultInferenceEngine


class TestFaultInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine_cnn = FaultInferenceEngine(model_type="cnn")
        cls.engine_svm = FaultInferenceEngine(model_type="svm")
        # Dummy test image
        cls.sample_normal = "data/raw/normal/normal_0001.png"
        cls.sample_defective = "data/raw/defective/defective_0001.png"

    def test_cnn_inference_runs(self):
        if os.path.exists(self.sample_normal):
            res = self.engine_cnn.predict(self.sample_normal)
            self.assertIn("label", res)
            self.assertIn(res["label"], ["Normal", "Defective"])
            self.assertIn("confidence", res)
            self.assertGreaterEqual(res["confidence"], 0.5)
            self.assertIn("annotated_bgr", res)
            self.assertEqual(res["annotated_bgr"].shape[1], 224)

    def test_svm_inference_runs(self):
        if os.path.exists(self.sample_defective):
            res = self.engine_svm.predict(self.sample_defective)
            self.assertIn("label", res)
            self.assertIn("confidence", res)
            self.assertIn("annotated_bgr", res)

    def test_anomaly_box_detection(self):
        dummy_flaw = np.full((224, 224), 160, dtype=np.uint8)
        dummy_flaw[80:120, 80:120] = 20  # dark patch
        boxes = self.engine_cnn.detect_anomaly_boxes(dummy_flaw)
        self.assertIsInstance(boxes, list)
        self.assertGreater(len(boxes), 0)


if __name__ == "__main__":
    unittest.main()
