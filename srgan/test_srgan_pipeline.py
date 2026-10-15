import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_srgan_inference import read_model_input, validate_inference_source


class DummyRasterSource:
    width = 2
    height = 2
    count = 4
    crs = "EPSG:32643"
    transform = object()

    def read(self, indices=None):
        bands = np.full((4, 2, 2), np.nan, dtype=np.float32)
        return bands


class SrganPipelineTests(unittest.TestCase):
    def test_read_model_input_handles_all_nan_values(self):
        output = read_model_input(DummyRasterSource())
        self.assertEqual(output.shape, (4, 2, 2))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())

    def test_validate_inference_source_rejects_all_invalid_pixels(self):
        with self.assertRaisesRegex(ValueError, "no finite pixels"):
            validate_inference_source(DummyRasterSource())

    def test_validate_inference_source_accepts_valid_metadata_and_pixels(self):
        source = DummyRasterSource()
        source.read = lambda indices=None: np.ones((4, 2, 2), dtype=np.float32)

        validate_inference_source(source)


if __name__ == "__main__":
    unittest.main()
