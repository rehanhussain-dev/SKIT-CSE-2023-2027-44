import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_srgan_inference import read_model_input


class DummyRasterSource:
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


if __name__ == "__main__":
    unittest.main()
