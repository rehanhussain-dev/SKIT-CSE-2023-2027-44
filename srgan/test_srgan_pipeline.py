import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_srgan import split_paths
from run_srgan_inference import read_model_input, validate_inference_source
from run_srresnet_inference import normalize, snapped_grid, super_resolve, utm_epsg
from upscale_timeseries import upscale_mask
from validate_sr_consistency import (
    block_mean,
    edge_strength,
    georeference_issues,
    lstm_series_consistency,
    ndvi,
    radiometric_consistency,
    season_of,
)


class DummyGrid:
    def __init__(self, height, width, res, bounds=(0.0, 0.0, 40.0, 20.0), crs="EPSG:32643"):
        self.height, self.width, self.res, self.bounds, self.crs = height, width, (res, res), bounds, crs


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
    def test_grouped_split_keeps_related_patches_together(self):
        paths = [
            Path("field_a_tile_01.npz"),
            Path("field_a_tile_02.npz"),
            Path("field_b_tile_01.npz"),
            Path("field_b_tile_02.npz"),
        ]

        train, validation = split_paths(
            paths,
            validation_fraction=0.5,
            seed=42,
            group_pattern=r"(?P<group>field_[a-z])_",
        )

        train_groups = {path.name.split("_")[1] for path in train}
        validation_groups = {path.name.split("_")[1] for path in validation}
        self.assertEqual(train_groups | validation_groups, {"a", "b"})
        self.assertTrue(
            train_groups.isdisjoint(validation_groups)
        )

    def test_grouped_split_rejects_pattern_without_group_capture(self):
        with self.assertRaisesRegex(ValueError, "named 'group'"):
            split_paths(
                [Path("field_a.npz"), Path("field_b.npz")],
                validation_fraction=0.5,
                seed=42,
                group_pattern=r"field_[a-z]",
            )

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

    def test_block_mean_inverts_nearest_neighbour_upsampling(self):
        lr = np.random.default_rng(0).random((4, 3, 5)).astype(np.float32)
        sr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)

        np.testing.assert_allclose(block_mean(sr), lr, rtol=1e-6)
        metrics = radiometric_consistency(lr, sr)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=6)
        self.assertAlmostEqual(metrics["r"], 1.0, places=6)

    def test_radiometric_consistency_reports_bias(self):
        lr = np.full((4, 2, 2), 0.2, dtype=np.float32)
        sr = np.full((4, 8, 8), 0.25, dtype=np.float32)

        metrics = radiometric_consistency(lr, sr)
        self.assertAlmostEqual(metrics["bias"], 0.05, places=6)
        self.assertAlmostEqual(metrics["rmse"], 0.05, places=6)

    def test_block_mean_ignores_masked_pixels(self):
        sr = np.ones((1, 4, 4), dtype=np.float32)
        sr[0, 0, 0] = np.nan
        np.testing.assert_allclose(block_mean(sr), [[[1.0]]])

    def test_edge_strength_is_higher_for_sharp_boundary(self):
        sharp = np.zeros((8, 8))
        sharp[:, 4:] = 1.0
        blurred = np.tile(np.linspace(0, 1, 8), (8, 1))
        self.assertGreater(edge_strength(sharp), edge_strength(blurred))

    def test_ndvi_uses_red_and_nir_and_masks_empty_pixels(self):
        bands = np.zeros((4, 1, 2), dtype=np.float32)
        bands[0, 0, 0], bands[3, 0, 0] = 0.1, 0.3
        result = ndvi(bands)
        self.assertAlmostEqual(float(result[0, 0]), 0.5, places=6)
        self.assertTrue(np.isnan(result[0, 1]))

    def test_upscale_mask_repeats_each_pixel(self):
        mask = np.array([[True, False]])
        self.assertEqual(upscale_mask(mask, 2).tolist(), [[True, True, False, False]] * 2)

    def test_georeference_check_accepts_exact_4x_grid(self):
        self.assertEqual(georeference_issues(DummyGrid(2, 4, 10.0), DummyGrid(8, 16, 2.5)), [])

    def test_georeference_check_flags_shifted_or_mis_sized_output(self):
        lr = DummyGrid(2, 4, 10.0)
        shifted = DummyGrid(8, 16, 2.5, bounds=(2.5, 0.0, 42.5, 20.0))
        self.assertIn("footprint (bounds) differs from the input", georeference_issues(lr, shifted))
        self.assertTrue(georeference_issues(lr, DummyGrid(4, 8, 5.0)))
        self.assertTrue(georeference_issues(lr, DummyGrid(8, 16, 2.5, crs="EPSG:4326")))

    def test_season_of_groups_rabi_dates(self):
        self.assertEqual(season_of("2022-11-05"), "2022-2023")
        self.assertEqual(season_of("2023-03-18"), "2022-2023")

    def test_lstm_series_consistency_is_per_season(self):
        rows = [
            {"season": "2022-2023", "date": f"2022-11-0{i}", "ndvi_mean_lr": 0.1 * i, "ndvi_mean_sr": 0.1 * i + 0.01}
            for i in range(1, 6)
        ] + [{"season": "2023-2024", "date": "2023-11-01", "ndvi_mean_lr": 0.3, "ndvi_mean_sr": 0.2}]

        result = lstm_series_consistency(rows)
        self.assertAlmostEqual(result["2022-2023"]["r"], 1.0, places=6)
        self.assertAlmostEqual(result["2022-2023"]["max_abs_diff"], 0.01, places=6)
        self.assertAlmostEqual(result["2023-2024"]["max_abs_diff"], 0.1, places=6)

    def test_utm_zone_for_rajasthan_is_43n(self):
        self.assertEqual(utm_epsg(75.13, 28.05), 32643)
        self.assertEqual(utm_epsg(75.13, -28.05), 32743)

    def test_snapped_grid_stays_inside_bounds_on_12m_lattice(self):
        left, top, width, height = snapped_grid((512480.5, 3102620.0, 512840.0, 3102880.0))
        self.assertEqual((left, top, width, height), (512484.0, 3102876.0, 29, 21))
        self.assertEqual(left % 12, 0)
        self.assertLessEqual(left + width * 12, 512840.0)
        self.assertGreaterEqual(top - height * 12, 3102620.0)

    def test_snapped_grid_rejects_input_smaller_than_one_pixel(self):
        with self.assertRaisesRegex(ValueError, "smaller than one"):
            snapped_grid((1.0, 1.0, 11.0, 11.0))

    def test_srresnet_normalize_scales_digital_numbers_and_keeps_nan(self):
        result = normalize(np.array([[2000.0, np.nan, -np.inf]], dtype=np.float32))
        self.assertAlmostEqual(float(result[0, 0]), 0.2, places=6)
        self.assertTrue(np.isnan(result[0, 1]) and np.isnan(result[0, 2]))

    def test_super_resolve_reorders_bands_and_masks_invalid_input(self):
        import torch

        def nearest_4x(tensor):  # stand-in for the network
            return torch.nn.functional.interpolate(tensor, scale_factor=4, mode="nearest")

        lr = np.stack([np.full((2, 2), value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.4)])
        lr[:, 0, 0] = np.nan
        sr = super_resolve(nearest_4x, lr)

        self.assertEqual(sr.shape, (4, 8, 8))
        self.assertTrue(np.isnan(sr[:, :4, :4]).all())
        # Model order Blue, Green, Red, NIR -> output order B4 (red), B3, B2, B8
        np.testing.assert_allclose(sr[:, 7, 7], [0.3, 0.2, 0.1, 0.4], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
