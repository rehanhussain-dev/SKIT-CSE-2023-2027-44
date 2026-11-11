# SRGAN pipeline

This directory is the Spatial Super-Resolution module of VHR-YieldNet, a
project that predicts crop yield for small farms from satellite imagery.

## What it does

Free Sentinel-2 imagery has 10 m pixels. On a small field, many of those
pixels mix the crop with field edges, paths and neighbouring plots, which
distorts the vegetation signal used for yield prediction. This module upscales
the Sentinel-2 bands 4x, to a 2.5–3 m grid, so that field-level detail is
available to the rest of the project. It also measures whether the upscaled
images can be trusted before anything downstream uses them.

The output feeds the LSTM branch of the project, which models crop growth
from the vegetation-index time series.

## How it works

1. **Review the input patches.** `review_patches.py` checks every exported
   GeoTIFF for the expected shape, band count, coordinate system and valid
   pixels, so that broken or cloud-damaged patches are caught early.
2. **Select and normalise the bands.** The super-resolution models work on
   four bands: Red, Green, Blue and near-infrared (`B4, B3, B2, B8`). These are
   picked out of the eight-band export and scaled to 0–1 reflectance.
3. **Upscale 4x.** Two models are supported:
   - a pretrained RGB-NIR SRGAN from the OpenSR project
     (`run_srgan_inference.py` for one patch, `upscale_timeseries.py` for a
     whole season of patches);
   - an SRResNet trained in this project on Sentinel-2 / PlanetScope image
     pairs (`kaggle_srgan_script.ipynb` to train,
     `run_srresnet_inference.py` to run).
4. **Keep the georeferencing.** Every output is written as a GeoTIFF with the
   same footprint and coordinate system as its input and a pixel size exactly
   one quarter of it, so it lines up with field boundaries and other layers.
5. **Fine-tune on agricultural patches.** `finetune_srgan.py` continues
   training the pretrained generator on paired low/high-resolution patches,
   with a validation split that can keep whole fields out of training.
6. **Validate the result.** `validate_sr_consistency.py` compares each
   upscaled image with its 10 m input, with plain bicubic upsampling, and with
   a real 3 m PlanetScope image where one exists. A model passes only if it
   keeps the original reflectance, keeps the seasonal NDVI curve, and matches
   the real high-resolution image at least as well as bicubic does.

## Files

| File | Purpose |
| --- | --- |
| `review_patches.py` | Quality check of exported GeoTIFF patches |
| `run_srgan_inference.py` | Upscale one patch with the pretrained SRGAN |
| `run_srgan_inference.ipynb` | The same inference as a Google Colab notebook |
| `upscale_timeseries.py` | Upscale a directory of patches (one per date) |
| `finetune_srgan.py` | Fine-tune the pretrained generator on paired patches |
| `kaggle_srgan_script.ipynb` | Train an SRResNet on Sentinel-2 / PlanetScope pairs |
| `run_srresnet_inference.py` | Upscale with the trained SRResNet checkpoint |
| `validate_sr_consistency.py` | Validate upscaled outputs and write a report |
| `test_srgan_pipeline.py` | Unit tests for the pipeline and the validation metrics |
| `requirements-srgan.txt` | Python dependencies for this module |

Imagery, model checkpoints and generated outputs are not committed to the
repository.

## Current status

The upscaling, georeferencing and validation steps work end to end. Neither
model passes the full validation yet: the pretrained SRGAN adds detail that
is not present in the real 3 m imagery, and the current SRResNet checkpoint
blurs the image and flattens the seasonal NDVI curve. Upscaled output should
not be used downstream until a retrained model passes
`validate_sr_consistency.py`. Details are in the sections below.

## Run the tests

```powershell
py -3 -m unittest srgan.test_srgan_pipeline
```

## Google Colab notebook

Open [`run_srgan_inference.ipynb`](run_srgan_inference.ipynb) in Google Colab,
enable a GPU runtime, mount Google Drive, and set `INPUT_TIF` to the exported
GeoTIFF path. The notebook installs the SRGAN dependencies, inspects the input,
selects `B4, B3, B2, B8` for the RGB-NIR model, and writes a georeferenced
four-band `_SR.tif` beside the input.

The expected eight-band GEE export order is:
`B2, B3, B4, B8, B11, NDVI, NDRE, SI`.

## Local install

```powershell
py -3 -m pip install -r srgan/requirements-srgan.txt
```

## Review exported patches

The current GEE export contains eight bands in this order:
`B2, B3, B4, B8, B11, NDVI, NDRE, SI`. The reviewer checks the expected
128x128 shape, band count, CRS, transform, masks, and finite values.

```powershell
py -3 srgan/review_patches.py data/patches
```

## Run inference locally

The inference runner selects `B4, B3, B2, B8` as Red, Green, Blue, NIR,
normalizes raw Sentinel-2 values, and writes a georeferenced `_SR.tif`.
Before loading the model, it rejects inputs with missing CRS/transform,
insufficient bands, invalid dimensions, or no finite pixels in the selected
RGB-NIR bands. Partial invalid pixels are converted to zero during
normalization.

```powershell
py -3 srgan/run_srgan_inference.py data/patches/field_01.tif
py -3 srgan/run_srgan_inference.py data/patches/field_01.tif --output data/sr/field_01_SR.tif
```

Use `--install` only when the environment has internet access and the SRGAN
dependencies are not installed yet. GPU inference is preferred; CPU inference
works but is slower.

## Fine-tune on paired agricultural patches

Create one `.npz` file per example with `lr` shaped `(4, H, W)` and `hr`
shaped `(4, H*4, W*4)`. Both 0–1 reflectance and 0–10000 Sentinel-2 values
are accepted:

```powershell
py -3 srgan/finetune_srgan.py data/paired_patches `
  --output srgan/checkpoints/rgb_nir_finetuned.pt `
  --epochs 10 --batch-size 4 --learning-rate 1e-5
```

The script starts from the pretrained RGB-NIR generator, uses supervised L1
loss on the paired high-resolution target, keeps a deterministic validation
split, clips gradients, and saves the best generator checkpoint. At least two
paired `.npz` files are required. If patches from the same field or scene share
information, use a named regex capture to keep those groups out of both
splits:

```powershell
py -3 srgan/finetune_srgan.py data/paired_patches `
  --group-pattern "(?P<group>field_[^_]+)" `
  --output srgan/checkpoints/rgb_nir_finetuned.pt
```

The pattern must match every filename and define a `group` capture. Group-aware
validation is recommended for estimating performance on unseen fields; without
it, the default split is by individual patch.

## Upscale a time series and validate it for the LSTM branch

Upscale every Sentinel-2 patch in a directory to 2.5 m, then check the
outputs before they are used downstream:

```powershell
py -3 srgan/upscale_timeseries.py srgan/low/LR --output-dir srgan/outputs/sr
py -3 srgan/validate_sr_consistency.py --lr-dir srgan/low/LR `
  --sr-dir srgan/outputs/sr --hr-dir srgan/high/high
```

`upscale_timeseries.py` writes one four-band (`B4, B3, B2, B8`) `_SR.tif` per
date, keeps input nodata as nodata, and skips patches with under 50% valid
pixels. `validate_sr_consistency.py` exits with status 1 if any check fails:

| Check | What it means |
| --- | --- |
| `georeferencing` | Same CRS and footprint as the input, pixel size exactly input / 4 |
| `radiometric_consistency` | SR averaged back to 10 m reproduces the input bands (RMSE at most 0.01) |
| `sharper_than_bicubic` | NDVI edges are steeper than plain bicubic upsampling |
| `lstm_ndvi_series` | Per-season patch-mean NDVI from SR tracks the 10 m series |
| `matches_planet_at_least_as_well_as_bicubic` | Only with `--hr-dir`: NDVI agrees with PlanetScope 3 m at least as well as bicubic does |

Results go to `srgan/outputs/` as `validation_per_date.csv`,
`validation_report.json` and `validation_summary.png`.

With the pretrained weights on the `farmA` series, the first four checks pass
and the PlanetScope check fails: the model adds detail that is not in the
3 m reference. Fine-tune on paired patches before relying on SR output for
field-boundary work. The model outputs RGB-NIR only, so NDVI is available at
2.5 m but NDRE and SI are not.

## Inference with the Kaggle-trained SRResNet

`kaggle_srgan_script.ipynb` trains an SRResNet on Sentinel-2 / PlanetScope
pairs. Download `srresnet_best.pt` from the notebook's Output tab into
`srgan/checkpoints/` (`.pt` files are not committed), then:

```powershell
py -3 srgan/run_srresnet_inference.py srgan/low/LR
py -3 srgan/validate_sr_consistency.py --lr-dir srgan/outputs/srresnet/lr12 `
  --sr-dir srgan/outputs/srresnet/sr --hr-dir srgan/high/high `
  --output-dir srgan/outputs/srresnet
```

The model expects Blue, Green, Red, NIR on a 12 m UTM grid, so the script
reprojects every input to that grid before upscaling to 3 m. It writes the 3 m
result to `sr/` (bands `B4, B3, B2, B8`) and the 12 m model input to `lr12/`,
which is the low-resolution side for the validation.

The current checkpoint (epoch 127) fails validation: it beats bicubic on L1
but agrees with PlanetScope NDVI worse than bicubic, is blurrier than bicubic,
and flattens the NDVI time series. Do not use its output downstream; see
`documentation.md` for the measurements.
