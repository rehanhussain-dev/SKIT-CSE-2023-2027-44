# SRGAN pipeline

This directory implements the Spatial Super-Resolution user story: review
Sentinel-2 patches, prepare the GEE band layout, and run the pretrained
RGB-NIR OpenSR model at 4x resolution (10 m to 2.5 m).

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

```powershell
py -3 srgan/run_srgan_inference.py data/patches/field_01.tif
py -3 srgan/run_srgan_inference.py data/patches/field_01.tif --output data/sr/field_01_SR.tif
```

Use `--install` only when the environment has internet access and the SRGAN
dependencies are not installed yet. GPU inference is preferred; CPU inference
works but is slower.