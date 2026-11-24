# VHR-YieldNet

**Farm Yield Prediction using Very High Spatial Resolution Data**

VHR-YieldNet is a final-year project investigating the integration of satellite vegetation observations, temporal deep learning and process-based crop simulation for farm-level yield prediction. The intended approach combines very-high-resolution satellite inputs, an LSTM growth-trajectory model and WOFOST crop-growth simulations through physics-informed learning.

This repository documents the WOFOST and temporal LSTM development work. The current implementation includes a locally tested native-resolution NDVI forecasting pipeline and WOFOST diagnostic experiments. Validated super-resolution inputs and joint yield prediction are not yet implemented in this pipeline.

**Development snapshot:** 24 November 2026 (week of 23–29 November 2026).

## Current development status

| Component | Status |
| --- | --- |
| WOFOST 7.2 engine and weather integration | Implemented; regional calibration pending |
| Soil and crop-management configuration | Representative test configuration available |
| Field-level satellite vegetation-index extraction | Implemented and locally tested |
| Supervised temporal sequence preparation | Implemented and numerically verified |
| LSTM training and checkpoint selection | Implemented and locally tested |
| Persistence and ridge-regression benchmarks | Implemented and evaluated |
| Validated 2.5 m super-resolution inputs | Pending |
| Wheat growth and measured yield validation | Pending |
| Physics-informed WOFOST–LSTM joint training | Planned |
| Threshold testing and explainability | Planned |

## Methodology

### Satellite preprocessing

The current satellite data consist of weekly Sentinel-2 composite exports for one field (`farmA`) in the Jhunjhunu region of Rajasthan, India. The exports have a nominal 10 m pixel grid; this does not imply that every source band has native 10 m resolution. The crop identity of this field is unverified.

The previous vegetation-index extraction included pixels outside the farm polygon because the exported rasters covered a buffered rectangular region. The improved extraction uses a pixel-centre polygon mask in the raster coordinate system, checks finite NDVI, NDRE and SI values, and retains observations with at least 80% valid field-pixel coverage. The coverage threshold is an experimental quality-control choice, not a guarantee of cloud-free observations.

| Dataset statistic | Value |
| --- | ---: |
| Exported composite rasters | 183 |
| Field pixel centres per raster | 227 |
| Accepted observations | 178 |
| Rejected observations | 5 |
| Seasons represented in accepted observations | 10 |

The corrected dataset is stored in `s2_field_vegetation_indices_timeseries.csv`. It retains quality flags, field coverage, source filenames, crop-identity status and vegetation-index statistics. The original `s2_vegetation_indices_timeseries.csv` is preserved for historical comparison.

### Temporal LSTM dataset

The exploratory task is to predict the next weekly NDVI composite from four preceding accepted observations. The forecast horizon is seven days between composite start dates, rather than between individual satellite acquisition dates.

The five input features, in order, are:

1. `ndvi_mean`
2. `ndre_mean`
3. `si_mean`
4. `days_since_season_start`
5. `days_since_previous_observation`

Season start is a calendar reference (1 October), not a verified sowing date. The first observation in each input window has its preceding-gap feature set to zero. Irregular observation intervals are retained and represented by the temporal features. Input standardization is fitted on training windows only; targets remain in original NDVI units.

| Split | Assigned seasons | Sequences | Input shape |
| --- | --- | ---: | --- |
| Training | 2016–17 to 2021–22 | 52 | `(52, 4, 5)` |
| Validation | 2022–23 and 2023–24 | 29 | `(29, 4, 5)` |
| Test | 2024–25 and 2025–26 | 27 | `(27, 4, 5)` |
| Total | 10 assigned seasons | 108 | — |

The 2016–17 season yields no eligible seven-day forecasting sequences, so five seasons contribute to training. The windows overlap within seasons and are not independent field experiments. No season appears in more than one split.

### LSTM architecture and training

The existing configurable `YieldLSTM` class is reused for NDVI forecasting without changing its original default architecture. The current experiment uses the following configuration:

| Setting | Value |
| --- | --- |
| Input observations / features | 4 / 5 |
| LSTM hidden size / layers | 16 / 1 |
| Output | One NDVI prediction per sequence |
| Trainable parameters | 1,489 |
| Batch size | 8 |
| Optimizer / learning rate | Adam / 0.001 |
| Loss function | Mean squared error (MSE) |
| Maximum epochs / early-stopping patience | 200 / 20 |
| Random seed / device | 42 / CPU |

The final LSTM time-step representation is passed to a linear regression layer. The best checkpoint is selected by validation MSE; the test split is excluded from training and checkpoint selection. Training ran for 128 epochs, with the best checkpoint at epoch 108.

## Experimental results

Three forecasting methods are evaluated on the same sequences: persistence (last observed NDVI), ridge regression (flattened input windows, alpha = 1.0, fitted on training data only), and the trained LSTM.

### Validation (29 sequences)

| Model | MAE | RMSE | R² | Mean bias |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.012880 | 0.017951 | 0.644899 | +0.006836 |
| Ridge regression | 0.011523 | 0.015651 | 0.730069 | +0.005116 |
| LSTM | **0.010869** | **0.014685** | **0.762360** | **+0.003012** |

### Temporal holdout test (27 sequences)

| Model | MAE | RMSE | R² | Mean bias |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.011212 | 0.015809 | 0.595797 | **+0.004316** |
| Ridge regression | **0.010586** | 0.014624 | 0.654120 | −0.006338 |
| LSTM | 0.010862 | **0.013634** | **0.699343** | −0.008341 |

The LSTM achieves the lowest test RMSE, approximately 13.75% below persistence and 6.77% below ridge regression. Ridge regression has a slightly lower test MAE, while persistence has the smallest absolute mean bias. Mean bias is defined as prediction minus observation. The LSTM underestimates NDVI on average in both test seasons.

The test results are preliminary temporal holdout results for one field. The later seasons were previously inspected during exploratory satellite analysis; they are not a completely untouched external validation dataset. The results do not establish statistically significant superiority over ridge regression or generalization to other farms. **These metrics describe NDVI forecasting, not measured crop yield prediction accuracy.**

### WOFOST integration and diagnostics

WOFOST 7.2 potential-production (`Wofost72_PP`) and water-limited (`Wofost72_WLP_CWB`) simulations were compared with the same `Winter_wheat_107` cultivar, weather, soil and management configuration for the 2025–26 simulation season.

| Simulated parameter | Potential production | Water-limited production |
| --- | ---: | ---: |
| Maximum leaf area index (LAI) | 7.0852 | 6.8934 |
| Anthesis proxy (DVS ≥ 1) | 2 April 2026 | 2 April 2026 |
| Maturity (DVS ≥ 2) | 5 May 2026 | 5 May 2026 |
| Final above-ground dry biomass (TAGP, kg/ha) | 9,048.45 | 5,542.74 |
| Final storage-organ dry biomass (TWSO, kg/ha) | 1,025.91 | 1.63 |
| Harvest index (TWSO / TAGP) | 0.11338 | 0.000295 |

Both simulations produce low storage-organ biomass and require regional calibration. Diagnostics identify water limitation and insufficient gross assimilation relative to potential maintenance respiration as contributors to premature grain accumulation failure. In potential production, available assimilates become zero from 11 April through the remaining recorded grain-filling rate calculations.

The 16 matched weekly satellite composites and corresponding mean WOFOST states show negative NDVI–LAI correlation (Pearson r = −0.8163). This preliminary comparison uses composite intervals rather than exact contributing acquisition dates and does not establish the cause of the disagreement. Satellite observations and simulated wheat states are therefore not treated as interchangeable training labels.

See [WOFOST documentation](../wofost/README.md) for the crop simulation module and additional experiments.

## Repository structure

The following is an overview of the main files; historical experiments and other existing WOFOST inputs and outputs are retained.

```text
VHR-YieldNet/
├── README.md
├── .gitignore
├── s2_vegetation_indices_timeseries.csv       # Original observations
├── s2_field_vegetation_indices_timeseries.csv # Corrected observations
├── lstm/
│   ├── lstm_model.py                         # Configurable LSTM architecture
│   ├── extract_field_timeseries.py            # Field polygon extraction
│   ├── prepare_growth_sequences.py            # Supervised temporal dataset
│   ├── train_growth_lstm.py                   # Training and early stopping
│   ├── evaluate_growth_lstm.py                # Comparative evaluation
│   ├── prepare_sequences.py                   # Existing WOFOST sequences
│   ├── prepare_satellite_sequences.py         # Legacy satellite sequences
│   ├── growth_sequences_metadata.csv          # Sequence traceability
│   └── outputs/
│       └── growth_lstm_training_history.csv
└── wofost/
    ├── README.md
    ├── run_wheat_test.py
    ├── compare_production_modes.py
    ├── diagnose_production_modes.py
    ├── agromanagement/
    ├── field/
    ├── soil/
    ├── weather/
    ├── wofost_site/
    └── outputs/
        └── production_mode_comparison.csv
```

## Running the current pipeline

Run the following commands from the repository root. Python with PyTorch, NumPy and pandas is required for LSTM development; PCSE is required for WOFOST, and tifffile for the raw satellite extraction. A pinned dependency environment and clean-machine reproduction have not yet been established. Use the existing working project environment where possible.

### 1. Regenerate the field-level observations (optional)

The corrected CSV is intended to be version-controlled, so the raw extraction step is not needed for training from that CSV. To regenerate it, obtain the team's original satellite export archive and place `vhr_yieldnet_weekly-20261103T161614Z-1-001.zip` in the repository root. The archive is a local input excluded by `.gitignore`.

```cmd
python lstm\extract_field_timeseries.py
```

### 2. Prepare the supervised dataset

```cmd
python lstm\prepare_growth_sequences.py
```

This creates `lstm/growth_sequences.pt` and updates `lstm/growth_sequences_metadata.csv`.

### 3. Train the LSTM

```cmd
python lstm\train_growth_lstm.py
```

This generates `lstm/outputs/growth_lstm_best.pt` and `lstm/outputs/growth_lstm_training_history.csv`.

### 4. Evaluate the model and baselines

```cmd
python lstm\evaluate_growth_lstm.py --split validation
python lstm\evaluate_growth_lstm.py --split test
```

The evaluator checks checkpoint compatibility, reports overall and season-wise metrics, and prints results to the terminal without creating additional evaluation files. Use validation for model selection; reserve test evaluation for a frozen configuration.

### 5. Run WOFOST production-mode experiments

```cmd
python wofost\compare_production_modes.py
python wofost\diagnose_production_modes.py
```

The scripts generate the production-mode summary and detailed daily simulation and physiological diagnostic CSVs under `wofost/outputs/`.

## Reproducibility and repository policy

- Retain reusable source code, corrected observations, sequence metadata, training history and the production-mode summary in Git, subject to the team's data-sharing policy.
- Keep raw satellite archives, original soil rasters, generated sequence tensors, trained checkpoints and new detailed production-mode diagnostics locally; these are excluded by `.gitignore`.
- Regenerate ignored LSTM artifacts using the commands above. Existing tracked historical outputs are preserved.
- All six new Python scripts have passed local syntax validation and have been executed successfully. Dataset checks, forward/backward propagation, checkpoint restoration and validation/test evaluation have passed locally.
- Clean-environment reproduction, a pinned dependency specification and independent external validation remain outstanding.

## Limitations and next steps

- Confirm the field's actual crop identity, sowing and harvest dates, cultivar and irrigation management.
- Obtain validated 2.5 m super-resolved vegetation inputs with appropriate spectral-band support and quality controls. The current experiment uses native-resolution exports.
- Expand to additional field-seasons and obtain measured crop growth and yield labels for independent validation.
- Calibrate WOFOST with representative regional cultivar, soil, weather and management evidence; the current soil configuration is based on representative SoilGrids information, not field laboratory measurements.
- Establish a scientifically justified relationship between satellite vegetation indices and simulated crop states before implementing physics-informed fusion. Do not force the LSTM to reproduce the current mismatched WOFOST trajectory.
- Preserve the current trained LSTM as an exploratory benchmark; evaluate future changes on new data rather than tuning on the reported test seasons.

The current implementation is a research baseline for the ongoing LSTM growth-trajectory development stage. Physics-informed joint training, validated wheat yield prediction and explainability are subsequent project objectives.



## Verified Development Environment

The current exploratory LSTM and WOFOST workflows were
successfully executed using the following local development
environment on 24 November 2026.

| Dependency | Tested Version |
|---|---|
| Python (Windows, 64-bit) | 3.12.10 |
| PyTorch | 2.14.1 |
| NumPy | 2.5.1 |
| Pandas | 3.0.3 |
| PCSE | 6.0.13 |
| Tifffile | 2026.11.16 |
| Matplotlib | 3.11.0 |

LSTM training was performed on CPU using a fixed random
seed of 42 and deterministic PyTorch operations.

The WOFOST diagnostic scripts access version-sensitive
PCSE internal interfaces. Compatibility with other PCSE
versions has not yet been established.

These versions describe the tested local environment,
rather than a complete dependency lockfile for the
entire team project.
