# WOFOST Module

This module contains the WOFOST/PCSE crop-growth simulation and physiological diagnostic components of **VHR-YieldNet — Farm Yield Prediction using Very High Spatial Resolution Data**. It provides process-based crop trajectories for future integration with satellite observations and a physics-informed LSTM.

**Development status (24 November 2026):** The WOFOST engine, Rajasthan test-field configuration, weather integration, sensitivity experiments and production-mode diagnostics have been implemented and executed locally. Regional calibration and validation against measured crop-growth and yield records remain pending.

## 1. Current implementation

- WOFOST conceptual study, PCSE installation and built-in demonstration.
- Complete baseline crop simulation and extraction of daily outputs into Pandas DataFrames.
- LAI, TAGP and TWSO visualization; DVS and seasonal summary analysis.
- Rajasthan test-field configuration for the Jhunjhunu region, with NASA POWER weather inputs and representative soil parameters derived from SoilGrids information.
- Experimental wheat crop calendar, cultivar selection, site and soil parameter providers.
- Soil-water, irrigation and cultivar sensitivity experiments.
- Leaf cohort, senescence, root growth and maintenance-respiration diagnostics.
- Controlled comparison of potential and water-limited production.
- Preliminary alignment of daily WOFOST states with field-masked weekly Sentinel-2 vegetation-index composites.

The original built-in demonstration is retained for reference. The Rajasthan configuration is an experimental setup, not a field-calibrated crop model. The assumed wheat cultivar and management calendar are not confirmed records of the crop cultivated on the satellite test field.

## 2. Module structure

```text
wofost/
├── README.md
├── first_simulation.py
├── run_wheat_test.py
├── test_wheat_setup.py
├── compare_cultivars.py
├── soil_water_sensitivity.py
├── irrigation_sensitivity.py
├── diagnose_grain_growth.py
├── compare_production_modes.py
├── diagnose_production_modes.py
├── agromanagement/
│   └── wheat_2025_26.py
├── field/
│   └── test_field.json
├── soil/
│   ├── rajasthan_sandy_clay_loam.json
│   ├── soil_provider.py
│   └── test_soil_pcse.py
├── weather/
│   ├── pcse_weather.py
│   ├── prepare_nasa_power.py
│   ├── prepare_weather.py
│   ├── download/
│   ├── raw/
│   └── processed/
├── wofost_site/
│   └── site_provider.py
└── outputs/
```

## 3. Simulation inputs and assumptions

| Component | Current experimental configuration |
|---|---|
| Model | WOFOST 7.2 via PCSE |
| Test region | Jhunjhunu, Rajasthan, India |
| Simulated crop | Wheat |
| Baseline cultivar | `Winter_wheat_107` |
| Simulated sowing date | 15 November 2025 |
| Weather | NASA POWER weather series prepared for PCSE |
| Soil | Representative sandy clay loam configuration informed by SoilGrids 250 m pixel composition |
| Crop management | Experimental schedule in `agromanagement/wheat_2025_26.py` |
| Production modes | `Wofost72_PP` and `Wofost72_WLP_CWB` |

The SoilGrids pixel is larger than the selected field. Soil properties are representative estimates, not field-laboratory measurements. The simulated sowing date, cultivar and irrigation treatments must not be treated as verified farm management records. Mustard parameterization is not established by the reported wheat experiments.

## 4. Main output variables

| Variable | Meaning | Typical unit |
|---|---|---|
| `DVS` | Crop development stage | Dimensionless |
| `LAI` | Leaf area index | m²/m² |
| `TAGP` | Total above-ground dry biomass | kg/ha |
| `TWSO` | Total storage-organ dry biomass | kg/ha |
| `TWLV` | Total leaf dry biomass | kg/ha |
| `TWST` | Total stem dry biomass | kg/ha |
| `TWRT` | Total root dry biomass | kg/ha |
| `TRA` | Actual transpiration rate | cm/day |
| `RD` | Rooting depth | cm |
| `SM` | Volumetric soil moisture | cm³/cm³ |
| `WWLOW` | Water content in rooted and lower soil zones (water-balance output) | cm |
| `RFTRA` | Transpiration reduction factor | Dimensionless |

Availability depends on the production mode and PCSE output configuration. `TWSO` is simulated dry storage-organ biomass; it is not a measured farm yield or a moisture-adjusted marketable yield.

## 5. Earlier sensitivity experiments

The soil-water, irrigation and cultivar experiments are retained in the existing scripts and tracked output files. Their purpose is to diagnose model behavior rather than optimize parameters against an unknown yield target.

For the water-limited cultivar comparison, `Winter_wheat_107` matured on 5 May 2026 with TWSO of 1.6324 kg/ha, while `Winter_wheat_106` matured on 19 May 2026 with TWSO of 0 kg/ha. The `apache` simulation exhausted its weather coverage on 31 May 2026 at DVS 0.724; it did not reach maturity, and its final recorded storage-organ biomass is not a mature yield estimate.

The irrigation sensitivity runs indicated improved simulated storage-organ production with additional irrigation, but did not establish a calibrated or agronomically representative irrigation schedule.

## 6. Production-mode comparison (23–29 November 2026 development week)

A controlled experiment compared potential production (`Wofost72_PP`) and water-limited production (`Wofost72_WLP_CWB`) using the same cultivar, weather, site, soil inputs and experimental crop calendar. No cultivar parameters were changed.

Potential production removes water limitation; it does not remove temperature, radiation, respiration or phenological constraints.

### Results from local execution

| Metric | Potential | Water-limited |
|---|---:|---:|
| Daily state records | 172 | 172 |
| Anthesis proxy (DVS ≥ 1) | 2026-04-02 | 2026-04-02 |
| Maturity (DVS ≥ 2) | 2026-05-05 | 2026-05-05 |
| Maximum LAI | 7.0852 | 6.8934 |
| LAI at anthesis | 3.3715 | 2.2274 |
| Mean grain-filling LAI | 1.1329 | 0.4148 |
| TAGP at anthesis (kg/ha) | 8,194.43 | 5,541.11 |
| TAGP at maturity (kg/ha) | 9,048.45 | 5,542.74 |
| TWSO at anthesis (kg/ha) | 171.89 | 0.00 |
| TWSO at maturity (kg/ha) | 1,025.91 | 1.63 |
| TWSO gain from anthesis to maturity (kg/ha) | 854.02 | 1.63 |
| Final harvest index (TWSO/TAGP) | 11.338% | 0.0295% |

### Interpretation

- Removing water limitation substantially increases simulated grain biomass, confirming water stress as a contributor to the poor baseline outcome.
- The potential-production harvest index is also low, motivating further physiological investigation.
- Both modes reach the same simulated anthesis and maturity dates under the current configuration.
- Potential-production grain biomass stops increasing after the state recorded on 11 April 2026, despite simulated maturity occurring on 5 May. Water-limited grain biomass reaches its final value on 9 April.
- These values are reproducible local model outputs, not validated field yields.

The reusable script is `compare_production_modes.py`. Its summary is retained in `outputs/production_mode_comparison.csv`.

## 7. Assimilation and respiration diagnosis

The production-mode diagnostic records daily crop states and physiological rates after the rate calculation. The rate records number 171 per mode, versus 172 daily state records; the terminal state need not have a subsequent rate calculation.

| Variable | Diagnostic meaning |
|---|---|
| `GASS` | Gross assimilation |
| `PMRES` | Potential maintenance respiration |
| `MRES` | Actual maintenance respiration |
| `ASRC` | Assimilates available for growth |
| `DMI` | Dry matter increase |
| `GRSO` | Storage-organ growth rate |
| `DALV` | Age-related leaf death rate |
| `TEMP` | Daily temperature driver |
| `IRRAD` | Daily irradiation driver |

For the recorded assimilation balance, actual maintenance respiration is limited by available gross assimilation (`MRES = min(GASS, PMRES)`), and `ASRC = GASS - MRES`. When all available assimilates are consumed by maintenance respiration, no assimilates remain for new growth.

### Potential-production diagnostic around grain-growth cessation

| Date (2026) | TEMP (°C) | LAI | GASS | PMRES | ASRC | GRSO |
|---|---:|---:|---:|---:|---:|---:|
| 10 Apr | 23.780 | 1.7654 | 236.1799 | 135.4422 | 100.7377 | 71.4230 |
| 11 Apr | 26.935 | 1.5944 | 141.7683 | 166.5997 | 0.0000 | 0.0000 |
| 12 Apr | 26.830 | 1.4167 | 145.1781 | 162.5376 | 0.0000 | 0.0000 |
| 16 Apr | 32.880 | 0.8572 | 0.0000 | 233.5449 | 0.0000 | 0.0000 |

The diagnostic rates are reported in the native model units. State accumulation and rate calculations can appear one day apart.

| Grain-filling rate records (DVS ≥ 1) | Potential | Water-limited |
|---|---:|---:|
| Recorded days | 33 | 33 |
| Missing ASRC values | 0 | 0 |
| Days with zero available assimilates | 24 | 32 |
| Days with positive storage-organ growth | 9 | 1 |

The experiment identifies an assimilation–maintenance-respiration imbalance as the immediate mechanism of growth cessation. Rising temperature and declining leaf area coincide with this imbalance; their independent causal contributions and the suitability of the cultivar parameters remain to be evaluated. Water limitation aggravates the problem but is not its only explanation.

The baseline crop parameters are preserved. Changing SPAN, TSUM1, TSUM2 or respiration coefficients merely to increase simulated yield is not justified without calibration evidence.

## 8. Preliminary satellite alignment and LSTM integration

Field polygon masking and quality filtering produced 178 accepted observations from 183 Sentinel-2 nominal 10 m export composites across ten seasons. The comparison uses the corrected field-level dataset, not the original buffered rectangle averages.

For 2025–2026, sixteen composites overlap the simulated wheat season. WOFOST states were averaged over each complete composite interval (normally seven days, with the final interval shortened to six days). This is an approximate temporal alignment because exact contributing satellite acquisition times are not available.

| Correlation | Composite start-date alignment | Weekly mean WOFOST alignment |
|---|---:|---:|
| NDVI–LAI | -0.8069 | -0.8163 |
| NDVI–TAGP | -0.8085 | -0.8294 |
| NDRE–LAI | -0.7523 | -0.7534 |

The observed vegetation indices decline while the model simulates canopy development. The disagreement remains after weekly alignment. These descriptive correlations from one season do not independently establish the cause of the mismatch. The satellite observations do not cover the simulated anthesis and grain-filling period.

Because the actual crop identity, regional parameters and management records are unverified, the satellite and WOFOST trajectories should not yet be treated as equivalent measurements of wheat growth or used to enforce an unvalidated physics loss.

The separate `lstm/` module currently supports exploratory seven-day NDVI forecasting from native-resolution satellite observations. Physics-informed fusion and actual yield validation are later development objectives. See the [main project README](../README.md) for LSTM training and evaluation results.

## 9. Running the module

Run commands from the repository root in the project's configured Python environment. The scripts use PCSE and existing local weather, soil and crop parameter inputs. The two production-mode scripts additionally use Pandas. A clean-environment installation has not yet been independently verified; use the compatible PCSE installation already established for this project. The PCSE crop parameter provider must have access to its supported wheat cultivar data.

Baseline and earlier sensitivity experiments:

```cmd
python wofost\run_wheat_test.py
python wofost\soil_water_sensitivity.py
python wofost\irrigation_sensitivity.py
python wofost\compare_cultivars.py
python wofost\diagnose_grain_growth.py
```

Current production-mode experiments:

```cmd
python wofost\compare_production_modes.py
python wofost\diagnose_production_modes.py
```

The two current scripts save new experiment outputs separately from the existing baseline. Re-running an experiment may overwrite its own generated output files.

### Output management

| Output (relative to `wofost/`) | Purpose / repository policy |
|---|---|
| `outputs/wheat_2025_26_daily.csv` | Existing tracked wheat baseline |
| `outputs/production_mode_comparison.csv` | Production-mode summary; retain for version control |
| `outputs/production_potential_daily.csv` | Regenerable daily states; ignored locally |
| `outputs/production_water_limited_daily.csv` | Regenerable daily states; ignored locally |
| `outputs/production_potential_diagnostic.csv` | Regenerable physiological rates; ignored locally |
| `outputs/production_water_limited_diagnostic.csv` | Regenerable physiological rates; ignored locally |

Existing tracked sensitivity outputs and visualizations remain preserved. Original SoilGrids rasters remain local for provenance. See the repository-root `.gitignore` for the specific exclusions.

## 10. Remaining requirements and next steps

- Confirm the actual test-field crop identity, cultivar, sowing and harvest dates, and management history.
- Obtain representative soil-water information, observed crop growth and measured yield labels for regional calibration and validation.
- Investigate the suitability of regional crop phenology, photosynthetic temperature response, leaf lifespan and respiration parameters using evidence-supported sensitivity and calibration experiments.
- Extend weather coverage when a cultivar experiment does not reach maturity.
- Obtain validated 2.5 m super-resolved vegetation-index inputs and check their spectral and spatial suitability.
- Align verified crop trajectories with the corresponding WOFOST simulations before designing physics-informed constraints.
- Implement and validate physics fusion, followed by joint training and independent yield evaluation. Evaluate mustard separately when suitable input data and parameters are available.

The November 2026 exploratory WOFOST diagnostics are locally executed and documented. A calibrated regional yield model, verified satellite-to-crop alignment and physics-informed joint model are **not yet complete**.
