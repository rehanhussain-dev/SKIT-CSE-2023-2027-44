# WOFOST Module

This module contains the initial WOFOST/PCSE experiments
for the VHR-YieldNet project.

## Current Status

Completed:

- WOFOST conceptual study
- PCSE installation and environment setup
- Built-in WOFOST demonstration
- Complete crop-growth simulation
- Extraction of daily WOFOST outputs
- Conversion of outputs to Pandas DataFrame
- LAI visualization
- TAGP visualization
- TWSO visualization
- DVS analysis
- Seasonal summary analysis

## Baseline Simulation

The current experiment uses the built-in PCSE/WOFOST
demonstration configuration.

It is a baseline experiment and is not yet the final
Rajasthan-specific WOFOST configuration.

## Main Output Variables

The current simulation provides:

- DVS
- LAI
- TAGP
- TWSO
- TWLV
- TWST
- TWRT
- TRA
- RD
- SM
- WWLOW
- RFTRA

## Next Steps

1. Understand WOFOST input requirements
2. Prepare weather data for Rajasthan
3. Configure soil parameters
4. Configure wheat and mustard crop parameters
5. Add crop management information
6. Run Rajasthan-specific simulations
7. Generate field-level WOFOST features
8. Align WOFOST outputs with Sentinel-2 observations
9. Provide WOFOST information to the physics-informed LSTM