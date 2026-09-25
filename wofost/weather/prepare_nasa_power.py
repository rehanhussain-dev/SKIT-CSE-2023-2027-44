"""
Convert NASA POWER daily weather data into a PCSE-friendly dataframe.

Development/test weather source for VHR-YieldNet.
Final project weather should use the specified IMD data where available.
"""

import json
from pathlib import Path

import pandas as pd

from pcse.util import reference_ET
from pcse.util import vap_from_relhum


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "wofost"
    / "weather"
    / "raw"
    / "nasa_power_wheat_2025_26.json"
)

RH_FILE = (
    PROJECT_ROOT
    / "wofost"
    / "weather"
    / "raw"
    / "nasa_power_rh2m_wheat_2025_26.json"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "wofost"
    / "weather"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "nasa_power_wheat_2025_26.csv"
)


# Test field location.
# Final project values should come from the field metadata/configuration.
LATITUDE = 28.04996394302817
ELEVATION = 339.73


# Development values for the Angstrom equation.
# These are not Rajasthan-calibrated coefficients.
ANGSTA = 0.25
ANGSTB = 0.50


def load_raw_weather():
    """Load the raw NASA POWER weather JSON response."""

    with RAW_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def load_relative_humidity():
    """Load the raw NASA POWER relative-humidity JSON response."""

    with RH_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def main():
    """Prepare NASA POWER weather data for PCSE/WOFOST."""

    data = load_raw_weather()
    rh_data = load_relative_humidity()

    parameters = data["properties"]["parameter"]
    rh_parameters = rh_data["properties"]["parameter"]

    dates = list(parameters["T2M"].keys())

    weather_data = []

    for date in dates:
        # Relative humidity at 2 m (%)
        rh = rh_parameters["RH2M"][date]

        # Daily mean temperature (Celsius)
        temp = parameters["T2M"][date]

        # Convert relative humidity + temperature
        # to actual vapour pressure.
        #
        # PCSE returns vapour pressure in kPa,
        # while the PCSE weather container expects hPa.
        vap_kpa = vap_from_relhum(rh, temp)
        vap_hpa = vap_kpa * 10.0

        day = pd.to_datetime(date, format="%Y%m%d")

        # NASA POWER -> PCSE input units
        tmin = parameters["T2M_MIN"][date]
        tmax = parameters["T2M_MAX"][date]

        # NASA POWER: mm/day
        # PCSE weather: cm/day
        rain_cm = parameters["PRECTOTCORR"][date] * 0.1

        # NASA POWER: MJ/m2/day
        # PCSE: J/m2/day
        irrad = parameters["ALLSKY_SFC_SW_DWN"][date] * 1_000_000

        # NASA POWER and PCSE: m/sec
        wind = parameters["WS2M"][date]

        # Calculate reference evapotranspiration.
        #
        # PCSE returns E0, ES0 and ET0 in mm/day.
        e0_mm, es0_mm, et0_mm = reference_ET(
            DAY=day.date(),
            LAT=LATITUDE,
            ELEV=ELEVATION,
            TMIN=tmin,
            TMAX=tmax,
            IRRAD=irrad,
            VAP=vap_hpa,
            WIND=wind,
            ANGSTA=ANGSTA,
            ANGSTB=ANGSTB,
        )

        weather_data.append(
            {
                "DAY": day,
                "TMIN": tmin,
                "TMAX": tmax,
                "RAIN": rain_cm,
                "IRRAD": irrad,
                "VAP": vap_hpa,
                "WIND": wind,

                # PCSE weather format uses cm/day.
                "E0": e0_mm * 0.1,
                "ES0": es0_mm * 0.1,
                "ET0": et0_mm * 0.1,
            }
        )

    df = pd.DataFrame(weather_data)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    print("VHR-YieldNet NASA POWER weather preparation")
    print()
    print(f"Input records: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Processed weather data:")
    print()
    print(df.head(10).to_string(index=False))
    print()
    print("Columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()