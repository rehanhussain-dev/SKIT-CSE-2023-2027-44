"""
Convert NASA POWER daily weather data into a PCSE-friendly dataframe.

Development/test weather source for VHR-YieldNet.
Final project weather should use the specified IMD data where available.
"""

import json
from pathlib import Path

import pandas as pd

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

        weather_data.append(
            {
                "DAY": pd.to_datetime(date, format="%Y%m%d"),
                "TMIN": parameters["T2M_MIN"][date],
                "TMAX": parameters["T2M_MAX"][date],

                # NASA POWER: mm/day
                # PCSE: cm/day
                "RAIN": parameters["PRECTOTCORR"][date] * 0.1,

                # NASA POWER: MJ/m2/day
                # PCSE: J/m2/day
                "IRRAD": (
                    parameters["ALLSKY_SFC_SW_DWN"][date]
                    * 1_000_000
                ),

                # NASA POWER and PCSE: m/sec
                "WIND": parameters["WS2M"][date],

                # PCSE requires hPa
                "VAP": vap_hpa,
            }
        )

    df = pd.DataFrame(weather_data)

    print("VHR-YieldNet NASA POWER weather preparation")
    print()
    print("Converted weather data:")
    print()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()