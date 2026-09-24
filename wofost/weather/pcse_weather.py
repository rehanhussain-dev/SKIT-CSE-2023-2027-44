"""
PCSE weather provider for VHR-YieldNet.

Reads a processed PCSE-compatible weather CSV and stores each
daily record as a WeatherDataContainer.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from pcse.base import WeatherDataContainer
from pcse.base import WeatherDataProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEATHER_FILE = (
    PROJECT_ROOT
    / "wofost"
    / "weather"
    / "processed"
    / "nasa_power_wheat_2025_26.csv"
)


class VHRWeatherProvider(WeatherDataProvider):
    """Weather provider for the VHR-YieldNet test field."""

    def __init__(self, weather_file=WEATHER_FILE):
        super().__init__()

        self.latitude = 28.04996394302817
        self.longitude = 75.12877653515253
        self.elevation = 339.73

        self.description = [
            "VHR-YieldNet development weather dataset",
            "Source: NASA POWER",
            "Crop: wheat",
        ]

        self.ETmodel = "PM"

        weather_file = Path(weather_file)

        if not weather_file.exists():
            raise FileNotFoundError(
                f"Weather file not found: {weather_file}"
            )

        df = pd.read_csv(weather_file)

        required_columns = [
            "DAY",
            "TMIN",
            "TMAX",
            "RAIN",
            "IRRAD",
            "VAP",
            "WIND",
            "E0",
            "ES0",
            "ET0",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing weather columns: {missing_columns}"
            )

        for _, row in df.iterrows():
            day = datetime.strptime(
                row["DAY"],
                "%Y-%m-%d",
            ).date()

            weather = WeatherDataContainer(
                DAY=day,
                LAT=self.latitude,
                LON=self.longitude,
                ELEV=self.elevation,
                TMIN=float(row["TMIN"]),
                TMAX=float(row["TMAX"]),
                VAP=float(row["VAP"]),
                RAIN=float(row["RAIN"]),
                E0=float(row["E0"]),
                ES0=float(row["ES0"]),
                ET0=float(row["ET0"]),
                IRRAD=float(row["IRRAD"]),
                WIND=float(row["WIND"]),
                ANGSTA=0.25,
                ANGSTB=0.50,
            )

            self._store_WeatherDataContainer(
                weather,
                day,
            )


if __name__ == "__main__":
    provider = VHRWeatherProvider()

    print(provider)
    print()
    print("First weather record:")
    print(provider(provider.first_date))
    print()
    print("Last weather record:")
    print(provider(provider.last_date))