"""
Weather preprocessing for VHR-YieldNet WOFOST module.

This module defines the weather variables required by PCSE/WOFOST
and provides basic validation of a prepared weather dataset.
"""

import pandas as pd


# Required weather variables for the PCSE weather interface.
REQUIRED_COLUMNS = [
    "DAY",
    "IRRAD",
    "TMIN",
    "TMAX",
    "VAP",
    "RAIN",
    "E0",
    "ES0",
    "ET0",
    "WIND",
]


# Expected units used by PCSE.
EXPECTED_UNITS = {
    "IRRAD": "J/m2/day",
    "TMIN": "Celsius",
    "TMAX": "Celsius",
    "VAP": "hPa",
    "RAIN": "cm/day",
    "E0": "cm/day",
    "ES0": "cm/day",
    "ET0": "cm/day",
    "WIND": "m/sec",
}


def validate_weather_dataframe(df: pd.DataFrame) -> None:
    """
    Check that a weather DataFrame contains all required PCSE columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared weather data.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required weather columns: {missing_columns}"
        )


if __name__ == "__main__":
    print("VHR-YieldNet WOFOST weather module")
    print()
    print("Required PCSE weather variables:")
    
    for column in REQUIRED_COLUMNS:
        if column == "DAY":
            print(f"- {column}: date")
        else:
            print(f"- {column}: {EXPECTED_UNITS[column]}")