
from copy import deepcopy
from pathlib import Path

import pandas as pd

from pcse.models import Wofost72_PP

from run_wheat_test import (
    ParameterProvider,
    YAMLCropDataProvider,
    load_soil,
    load_site,
    VHRWeatherProvider,
    AGROMANAGEMENT,
    Wofost72_WLP_CWB,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CULTIVAR = "Winter_wheat_107"


def run_model(model_class):

    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", CULTIVAR)

    params = ParameterProvider(
        cropdata=crop,
        soildata=load_soil(),
        sitedata=load_site(),
    )

    model = model_class(
        params,
        VHRWeatherProvider(),
        deepcopy(AGROMANAGEMENT),
    )

    model.run_till_terminate()

    df = pd.DataFrame(model.get_output())

    if df.empty:
        raise RuntimeError("No simulation output.")

    df["day"] = pd.to_datetime(df["day"])

    if df["day"].duplicated().any():
        raise ValueError("Duplicate simulation dates.")

    if df["DVS"].iloc[-1] < 2.0:
        raise ValueError("Simulation did not reach maturity.")

    return df


def summarize(mode, df):

    anthesis = df[df["DVS"] >= 1.0].iloc[0]
    final = df.iloc[-1]

    grain_filling = df[df["DVS"] >= 1.0]

    return {
        "production_mode": mode,
        "cultivar": CULTIVAR,
        "simulation_days": len(df),
        "anthesis_date": anthesis["day"].date(),
        "maturity_date": final["day"].date(),
        "max_LAI": df["LAI"].max(),
        "LAI_at_anthesis": anthesis["LAI"],
        "mean_grain_filling_LAI": grain_filling["LAI"].mean(),
        "TAGP_at_anthesis": anthesis["TAGP"],
        "TAGP_at_maturity": final["TAGP"],
        "TWSO_at_anthesis": anthesis["TWSO"],
        "final_TWSO": final["TWSO"],
        "grain_biomass_gain": (
            final["TWSO"] - anthesis["TWSO"]
        ),
        "harvest_index": (
            final["TWSO"] / final["TAGP"]
        ),
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "potential": Wofost72_PP,
        "water_limited": Wofost72_WLP_CWB,
    }

    results = {}
    summaries = []

    for mode, model_class in models.items():

        print("\nRunning:", mode)

        df = run_model(model_class)

        results[mode] = df

        summaries.append(summarize(mode, df))

        df.to_csv(
            OUTPUT_DIR / f"production_{mode}_daily.csv",
            index=False,
        )

    summary = pd.DataFrame(summaries)

    summary.to_csv(
        OUTPUT_DIR / "production_mode_comparison.csv",
        index=False,
    )

    print("\nPRODUCTION MODE SUMMARY:")
    print(summary.T.to_string())

    pp = results["potential"]
    wlp = results["water_limited"]

    columns = ["day", "DVS", "LAI", "TAGP", "TWSO"]

    comparison = pp[columns].merge(
        wlp[columns],
        on="day",
        suffixes=("_PP", "_WLP"),
        validate="one_to_one",
    )

    print("\nGRAIN-FILLING COMPARISON:")

    grain = comparison[
        comparison["DVS_PP"] >= 1.0
    ]

    print(
        grain.round(4).to_string(index=False)
    )

    print("\nProduction mode comparison: PASS")
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
