
from copy import deepcopy
from pathlib import Path

import pandas as pd

from pcse.models import Wofost72_PP, Wofost72_WLP_CWB

from run_wheat_test import (
    ParameterProvider,
    YAMLCropDataProvider,
    load_soil,
    load_site,
    VHRWeatherProvider,
    AGROMANAGEMENT,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CULTIVAR = "Winter_wheat_107"


def run_diagnostic(model_class, mode):

    records = []

    class DiagnosticModel(model_class):

        def calc_rates(self, day, drv):

            super().calc_rates(day, drv)

            if self.crop is None:
                return

            crop = self.crop

            row = {
                "day": day,
                "production_mode": mode,
                "TEMP": drv.TEMP,
                "IRRAD": drv.IRRAD,
            }

            # Current crop states.
            for name in [
                "DVS", "LAI", "TAGP", "TWSO",
                "WRT", "WLV", "WST", "WSO"
            ]:
                row[name] = crop.get_variable(name)

            # Assimilation and dry matter production.
            rates = crop.rates._trait_values

            for name in [
                "GASS", "MRES", "ASRC", "DMI", "ADMI"
            ]:
                row[name] = rates.get(
                    name, float("nan")
                )

            # Potential maintenance respiration.
            row["PMRES"] = (
                crop.mres.rates._trait_values.get(
                    "PMRES", float("nan")
                )
            )

            # Grain growth rate.
            row["GRSO"] = (
                crop.so_dynamics.rates._trait_values.get(
                    "GRSO", float("nan")
                )
            )

            # Leaf death rates.
            leaf_rates = (
                crop.lv_dynamics.rates._trait_values
            )

            for name in ["DALV", "DRLV", "GRLV"]:
                row[name] = leaf_rates.get(
                    name, float("nan")
                )

            records.append(row)

    crop_data = YAMLCropDataProvider()
    crop_data.set_active_crop("wheat", CULTIVAR)

    params = ParameterProvider(
        cropdata=crop_data,
        soildata=load_soil(),
        sitedata=load_site(),
    )

    model = DiagnosticModel(
        params,
        VHRWeatherProvider(),
        deepcopy(AGROMANAGEMENT),
    )

    model.run_till_terminate()

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            f"No diagnostic records for {mode}"
        )

    df["day"] = pd.to_datetime(df["day"])

    if df["day"].duplicated().any():
        raise ValueError(
            f"Duplicate diagnostic dates for {mode}"
        )

    return df


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "potential": Wofost72_PP,
        "water_limited": Wofost72_WLP_CWB,
    }

    for mode, model_class in models.items():

        print("\nRunning diagnostic:", mode)

        df = run_diagnostic(model_class, mode)

        output_file = (
            OUTPUT_DIR
            / f"production_{mode}_diagnostic.csv"
        )

        df.to_csv(output_file, index=False)

        print("Records:", len(df))

        period = df[
            (df["day"] >= "2026-04-07")
            & (df["day"] <= "2026-04-16")
        ]

        columns = [
            "day", "DVS", "LAI", "TEMP",
            "GASS", "PMRES", "MRES",
            "ASRC", "DMI", "GRSO", "DALV"
        ]

        print("\nASSIMILATION AND RESPIRATION:")

        print(
            period[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        grain = df[df["DVS"] >= 1.0]

        print("\nGRAIN-FILLING DIAGNOSTICS:")

        print("Recorded days:", len(grain))

        print(
            "Missing ASRC values:",
            grain["ASRC"].isna().sum(),
        )

        print(
            "Days with zero available assimilates:",
            grain["ASRC"].abs().le(1e-9).sum(),
        )

        print(
            "Days with positive grain growth:",
            grain["GRSO"].gt(1e-9).sum(),
        )

        print("Saved:", output_file)

    print("\nProduction diagnostics completed.")


if __name__ == "__main__":
    main()
