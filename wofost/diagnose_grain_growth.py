
import sys
import csv
from pathlib import Path

# Ensure the WOFOST project directory is available for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from pcse.base import ParameterProvider
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


class DiagnosticWofost(Wofost72_WLP_CWB):

    daily_rates = []

    def calc_rates(self, day, drv):

        # Calculate normal WOFOST daily rates first
        super().calc_rates(day, drv)

        if self.crop is None:
            return

        # Initialize daily diagnostic record
        row = {"day": day}

        # -------------------------------------------------
        # 1. Weather and maintenance respiration
        # -------------------------------------------------

        row["TEMP"] = drv.TEMP

        row["PMRES"] = self.crop.mres.rates._trait_values.get(
            "PMRES", float("nan")
        )

        # -------------------------------------------------
        # 2. Plant organ biomass
        # -------------------------------------------------

        for name in ["WRT", "WLV", "WST", "WSO"]:
            row[name] = self.crop.get_variable(name)

        # -------------------------------------------------
        # 3. Leaf cohort age and biomass diagnostics
        # -------------------------------------------------

        leaf_states = self.crop.lv_dynamics.states

        leaf_ages = list(leaf_states.LVAGE)
        leaf_biomass = list(leaf_states.LV)

        span = self.crop.lv_dynamics.params.SPAN

        aged_cohorts = [
            (biomass, age)
            for biomass, age in zip(leaf_biomass, leaf_ages)
            if age > span
        ]

        row["SPAN"] = span
        row["LEAF_COHORTS"] = len(leaf_ages)

        row["OLDEST_LEAF_AGE"] = (
            max(leaf_ages) if leaf_ages else 0.0
        )

        row["YOUNGEST_LEAF_AGE"] = (
            min(leaf_ages) if leaf_ages else 0.0
        )

        row["AGED_COHORTS"] = len(aged_cohorts)

        row["AGED_LV_BIOMASS"] = sum(
            biomass for biomass, age in aged_cohorts
        )

        # Extract daily leaf loss rates
        leaf_rates = self.crop.lv_dynamics.rates._trait_values

        for name in ["DSLV1", "DSLV2", "DSLV3", "DALV", "DRLV", "FYSAGE"]:
            row[name] = leaf_rates.get(name, float("nan"))

        row["GRLV"] = self.crop.lv_dynamics.rates._trait_values.get("GRLV", float("nan"))

        # Root growth, death and accumulated dead-root biomass
        root_rates = self.crop.ro_dynamics.rates._trait_values
        root_states = self.crop.ro_dynamics.states._trait_values
        row["GRRT"] = root_rates.get("GRRT", float("nan"))
        row["DRRT"] = root_rates.get("DRRT", float("nan"))
        row["DWRT"] = root_states.get("DWRT", float("nan"))

        # -------------------------------------------------
        # 4. Crop development and allocation variables
        # -------------------------------------------------

        for name in [
            "DVS",
            "FO",
            "ADMI",
            "GRSO",
            "GASS",
            "MRES",
            "ASRC",
            "DMI",
            "RFTRA",
            "TWSO",
            "TAGP",
        ]:

            # Internal daily rate variables
                        # Internal daily rate variables
            if name == "GRSO":
                row[name] = self.crop.so_dynamics.rates._trait_values.get(
                    "GRSO", float("nan")
                )

            elif name in {
                "MRES",
                "ASRC",
                "GASS",
                "DMI",
                "ADMI",
            }:
                row[name] = self.crop.rates._trait_values.get(
                    name, float("nan")
                )

            # Published crop variables
            else:
                row[name] = self.crop.get_variable(name)

        # Store the daily record
        type(self).daily_rates.append(row)


def main():

    DiagnosticWofost.daily_rates = []

    # -------------------------------------------------
    # 1. Load crop parameters
    # -------------------------------------------------

    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", "Winter_wheat_107")

    # -------------------------------------------------
    # 2. Load soil and site parameters
    # -------------------------------------------------

    soil = load_soil()
    site = load_site()

    params = ParameterProvider(
        cropdata=crop,
        soildata=soil,
        sitedata=site,
    )

    # -------------------------------------------------
    # 3. Initialize WOFOST model
    # -------------------------------------------------

    model = DiagnosticWofost(
        params,
        VHRWeatherProvider(),
        AGROMANAGEMENT,
    )

    # -------------------------------------------------
    # 4. Run simulation
    # -------------------------------------------------

    model.run_till_terminate()

    # -------------------------------------------------
    # 5. Save diagnostic output
    # -------------------------------------------------

    output_file = (
        BASE_DIR
        / "outputs"
        / "grain_growth_diagnostic.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "day",
        "DVS",
        "FO",
        "ADMI",
        "GRLV",
        "GRRT",
        "DRRT",
        "DWRT",
        "GRSO",
        "GASS",
        "MRES",
        "PMRES",
        "ASRC",
        "DMI",
        "RFTRA",
        "TWSO",
        "TAGP",
        "TEMP",
        "WRT",
        "WLV",
        "WST",
        "WSO",
        "DSLV1",
        "DSLV2",
        "DSLV3",
        "DALV",
        "DRLV",
        "FYSAGE",
        "SPAN",
        "LEAF_COHORTS",
        "OLDEST_LEAF_AGE",
        "YOUNGEST_LEAF_AGE",
        "AGED_COHORTS",
        "AGED_LV_BIOMASS",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(DiagnosticWofost.daily_rates)

    # -------------------------------------------------
    # 6. Print simulation summary
    # -------------------------------------------------

    print("\nWOFOST grain-growth diagnostic completed.")

    print(
        "Days recorded:",
        len(DiagnosticWofost.daily_rates),
    )

    print(
        "Saved to:",
        output_file,
    )

    if DiagnosticWofost.daily_rates:

        print("\nFinal diagnostic record:")

        print(
            DiagnosticWofost.daily_rates[-1]
        )


if __name__ == "__main__":
    main()
