
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        super().calc_rates(day, drv)
        if day.year == 2026 and day.month == 4 and day.day == 8:
           print("Rate object type:", type(self.crop.rates))
           print("Stored rate values:", self.crop.rates._trait_values)
     
        if self.crop is None:
            return

        row = {"day": day}
        for name in [
            "DVS", "FO", "ADMI", "GRSO",
            "GASS", "MRES", "ASRC", "DMI",
            "RFTRA", "TWSO", "TAGP"
        ]:
            if name in {"MRES", "ASRC", "GASS", "DMI", "ADMI"}:
               row[name] = self.crop.rates._trait_values.get(
               name, float("nan") )
            else:
               row[name] = self.crop.get_variable(name)

        type(self).daily_rates.append(row)


def main():
    DiagnosticWofost.daily_rates = []

    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", "Winter_wheat_107")

    params = ParameterProvider(
        cropdata=crop,
        soildata=load_soil(),
        sitedata=load_site(),
    )

    model = DiagnosticWofost(
        params,
        VHRWeatherProvider(),
        AGROMANAGEMENT,
    )
    model.run_till_terminate()

    output_file = (
        Path(__file__).resolve().parent
        / "outputs"
        / "grain_growth_diagnostic.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "day", "DVS", "FO", "ADMI", "GRSO",
        "GASS", "RFTRA", "TWSO", "TAGP","MRES", "ASRC", "DMI", "PGASS"
    ]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(DiagnosticWofost.daily_rates)

    print("Diagnostic completed.")
    print("Days recorded:", len(DiagnosticWofost.daily_rates))
    print("Saved to:", output_file)
    if DiagnosticWofost.daily_rates:
        print("Final row:", DiagnosticWofost.daily_rates[-1])


if __name__ == "__main__":
    main()
