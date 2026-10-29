import csv
from pathlib import Path

from pcse.base import ParameterProvider
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


def run_scenario(wav):
    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", "Winter_wheat_107")

    soil = load_soil()
    site = load_site()
    site["WAV"] = wav

    params = ParameterProvider(
        cropdata=crop,
        soildata=soil,
        sitedata=site,
    )

    model = Wofost72_WLP_CWB(
        params,
        VHRWeatherProvider(),
        AGROMANAGEMENT,
    )
    model.run_till_terminate()
    output = model.get_output()

    if not output:
        raise RuntimeError(f"No output for WAV={wav}")

    final = output[-1]
    return {
        "WAV": wav,
        "final_day": final["day"],
        "final_TAGP_kg_ha": final["TAGP"],
        "final_TWSO_kg_ha": final["TWSO"],
        "minimum_RFTRA": min(row["RFTRA"] for row in output),
        "mean_RFTRA": sum(row["RFTRA"] for row in output) / len(output),
    }


def main():
    results = [
        run_scenario(49.2),
        run_scenario(20.0),
    ]

    output_file = (
        Path(__file__).resolve().parent
        / "outputs"
        / "soil_water_sensitivity.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(row)

    print("Saved to:", output_file)


if __name__ == "__main__":
    main()