import csv
from pathlib import Path

from pcse.base import ParameterProvider
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


def main():
    print("=== VHR-YieldNet WOFOST wheat simulation ===")

    # Crop
    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", "Winter_wheat_107")

    # Soil and site
    soil = load_soil()
    site = load_site()

    # Parameter provider
    params = ParameterProvider(
        cropdata=crop,
        soildata=soil,
        sitedata=site
    )

    # Weather
    weather = VHRWeatherProvider()

    # Model
    model = Wofost72_WLP_CWB(
        params,
        weather,
        AGROMANAGEMENT
    )

    print("Simulation start:", model.day)

    # Run simulation
    model.run_till_terminate()

    output = model.get_output()

    print("Simulation finished.")
    print("Output records:", len(output))

    if output:
        print("\nFirst record:")
        print(output[0])

        print("\nLast record:")
        print(output[-1])

        # Save daily output
        output_dir = Path("wofost") / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "wheat_2025_26_daily.csv"

        fieldnames = list(output[0].keys())

        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)

        print("\nDaily output saved to:")
        print(output_file)


if __name__ == "__main__":
    main()