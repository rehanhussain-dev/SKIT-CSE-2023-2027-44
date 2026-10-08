from pcse.base import ParameterProvider
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


def main():
    print("=== VHR-YieldNet WOFOST integration test ===")

    # 1. Crop parameters
    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", "Winter_wheat_107")

    print("Crop: wheat")
    print("Variety: Winter_wheat_107")
    print("TSUM1:", crop["TSUM1"])
    print("TSUM2:", crop["TSUM2"])

    # 2. Soil parameters
    soil = load_soil()

    print("\nSoil:")
    for key, value in soil.items():
        print(f"  {key}: {value}")

    # 3. Site parameters
    site = load_site()

    print("\nSite:")
    for key, value in site.items():
        print(f"  {key}: {value}")

    # 4. Combine crop, soil and site parameters
    params = ParameterProvider(
        cropdata=crop,
        soildata=soil,
        sitedata=site
    )

    # 5. Weather
    weather = VHRWeatherProvider()

    print("\nWeather:")
    print("  first:", weather.first_date)
    print("  last:", weather.last_date)

    # 6. WOFOST model initialization
    model = Wofost72_WLP_CWB(
        params,
        weather,
        AGROMANAGEMENT
    )

    print("\nSUCCESS: WOFOST model initialized.")
    print("Model start date:", model.day)


if __name__ == "__main__":
    main()