
import csv
from copy import deepcopy
from pathlib import Path

from pcse.base import ParameterProvider
from pcse.exceptions import WeatherDataProviderError
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


# Cultivars selected for controlled sensitivity comparison.
CULTIVARS = [
    "Winter_wheat_107",
    "Winter_wheat_106",
    "apache",
]

# Maturity criterion used for this experiment.
MATURITY_DVS = 2.0

# Save results inside wofost/outputs.
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cultivar(name):
    """Run one cultivar and preserve output if weather coverage ends."""

    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", name)

    soil = load_soil()
    site = load_site()

    params = ParameterProvider(
        cropdata=crop,
        soildata=soil,
        sitedata=site
    )

    agro = deepcopy(AGROMANAGEMENT)

    campaign = agro["AgroManagement"][0]
    calendar = next(iter(campaign.values()))["CropCalendar"]
    calendar["variety_name"] = name

    weather = VHRWeatherProvider()

    model = Wofost72_WLP_CWB(
        params,
        weather,
        agro
    )

    weather_exhausted = False

    try:
        model.run_till_terminate()

    except WeatherDataProviderError:
        weather_exhausted = True
        print(
            "Weather data exhausted; "
            "retaining available daily output."
        )

    output = model.get_output()

    if not output:
        raise RuntimeError(
            f"No daily output generated for cultivar: {name}"
        )

    return output, weather_exhausted


def create_summary(name, output, weather_exhausted):
    """Calculate cultivar performance without mislabelling maturity."""

    anthesis_date = next(
        (
            row["day"]
            for row in output
            if row["DVS"] >= 1.0
        ),
        None
    )

    maturity_date = next(
        (
            row["day"]
            for row in output
            if row["DVS"] >= MATURITY_DVS
        ),
        None
    )

    final_row = output[-1]

    if maturity_date is not None:
        status = "matured"
    elif weather_exhausted:
        status = "incomplete_weather_coverage"
    else:
        status = "not_matured"

    summary = {
        "cultivar": name,
        "simulation_status": status,
        "anthesis_proxy_date": anthesis_date,
        "maturity_date": maturity_date,
        "last_simulated_day": final_row["day"],
        "last_DVS": final_row["DVS"],
        "max_LAI_to_date": max(
            row["LAI"] for row in output
        ),
        "TAGP_at_last_day": final_row["TAGP"],
        "TWSO_at_last_day": final_row["TWSO"],
        "mean_RFTRA_to_last_day": sum(
            row["RFTRA"] for row in output
        ) / len(output),
        "minimum_RFTRA_to_last_day": min(
            row["RFTRA"] for row in output
        ),
        "weather_exhausted": weather_exhausted,
    }

    return summary


def main():

    summaries = []
    daily_records = []

    for name in CULTIVARS:

        print(f"\nRunning cultivar: {name}")

        output, weather_exhausted = run_cultivar(name)

        summary = create_summary(
            name,
            output,
            weather_exhausted
        )

        summaries.append(summary)

        for row in output:
            daily_records.append({
                "cultivar": name,
                **row
            })

        print(summary)

    # Write summary CSV.
    summary_file = OUTPUT_DIR / "cultivar_sensitivity.csv"

    with summary_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summaries[0].keys()
        )

        writer.writeheader()
        writer.writerows(summaries)

    # Write daily simulation CSV.
    daily_file = OUTPUT_DIR / "cultivar_sensitivity_daily.csv"

    with daily_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=daily_records[0].keys()
        )

        writer.writeheader()
        writer.writerows(daily_records)

    print("\n" + "=" * 55)
    print("CULTIVAR SENSITIVITY EXPERIMENT COMPLETED")
    print("=" * 55)

    print("\nSummary:")
    for item in summaries:
        print(
            f"{item['cultivar']}: "
            f"{item['simulation_status']}, "
            f"last DVS={item['last_DVS']:.4f}"
        )

    print("\nOutput files:")
    print(summary_file)
    print(daily_file)


if __name__ == "__main__":
    main()
