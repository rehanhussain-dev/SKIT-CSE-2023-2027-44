
import csv
from pathlib import Path

from pcse.base import ParameterProvider
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


SCENARIOS = [20.0, 30.0, 37.95, 49.2]


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

    daily_records = []

    for row in output:
        daily_records.append({
            "WAV": wav,
            "day": row["day"],
            "DVS": row["DVS"],
            "LAI": row["LAI"],
            "SM": row["SM"],
            "WWLOW": row["WWLOW"],
            "RD": row["RD"],
            "TRA": row["TRA"],
            "RFTRA": row["RFTRA"],
            "TAGP": row["TAGP"],
            "TWSO": row["TWSO"],
        })

    final = output[-1]

    summary = {
        "WAV": wav,
        "final_day": final["day"],
        "final_TAGP_kg_ha": final["TAGP"],
        "final_TWSO_kg_ha": final["TWSO"],
        "final_LAI": final["LAI"],
        "final_SM": final["SM"],
        "final_WWLOW": final["WWLOW"],
        "minimum_RFTRA": min(row["RFTRA"] for row in output),
        "mean_RFTRA": sum(
            row["RFTRA"] for row in output
        ) / len(output),
    }

    return summary, daily_records


def main():

    all_summaries = []
    all_daily_records = []

    for wav in SCENARIOS:

        print(f"\nRunning scenario: WAV={wav} mm")

        summary, daily_records = run_scenario(wav)

        all_summaries.append(summary)
        all_daily_records.extend(daily_records)

        print(summary)

    output_dir = (
        Path(__file__).resolve().parent / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scenario summary
    summary_file = output_dir / "soil_water_sensitivity.csv"

    with summary_file.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=all_summaries[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    # Save daily scenario results
    daily_file = (
        output_dir / "soil_water_sensitivity_daily.csv"
    )

    with daily_file.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=all_daily_records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(all_daily_records)

    print("\nSensitivity experiment completed.")
    print("Scenarios:", len(SCENARIOS))
    print("Summary:", summary_file)
    print("Daily results:", daily_file)


if __name__ == "__main__":
    main()
