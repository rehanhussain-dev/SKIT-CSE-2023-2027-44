
import csv
import copy
from datetime import date
from pathlib import Path

from pcse.base import ParameterProvider
from pcse.input.yaml_cropdataprovider import YAMLCropDataProvider
from pcse.models import Wofost72_WLP_CWB

from wofost_site.site_provider import load_site
from soil.soil_provider import load_soil
from weather.pcse_weather import VHRWeatherProvider
from agromanagement.wheat_2025_26 import AGROMANAGEMENT


OUTPUT_DIR = Path("wofost") / "outputs"

IRRIGATION_EFFICIENCY = 0.80

SCENARIOS = {
    "S0_rainfed": [],
    "S1_single_irrigation": [
        (date(2026, 3, 3), 3.0)
    ],
    "S2_double_irrigation": [
        (date(2026, 3, 3), 3.0),
        (date(2026, 3, 20), 3.0)
    ],
}


def build_agromanagement(irrigations):
    agro = copy.deepcopy(AGROMANAGEMENT)

    if not irrigations:
        return agro

    events = [
        {
            event_date: {
                "amount": amount_cm,
                "efficiency": IRRIGATION_EFFICIENCY
            }
        }
        for event_date, amount_cm in irrigations
    ]

    agro["AgroManagement"][0][date(2025, 11, 15)][
        "TimedEvents"
    ] = [
        {
            "event_signal": "irrigate",
            "name": "VHR-YieldNet irrigation experiment",
            "comment": "Irrigation amounts in cm",
            "events_table": events
        }
    ]

    return agro


def run_scenario(name, irrigations):
    print(f"\nRunning {name}...")

    crop = YAMLCropDataProvider()
    crop.set_active_crop("wheat", "Winter_wheat_107")

    params = ParameterProvider(
        cropdata=crop,
        soildata=load_soil(),
        sitedata=load_site()
    )

    weather = VHRWeatherProvider()
    agro = build_agromanagement(irrigations)

    model = Wofost72_WLP_CWB(params, weather, agro)

    model.run_till_terminate()
    output = model.get_output()

    if not output:
        raise RuntimeError(f"No output produced for {name}")

    daily_file = OUTPUT_DIR / f"{name}_daily.csv"

    with daily_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    final = output[-1]

    grain_filling = [
        row for row in output
        if 1.0 <= float(row["DVS"]) <= 2.0
    ]

    mean_rftra = (
        sum(float(row["RFTRA"]) for row in grain_filling)
        / len(grain_filling)
        if grain_filling else None
    )

    maturity_rows = [
        row for row in output
        if float(row["DVS"]) >= 2.0
    ]

    maturity_date = maturity_rows[-1]["day"] if maturity_rows else None

    result = {
        "scenario": name,
        "simulation_days": len(output),
        "last_simulated_day": str(final["day"]),
        "maturity_date": str(maturity_date),
        "final_DVS": final["DVS"],
        "final_TAGP": final["TAGP"],
        "final_TWSO": final["TWSO"],
        "mean_grain_filling_RFTRA": mean_rftra,
        "irrigation_events": len(irrigations),
        "gross_irrigation_cm": sum(x[1] for x in irrigations),
        "effective_irrigation_cm": (
            sum(x[1] for x in irrigations) * IRRIGATION_EFFICIENCY
        )
    }

    print(result)
    print("Daily output:", daily_file)

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name, irrigations in SCENARIOS.items():
        results.append(run_scenario(name, irrigations))

    summary_file = OUTPUT_DIR / "irrigation_sensitivity_summary.csv"

    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== IRRIGATION SENSITIVITY SUMMARY ===")

    for result in results:
        print(result)

    print("\nSummary saved to:", summary_file)


if __name__ == "__main__":
    main()
