import json
from pathlib import Path


SOIL_FILE = Path(__file__).parent / "rajasthan_sandy_clay_loam.json"


def load_soil():
    """Load the VHR-YieldNet representative soil configuration."""

    with open(SOIL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["wofost_parameters"]


if __name__ == "__main__":
    soil = load_soil()

    print("VHR-YieldNet soil parameters")
    print("-----------------------------")

    for name, value in soil.items():
        print(f"{name}: {value}")