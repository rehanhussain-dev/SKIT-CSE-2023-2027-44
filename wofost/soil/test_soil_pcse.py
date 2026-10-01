from pcse.base import ParameterProvider

from soil_provider import load_soil


def main():
    soil = load_soil()

    print("Testing WOFOST soil parameters with PCSE")
    print("-----------------------------------------")

    for key, value in soil.items():
        print(f"{key}: {value}")

    # ParameterProvider accepts soil parameters as a dictionary.
    params = ParameterProvider(soildata=soil)

    print("\nPCSE ParameterProvider accepted the soil.")
    print("Soil parameters loaded successfully.")


if __name__ == "__main__":
    main()