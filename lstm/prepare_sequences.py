from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "wofost"
    / "outputs"
    / "wheat_2025_26_daily.csv"
)

OUTPUT_FILE = PROJECT_ROOT / "lstm" / "wofost_sequences.pt"

SEQUENCE_LENGTH = 30

# Daily WOFOST growth and water-status variables.
FEATURE_COLUMNS = [
    "DVS",
    "LAI",
    "TAGP",
    "TWLV",
    "TWST",
    "TWRT",
    "SM",
    "RFTRA",
]


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["day"])
    df = df.sort_values("day").reset_index(drop=True)

    missing_columns = [
        col for col in FEATURE_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    features = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    )

    if features.isna().any().any():
        raise ValueError("Feature data contains missing values.")

    if not features.apply(
        lambda column: column.map(lambda value: abs(value) != float("inf"))
    ).all().all():
        raise ValueError("Feature data contains infinite values.")

    # Standardize each feature for numerical stability.
    means = features.mean()
    stds = features.std(ddof=0).replace(0, 1)
    normalized = (features - means) / stds

    values = torch.tensor(
        normalized.to_numpy(),
        dtype=torch.float32,
    )

    sequences = values.unfold(
        dimension=0,
        size=SEQUENCE_LENGTH,
        step=1,
    ).permute(0, 2, 1).contiguous()

    torch.save(
        {
            "sequences": sequences,
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "normalization_mean": means.to_dict(),
            "normalization_std": stds.to_dict(),
            "source": str(INPUT_FILE),
            "data_type": "simulated WOFOST outputs; not satellite observations",
        },
        OUTPUT_FILE,
    )

    print("WOFOST sequence preparation: PASS")
    print("Source records:", len(df))
    print("Features:", FEATURE_COLUMNS)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Number of sequences:", sequences.shape[0])
    print("Tensor shape:", tuple(sequences.shape))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()