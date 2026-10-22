
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "s2_vegetation_indices_timeseries.csv"
OUTPUT_FILE = PROJECT_ROOT / "lstm" / "satellite_sequences.pt"
METADATA_FILE = PROJECT_ROOT / "lstm" / "satellite_sequences_metadata.csv"

SEQUENCE_LENGTH = 14
MIN_VALID_PIXELS = 500

# Start with mean vegetation indices as model features.
FEATURE_COLUMNS = ["ndvi_mean", "ndre_mean", "si_mean"]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Satellite CSV not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "date", "season", "valid_pixels", "source_file",
        *FEATURE_COLUMNS,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["valid_pixels"] = pd.to_numeric(
        df["valid_pixels"], errors="raise"
    )

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[required].isna().any().any():
        raise ValueError("Required fields contain missing values.")

    if df.duplicated(["season", "date"]).any():
        raise ValueError("Duplicate date within a season detected.")

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Feature values contain infinity or NaN.")

    # Preserve all original rows in the source CSV.
    # Exclude low-valid-pixel observations only from this derived dataset.
    rejected = df[df["valid_pixels"] < MIN_VALID_PIXELS].copy()
    accepted = df[df["valid_pixels"] >= MIN_VALID_PIXELS].copy()

    accepted = accepted.sort_values(
        ["season", "date"]
    ).reset_index(drop=True)

    if accepted.empty:
        raise ValueError("No observations remain after quality filtering.")

    # Normalize using accepted observations. Save these parameters for reuse.
    means = accepted[FEATURE_COLUMNS].mean()
    stds = accepted[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)
    normalized = (accepted[FEATURE_COLUMNS] - means) / stds

    accepted[FEATURE_COLUMNS] = normalized

    sequences = []
    metadata = []

    # Never create a sequence that crosses a season boundary.
    for season, group in accepted.groupby("season", sort=True):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) < SEQUENCE_LENGTH:
            continue

        values = torch.tensor(
            group[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        for start in range(len(group) - SEQUENCE_LENGTH + 1):
            end = start + SEQUENCE_LENGTH
            sequences.append(values[start:end])

            window = group.iloc[start:end]
            metadata.append({
                "season": season,
                "start_date": window["date"].iloc[0].date().isoformat(),
                "end_date": window["date"].iloc[-1].date().isoformat(),
                "observations": len(window),
                "start_source": window["source_file"].iloc[0],
                "end_source": window["source_file"].iloc[-1],
            })

    if not sequences:
        raise ValueError(
            f"No sequences of length {SEQUENCE_LENGTH} could be created."
        )

    sequence_tensor = torch.stack(sequences)
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(METADATA_FILE, index=False)

    torch.save(
        {
            "sequences": sequence_tensor,
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "normalization_mean": means.to_dict(),
            "normalization_std": stds.to_dict(),
            "source": str(INPUT_FILE),
            "quality_filter_min_valid_pixels": MIN_VALID_PIXELS,
            "rejected_observations": rejected[
                ["date", "season", "valid_pixels", "source_file"]
            ].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records"),
            "data_type": (
                "Satellite vegetation-index features; "
                "not yield-labelled training data"
            ),
        },
        OUTPUT_FILE,
    )

    print("Satellite sequence preparation: PASS")
    print("Input observations:", len(df))
    print("Accepted observations:", len(accepted))
    print("Low-valid-pixel observations excluded:", len(rejected))
    print("Features:", FEATURE_COLUMNS)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Sequences created:", len(sequence_tensor))
    print("Tensor shape:", tuple(sequence_tensor.shape))
    print("Seasons represented:", sorted(metadata_df["season"].unique()))
    print("Saved tensor:", OUTPUT_FILE)
    print("Saved metadata:", METADATA_FILE)


if __name__ == "__main__":
    main()
