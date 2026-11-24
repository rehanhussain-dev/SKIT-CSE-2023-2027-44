
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "s2_field_vegetation_indices_timeseries.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "lstm" / "growth_sequences.pt"
)

METADATA_FILE = (
    PROJECT_ROOT / "lstm" / "growth_sequences_metadata.csv"
)

SEQUENCE_LENGTH = 4
MIN_VALID_COVERAGE = 0.80
FORECAST_DAYS = 7

VEGETATION_FEATURES = [
    "ndvi_mean",
    "ndre_mean",
    "si_mean",
]

FEATURE_COLUMNS = [
    *VEGETATION_FEATURES,
    "days_since_season_start",
    "days_since_previous_observation",
]

SEASON_SPLITS = {
    "train": [
        "2016-2017",
        "2017-2018",
        "2018-2019",
        "2019-2020",
        "2020-2021",
        "2021-2022",
    ],
    "validation": [
        "2022-2023",
        "2023-2024",
    ],
    "test": [
        "2024-2025",
        "2025-2026",
    ],
}


def load_observations():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "field_id",
        "date",
        "season",
        "valid_coverage",
        "quality_pass",
        *VEGETATION_FEATURES,
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    )

    df["valid_coverage"] = pd.to_numeric(
        df["valid_coverage"], errors="raise"
    )

    quality = (
        df["quality_pass"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not quality.isin(["true", "false"]).all():
        raise ValueError("Invalid quality flags.")

    accepted = df[
        (quality == "true")
        & (
            df["valid_coverage"]
            >= MIN_VALID_COVERAGE
        )
    ].copy()

    for col in VEGETATION_FEATURES:

        accepted[col] = pd.to_numeric(
            accepted[col], errors="raise"
        )

    if accepted.empty:
        raise ValueError("No accepted observations.")

    if accepted[required].isna().any().any():
        raise ValueError("Missing accepted values.")

    if not np.isfinite(
        accepted[VEGETATION_FEATURES].to_numpy()
    ).all():
        raise ValueError("Invalid vegetation values.")

    if accepted.duplicated(
        ["field_id", "season", "date"]
    ).any():
        raise ValueError("Duplicate observations.")

    if not accepted["valid_coverage"].between(
        0, 1
    ).all():
        raise ValueError("Invalid coverage values.")

    if not accepted["ndvi_mean"].between(
        -1, 1
    ).all():
        raise ValueError("NDVI outside valid range.")

    known_seasons = {
        season
        for seasons in SEASON_SPLITS.values()
        for season in seasons
    }

    if not set(accepted["season"]).issubset(
        known_seasons
    ):
        raise ValueError("Unexpected season detected.")

    return accepted.sort_values(
        ["field_id", "season", "date"]
    ).reset_index(drop=True)


def create_sequences(df, split):

    sequences = []
    targets = []
    metadata = []

    subset = df[
        df["season"].isin(SEASON_SPLITS[split])
    ]

    groups = subset.groupby(
        ["field_id", "season"], sort=True
    )

    for (field_id, season), group in groups:

        group = group.sort_values(
            "date"
        ).reset_index(drop=True)

        season_year = int(season.split("-")[0])

        # Calendar reference, not actual sowing date.
        season_start = pd.Timestamp(
            year=season_year,
            month=10,
            day=1,
        )

        for start in range(
            len(group) - SEQUENCE_LENGTH
        ):

            end = start + SEQUENCE_LENGTH

            window = group.iloc[start:end].copy()
            target = group.iloc[end]

            last_date = window["date"].iloc[-1]

            forecast_gap = (
                target["date"] - last_date
            ).days

            # Predict the next scheduled weekly
            # composite, not an arbitrary future date.
            if forecast_gap != FORECAST_DAYS:
                continue

            window["days_since_season_start"] = (
                window["date"] - season_start
            ).dt.days

            window["days_since_previous_observation"] = (
                window["date"]
                .diff()
                .dt.days
                .fillna(0)
            )

            # Zero denotes the first input observation,
            # for which the preceding gap is not supplied.

            values = window[
                FEATURE_COLUMNS
            ].to_numpy(dtype=np.float32)

            sequences.append(values)

            targets.append(
                float(target["ndvi_mean"])
            )

            metadata.append({
                "split": split,
                "field_id": field_id,
                "season": season,
                "start_date": (
                    window["date"].iloc[0]
                    .date().isoformat()
                ),
                "end_date": (
                    last_date.date().isoformat()
                ),
                "target_date": (
                    target["date"].date().isoformat()
                ),
                "forecast_days": forecast_gap,
                "window_duration_days": (
                    last_date
                    - window["date"].iloc[0]
                ).days,
                "last_ndvi": float(
                    window["ndvi_mean"].iloc[-1]
                ),
                "target_ndvi": float(
                    target["ndvi_mean"]
                ),
            })

    if not sequences:
        raise ValueError(
            f"No sequences created for {split}"
        )

    return (
        np.stack(sequences),
        np.asarray(targets, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def main():

    df = load_observations()

    datasets = {}
    metadata_parts = []

    for split in SEASON_SPLITS:

        x, y, meta = create_sequences(df, split)

        datasets[split] = {
            "X": x,
            "y": y,
        }

        metadata_parts.append(meta)

    # Fit normalization on training inputs only.
    train_x = datasets["train"]["X"]

    flat_train = train_x.reshape(
        -1, len(FEATURE_COLUMNS)
    ).astype(np.float64)

    means = flat_train.mean(axis=0)
    stds = flat_train.std(axis=0)

    stds = np.where(
        stds < 1e-8, 1.0, stds
    )

    for split, data in datasets.items():

        x = (data["X"] - means) / stds

        if not np.isfinite(x).all():
            raise ValueError(
                f"Invalid normalized inputs: {split}"
            )

        if not np.isfinite(data["y"]).all():
            raise ValueError(
                f"Invalid targets: {split}"
            )

        data["X"] = torch.tensor(
            x, dtype=torch.float32
        )

        # Keep targets in original NDVI units.
        data["y"] = torch.tensor(
            data["y"], dtype=torch.float32
        )

    metadata = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    if (
        metadata.groupby("season")["split"]
        .nunique().max() != 1
    ):
        raise ValueError("Season leakage detected.")

    metadata.to_csv(
        METADATA_FILE, index=False
    )

    torch.save(
        {
            "datasets": datasets,
            "feature_columns": FEATURE_COLUMNS,
            "sequence_length": SEQUENCE_LENGTH,
            "forecast_days": FORECAST_DAYS,
            "normalization_mean": means.tolist(),
            "normalization_std": stds.tolist(),
            "normalization_fitted_on": "train",
            "season_splits": SEASON_SPLITS,
            "target": "next_weekly_ndvi",
            "source": str(INPUT_FILE),
            "data_type": (
                "Exploratory vegetation-index forecasting; "
                "not measured yield or validated wheat growth"
            ),
        },
        OUTPUT_FILE,
    )

    print("\nGROWTH SEQUENCE PREPARATION: PASS")

    print("Accepted observations:", len(df))
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Input features:", FEATURE_COLUMNS)
    print("Forecast horizon:", FORECAST_DAYS)
    print("Normalization fitted on: train only")

    for split, data in datasets.items():

        print("\nSPLIT:", split)
        print("X shape:", tuple(data["X"].shape))
        print("y shape:", tuple(data["y"].shape))

        print(
            "Seasons:",
            sorted(
                metadata.loc[
                    metadata["split"] == split,
                    "season"
                ].unique()
            )
        )

        print(
            "Target NDVI range:",
            float(data["y"].min()),
            float(data["y"].max()),
        )

    print("\nSEQUENCES PER SEASON:")

    print(
        metadata.groupby(
            ["split", "season"]
        ).size().to_string()
    )

    print("\nINPUT WINDOW DURATION:")

    print(
        metadata["window_duration_days"]
        .describe().to_string()
    )

    print("\nTotal sequences:", len(metadata))
    print("Saved tensor:", OUTPUT_FILE)
    print("Saved metadata:", METADATA_FILE)


if __name__ == "__main__":
    main()
