
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lstm_model import YieldLSTM


BASE_DIR = Path(__file__).resolve().parent

RIDGE_ALPHA = 1.0


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if actual.shape != predicted.shape:
        raise ValueError("Prediction shape mismatch.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError("Non-finite evaluation values.")

    error = predicted - actual

    mse = np.mean(error ** 2)

    ss_tot = np.sum(
        (actual - actual.mean()) ** 2
    )

    return {
        "MAE": np.mean(np.abs(error)),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": (
            1 - np.sum(error ** 2) / ss_tot
            if ss_tot > 0
            else np.nan
        ),
        "Bias": np.mean(error),
    }


def ridge_predict(train_x, train_y, eval_x):

    x = train_x.reshape(
        len(train_x), -1
    ).astype(np.float64)

    z = eval_x.reshape(
        len(eval_x), -1
    ).astype(np.float64)

    y = train_y.astype(np.float64)

    x_mean = x.mean(axis=0)
    y_mean = y.mean()

    xc = x - x_mean
    yc = y - y_mean

    coefficients = np.linalg.solve(
        xc.T @ xc
        + RIDGE_ALPHA * np.eye(x.shape[1]),
        xc.T @ yc,
    )

    intercept = y_mean - x_mean @ coefficients

    return z @ coefficients + intercept


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    args = parser.parse_args()
    split = args.split

    torch.set_num_threads(1)

    data = torch.load(
        BASE_DIR / "growth_sequences.pt",
        map_location="cpu",
        weights_only=True,
    )

    checkpoint = torch.load(
        BASE_DIR / "outputs" / "growth_lstm_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Validate checkpoint compatibility.
    for key in [
        "feature_columns",
        "sequence_length",
        "forecast_days",
        "normalization_mean",
        "normalization_std",
        "normalization_fitted_on",
        "season_splits",
        "target",
    ]:

        if data[key] != checkpoint[key]:
            raise ValueError(
                f"Checkpoint mismatch: {key}"
            )

    if data["target"] != "next_weekly_ndvi":
        raise ValueError("Unexpected target.")

    if data["normalization_fitted_on"] != "train":
        raise ValueError("Invalid normalization.")

    train_x = data["datasets"]["train"]["X"]
    train_y = data["datasets"]["train"]["y"]

    eval_x = data["datasets"][split]["X"]
    actual = data["datasets"][split]["y"].numpy()

    metadata = pd.read_csv(
        BASE_DIR / "growth_sequences_metadata.csv"
    )

    metadata = metadata[
        metadata["split"] == split
    ].reset_index(drop=True)

    if len(metadata) != len(actual):
        raise ValueError("Metadata count mismatch.")

    if not np.allclose(
        actual,
        metadata["target_ndvi"].to_numpy(),
    ):
        raise ValueError("Metadata target mismatch.")

    # 1. Persistence baseline.
    idx = data["feature_columns"].index("ndvi_mean")

    persistence = (
        eval_x[:, -1, idx].numpy()
        * data["normalization_std"][idx]
        + data["normalization_mean"][idx]
    )

    if not np.allclose(
        persistence,
        metadata["last_ndvi"].to_numpy(),
    ):
        raise ValueError("Last NDVI mismatch.")

    # 2. Ridge regression baseline.
    ridge = ridge_predict(
        train_x.numpy(),
        train_y.numpy(),
        eval_x.numpy(),
    )

    # 3. Trained LSTM.
    model = YieldLSTM(
        **checkpoint["model_config"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with torch.no_grad():
        lstm = model(eval_x).squeeze(-1).numpy()

    predictions = {
        "Persistence": persistence,
        "Ridge": ridge,
        "LSTM": lstm,
    }

    results = pd.DataFrame([
        {
            "Model": name,
            **calculate_metrics(actual, values),
        }
        for name, values in predictions.items()
    ]).sort_values("RMSE")

    # Verify saved checkpoint performance.
    if split == "validation":

        restored_mse = calculate_metrics(
            actual, lstm
        )["MSE"]

        if not np.isclose(
            restored_mse,
            checkpoint["best_validation_mse"],
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError(
                "Checkpoint validation MSE mismatch."
            )

        print("Checkpoint reproduction: PASS")

    print("\nGROWTH MODEL EVALUATION")
    print("-----------------------")
    print("Split:", split)
    print("Samples:", len(actual))
    print("Checkpoint epoch:", checkpoint["best_epoch"])
    print("Ridge alpha:", RIDGE_ALPHA)

    print("\nOVERALL MODEL COMPARISON:")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    print("\nSEASON-WISE PERFORMANCE:")

    for season, group in metadata.groupby("season"):

        indices = group.index.to_numpy()

        season_results = pd.DataFrame([
            {
                "Model": name,
                **calculate_metrics(
                    actual[indices],
                    values[indices],
                ),
            }
            for name, values in predictions.items()
        ])

        print("\nSeason:", season)
        print("Samples:", len(indices))

        print(
            season_results.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    print("\nNDVI RANGE CHECK:")

    for name, values in predictions.items():

        outside = np.sum(
            (values < -1) | (values > 1)
        )

        print(name, "| Outside [-1, 1]:", int(outside))

    print(
        "\nLowest RMSE model:",
        results.iloc[0]["Model"],
    )

    print("Additional output files created: NONE")
    print("\nMODEL EVALUATION: PASS")


if __name__ == "__main__":
    main()
