
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from lstm_model import YieldLSTM


BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "growth_sequences.pt"

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_FILE = OUTPUT_DIR / "growth_lstm_best.pt"

HISTORY_FILE = OUTPUT_DIR / "growth_lstm_training_history.csv"


SEED = 42
HIDDEN_SIZE = 16
NUM_LAYERS = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
PATIENCE = 20

DEVICE = torch.device("cpu")


def set_seed():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(x).squeeze(-1)

        if predictions.shape != y.shape:
            raise ValueError(
                "Prediction-target shape mismatch."
            )

        loss = criterion(predictions, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss."
            )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(x)
        total_samples += len(x)

    return total_loss / total_samples


def evaluate(model, x, y, criterion):

    model.eval()

    with torch.no_grad():

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        predictions = model(x).squeeze(-1)

        if predictions.shape != y.shape:
            raise ValueError(
                "Prediction-target shape mismatch."
            )

        loss = criterion(predictions, y)

        mae = torch.mean(
            torch.abs(predictions - y)
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite validation loss."
            )

    return loss.item(), mae.item()


def main():

    set_seed()

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = torch.load(
        DATA_FILE,
        map_location="cpu",
        weights_only=True,
    )

    if data["target"] != "next_weekly_ndvi":
        raise ValueError("Unexpected prediction target.")

    if data["normalization_fitted_on"] != "train":
        raise ValueError("Invalid normalization source.")

    train_x = data["datasets"]["train"]["X"]
    train_y = data["datasets"]["train"]["y"]

    val_x = data["datasets"]["validation"]["X"]
    val_y = data["datasets"]["validation"]["y"]

    input_size = len(data["feature_columns"])

    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "output_size": 1,
        "dropout": 0.0,
    }

    for name, x, y in [
        ("train", train_x, train_y),
        ("validation", val_x, val_y),
    ]:

        if (
            x.ndim != 3
            or x.shape[1] != data["sequence_length"]
            or x.shape[2] != input_size
            or y.shape != (len(x),)
            or len(x) == 0
        ):
            raise ValueError(
                f"Invalid dataset shape: {name}"
            )

        if not (
            torch.isfinite(x).all()
            and torch.isfinite(y).all()
        ):
            raise ValueError(
                f"Invalid dataset values: {name}"
            )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = YieldLSTM(
        **model_config
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("GROWTH LSTM TRAINING")
    print("--------------------")

    print("Device:", DEVICE)
    print("Random seed:", SEED)
    print("Training samples:", len(train_x))
    print("Validation samples:", len(val_x))
    print("Model configuration:", model_config)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    initial_val_loss, initial_val_mae = evaluate(
        model,
        val_x,
        val_y,
        criterion,
    )

    print(
        "Initial validation MSE:",
        initial_val_loss,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_mae = evaluate(
            model,
            val_x,
            val_y,
            criterion,
        )

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
            "validation_mae": val_mae,
        })

        improved = val_loss < best_val_loss

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_loss:.8f} | "
                f"Val MSE: {val_loss:.8f} | "
                f"Val MAE: {val_mae:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint."
        )

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    final_val_loss, final_val_mae = evaluate(
        model,
        val_x,
        val_y,
        criterion,
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        HISTORY_FILE,
        index=False,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_columns": data["feature_columns"],
            "sequence_length": data["sequence_length"],
            "forecast_days": data["forecast_days"],
            "normalization_mean": data["normalization_mean"],
            "normalization_std": data["normalization_std"],
            "normalization_fitted_on": "train",
            "season_splits": data["season_splits"],
            "target": data["target"],
            "best_epoch": best_epoch,
            "best_validation_mse": final_val_loss,
            "best_validation_mae": final_val_mae,
            "seed": SEED,
            "training_config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
            },
            "data_type": (
                "Exploratory native-resolution NDVI "
                "forecasting model; not a validated "
                "wheat yield prediction model"
            ),
        },
        MODEL_FILE,
    )

    print("\nTRAINING SUMMARY")
    print("----------------")

    print("Epochs executed:", len(history))
    print("Best epoch:", best_epoch)

    print(
        "Initial validation MSE:",
        f"{initial_val_loss:.8f}",
    )

    print(
        "Initial validation MAE:",
        f"{initial_val_mae:.6f}",
    )

    print(
        "Best validation MSE:",
        f"{final_val_loss:.8f}",
    )

    print(
        "Best validation RMSE:",
        f"{np.sqrt(final_val_loss):.6f}",
    )

    print(
        "Best validation MAE:",
        f"{final_val_mae:.6f}",
    )

    print("Saved model:", MODEL_FILE)
    print("Saved history:", HISTORY_FILE)

    print("Test dataset evaluated: NO")

    print("\nLSTM TRAINING: PASS")


if __name__ == "__main__":
    main()
