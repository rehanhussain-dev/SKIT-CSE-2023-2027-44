"""Fine-tune the pretrained RGB-NIR OpenSR generator on paired patches.

The dataset directory must contain ``.npz`` files with two arrays:

    lr: (4, height, width)     10 m RGB-NIR input
    hr: (4, height * 4, width * 4) 2.5 m target

Arrays may be stored as 0..1 reflectance or Sentinel-2-style 0..10000
values. The script normalizes both representations to 0..1 before training.
This is supervised generator fine-tuning; the pretrained discriminator is not
updated because local paired agricultural targets are the evidence needed for
this adaptation step.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

FACTOR = 4
CHANNELS = 4
SCALE_FACTOR = 10000.0


def _normalize(array: np.ndarray) -> np.ndarray:
    """Convert reflectance-like data to finite float32 values in 0..1."""
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("A training patch contains no finite pixels")
    if float(np.nanmax(values[finite])) > 20:
        values = values / SCALE_FACTOR
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 1)


def load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one paired ``.npz`` training example."""
    with np.load(path) as archive:
        if "lr" not in archive or "hr" not in archive:
            raise ValueError(f"{path} must contain 'lr' and 'hr' arrays")
        lr = _normalize(archive["lr"])
        hr = _normalize(archive["hr"])

    if lr.ndim != 3 or hr.ndim != 3:
        raise ValueError(f"{path} arrays must have shape (channels, height, width)")
    if lr.shape[0] != CHANNELS or hr.shape[0] != CHANNELS:
        raise ValueError(f"{path} arrays must contain exactly {CHANNELS} bands")
    if hr.shape[1:] != (lr.shape[1] * FACTOR, lr.shape[2] * FACTOR):
        raise ValueError(
            f"{path} HR shape {hr.shape[1:]} does not match "
            f"{FACTOR}x LR shape {lr.shape[1:]}"
        )
    return lr, hr


class PairedPatchDataset:
    """Lazy dataset for paired ``.npz`` patches, without a hard PyTorch import."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        import torch

        lr, hr = load_pair(self.paths[index])
        return torch.from_numpy(lr), torch.from_numpy(hr)


@dataclass
class TrainConfig:
    preset: str
    data_dir: str
    output: str
    epochs: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    seed: int
    device: str


def split_paths(paths: list[Path], validation_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Create a deterministic, non-empty train/validation split."""
    if len(paths) < 2:
        raise ValueError("At least two paired patches are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def _run_epoch(model: Any, loader: Any, optimizer: Any | None, device: Any) -> float:
    import torch
    import torch.nn.functional as functional

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(lr)
            loss = functional.l1_loss(prediction, hr)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / batches


def fine_tune(config: TrainConfig) -> dict[str, Any]:
    """Fine-tune and checkpoint the pretrained generator."""
    import torch
    from torch.utils.data import DataLoader
    from opensr_srgan import load_inference_model

    torch.manual_seed(config.seed)
    paths = sorted(Path(config.data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz paired patches found in {config.data_dir}")
    train_paths, validation_paths = split_paths(paths, config.validation_fraction, config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    pretrained = load_inference_model(config.preset, map_location=device)
    generator = pretrained.generator.to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=config.learning_rate)
    train_loader = DataLoader(PairedPatchDataset(train_paths), batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(PairedPatchDataset(validation_paths), batch_size=config.batch_size)

    best_validation = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(generator, train_loader, optimizer, device)
        validation_loss = _run_epoch(generator, validation_loader, None, device)
        metrics = {"epoch": epoch, "train_l1": train_loss, "validation_l1": validation_loss}
        history.append(metrics)
        print(
            f"epoch {epoch:03d}/{config.epochs:03d} "
            f"train_l1={train_loss:.6f} validation_l1={validation_loss:.6f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            Path(config.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_l1": best_validation,
                    "config": asdict(config),
                    "train_files": [str(path) for path in train_paths],
                    "validation_files": [str(path) for path in validation_paths],
                    "history": history,
                },
                config.output,
            )
    return {"best_validation_l1": best_validation, "history": history}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("srgan/checkpoints/rgb_nir_finetuned.pt"))
    parser.add_argument("--preset", default="RGB-NIR", choices=("RGB-NIR",))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    return TrainConfig(
        preset=args.preset,
        data_dir=str(args.data_dir),
        output=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    result = fine_tune(parse_args())
    print(json.dumps({"best_validation_l1": result["best_validation_l1"]}, indent=2))
