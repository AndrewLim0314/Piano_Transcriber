from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    valid_loss: float
    train_onset_accuracy: float
    valid_onset_accuracy: float
    train_onset_precision: float
    valid_onset_precision: float
    train_onset_recall: float
    valid_onset_recall: float
    train_onset_f1: float
    valid_onset_f1: float
    train_frame_accuracy: float
    valid_frame_accuracy: float
    train_frame_precision: float
    valid_frame_precision: float
    train_frame_recall: float
    valid_frame_recall: float
    train_frame_f1: float
    valid_frame_f1: float
    checkpoint_path: str


def save_history(history: list[EpochMetrics], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "history.json"
    csv_path = output_dir / "history.csv"

    rows = [asdict(item) for item in history]
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


def plot_history(history: list[EpochMetrics], output_dir: Path) -> Path | None:
    if not history:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [item.epoch for item in history]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle("Training Summary", fontsize=14)

    axes[0, 0].plot(epochs, [item.train_loss for item in history], label="train")
    axes[0, 0].plot(epochs, [item.valid_loss for item in history], label="valid")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [item.train_onset_f1 for item in history], label="train")
    axes[0, 1].plot(epochs, [item.valid_onset_f1 for item in history], label="valid")
    axes[0, 1].set_title("Onset F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [item.train_frame_f1 for item in history], label="train")
    axes[1, 0].plot(epochs, [item.valid_frame_f1 for item in history], label="valid")
    axes[1, 0].set_title("Frame F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [item.train_frame_accuracy for item in history], label="train")
    axes[1, 1].plot(epochs, [item.valid_frame_accuracy for item in history], label="valid")
    axes[1, 1].set_title("Frame Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)

    figure.tight_layout()
    plot_path = output_dir / "training_summary.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path
