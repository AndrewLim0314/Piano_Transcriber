from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_project.schemas import NoteEvent


def plot_threshold_heatmap(
    onset_thresholds: list[float],
    frame_thresholds: list[float],
    values: list[list[float]],
    output_path: Path,
    title: str,
) -> Path:
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(values, origin="lower", aspect="auto", cmap="magma")
    axis.set_title(title)
    axis.set_xlabel("Frame threshold")
    axis.set_ylabel("Onset threshold")
    axis.set_xticks(range(len(frame_thresholds)))
    axis.set_xticklabels([f"{value:.2f}" for value in frame_thresholds], rotation=45, ha="right")
    axis.set_yticks(range(len(onset_thresholds)))
    axis.set_yticklabels([f"{value:.2f}" for value in onset_thresholds])
    figure.colorbar(image, ax=axis, label="F1")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_piano_roll_comparison(
    predicted: list[NoteEvent],
    truth: list[NoteEvent],
    output_path: Path,
    title: str,
) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    _draw_note_events(axes[0], truth, "Ground Truth")
    _draw_note_events(axes[1], predicted, "Prediction")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("MIDI Pitch")
    axes[0].set_ylabel("MIDI Pitch")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _draw_note_events(axis: plt.Axes, note_events: list[NoteEvent], label: str) -> None:
    axis.set_title(label)
    for event in note_events:
        axis.broken_barh(
            [(event.start_time, max(event.end_time - event.start_time, 1e-3))],
            (event.pitch - 0.4, 0.8),
            facecolors="#1f77b4",
            alpha=0.8,
        )
    axis.grid(alpha=0.2)
