from __future__ import annotations

import json
from pathlib import Path

from audio_project.training.reporting import EpochMetrics, plot_history, save_history


def sample_history() -> list[EpochMetrics]:
    return [
        EpochMetrics(
            epoch=1,
            train_loss=0.9,
            valid_loss=0.8,
            train_onset_accuracy=0.70,
            valid_onset_accuracy=0.68,
            train_onset_precision=0.30,
            valid_onset_precision=0.25,
            train_onset_recall=0.40,
            valid_onset_recall=0.35,
            train_onset_f1=0.34,
            valid_onset_f1=0.29,
            train_frame_accuracy=0.80,
            valid_frame_accuracy=0.78,
            train_frame_precision=0.50,
            valid_frame_precision=0.45,
            train_frame_recall=0.60,
            valid_frame_recall=0.55,
            train_frame_f1=0.55,
            valid_frame_f1=0.49,
            checkpoint_path="artifact_1.pt",
        ),
        EpochMetrics(
            epoch=2,
            train_loss=0.6,
            valid_loss=0.5,
            train_onset_accuracy=0.75,
            valid_onset_accuracy=0.73,
            train_onset_precision=0.35,
            valid_onset_precision=0.31,
            train_onset_recall=0.46,
            valid_onset_recall=0.41,
            train_onset_f1=0.40,
            valid_onset_f1=0.35,
            train_frame_accuracy=0.87,
            valid_frame_accuracy=0.84,
            train_frame_precision=0.62,
            valid_frame_precision=0.57,
            train_frame_recall=0.70,
            valid_frame_recall=0.64,
            train_frame_f1=0.66,
            valid_frame_f1=0.60,
            checkpoint_path="artifact_2.pt",
        ),
    ]


def test_save_history_writes_json_and_csv(tmp_path: Path) -> None:
    json_path, csv_path = save_history(sample_history(), tmp_path)

    assert json_path.exists()
    assert csv_path.exists()
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[1]["valid_frame_f1"] == 0.60


def test_plot_history_writes_png(tmp_path: Path) -> None:
    plot_path = plot_history(sample_history(), tmp_path)

    assert plot_path is not None
    assert plot_path.exists()
    assert plot_path.suffix == ".png"
