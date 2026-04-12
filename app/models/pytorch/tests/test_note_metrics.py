from __future__ import annotations

from pathlib import Path

from audio_project.evaluation.note_metrics import compute_note_metrics
from audio_project.evaluation.visualization import plot_piano_roll_comparison, plot_threshold_heatmap
from audio_project.inference.postprocess import postprocess_note_events
from audio_project.schemas import NoteEvent


def test_compute_note_metrics_matches_pitch_and_timing() -> None:
    truth = [NoteEvent(pitch=60, start_time=0.5, end_time=1.0)]
    predicted = [NoteEvent(pitch=60, start_time=0.52, end_time=1.03)]

    metrics = compute_note_metrics(predicted, truth, onset_tolerance_seconds=0.05, offset_tolerance_seconds=0.1)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.f1 == 1.0


def test_postprocess_note_events_quantizes_and_merges() -> None:
    events = [
        NoteEvent(pitch=60, start_time=0.48, end_time=0.60),
        NoteEvent(pitch=60, start_time=0.61, end_time=0.98),
    ]

    processed = postprocess_note_events(
        events,
        min_note_duration_seconds=0.05,
        merge_gap_seconds=0.05,
        quantization_step_seconds=0.25,
    )

    assert len(processed) == 1
    assert processed[0].start_time == 0.5
    assert processed[0].end_time == 1.0


def test_visualizations_write_images(tmp_path: Path) -> None:
    heatmap_path = plot_threshold_heatmap(
        onset_thresholds=[0.2, 0.5],
        frame_thresholds=[0.2, 0.5],
        values=[[0.1, 0.2], [0.3, 0.4]],
        output_path=tmp_path / "heatmap.png",
        title="Heatmap",
    )
    piano_roll_path = plot_piano_roll_comparison(
        predicted=[NoteEvent(pitch=60, start_time=0.5, end_time=1.0)],
        truth=[NoteEvent(pitch=60, start_time=0.5, end_time=1.0)],
        output_path=tmp_path / "piano_roll.png",
        title="Comparison",
    )

    assert heatmap_path.exists()
    assert piano_roll_path.exists()
