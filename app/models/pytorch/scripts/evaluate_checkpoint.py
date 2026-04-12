from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_project.config import ProjectConfig
from audio_project.data.midi import load_note_events
from audio_project.evaluation.note_metrics import compute_note_metrics
from audio_project.evaluation.visualization import plot_piano_roll_comparison, plot_threshold_heatmap
from audio_project.inference.postprocess import postprocess_note_events
from audio_project.inference.service import TranscriptionService, decode_note_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a transcription checkpoint on a manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reports and images.")
    parser.add_argument("--sample-limit", type=int, default=16, help="Maximum number of samples to score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    service = TranscriptionService(config=config, checkpoint_path=args.checkpoint)
    manifest_rows = load_manifest_rows(args.manifest)[: args.sample_limit]
    onset_thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]
    frame_thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]
    quantization_steps = [None, 0.125, 0.25]
    merge_gaps = [0.05, 0.1, 0.2]
    min_note_durations = [0.04, 0.08]

    cached_predictions = []
    for row in manifest_rows:
        onset_probs, frame_probs = service.predict_file(Path(row["audio_path"]))
        cached_predictions.append(
            {
                "audio_path": row["audio_path"],
                "midi_path": row["midi_path"],
                "onset_probs": onset_probs,
                "frame_probs": frame_probs,
                "truth": load_note_events(Path(row["midi_path"])),
            }
        )

    heatmap: list[list[float]] = []
    best_result: dict[str, float | list[dict[str, float | int | str]] | str] | None = None
    best_predictions = None
    best_truth = None
    best_sample_path = None

    for onset_threshold in onset_thresholds:
        row_values: list[float] = []
        for frame_threshold in frame_thresholds:
            average_f1 = -1.0
            average_precision = 0.0
            average_recall = 0.0
            best_local_predictions = []
            best_local_truth = []
            best_local_audio_path = ""
            best_local_settings = None
            for quantization_step in quantization_steps:
                for merge_gap in merge_gaps:
                    for min_note_duration in min_note_durations:
                        metrics = []
                        last_predicted = []
                        last_truth = []
                        last_audio_path = ""
                        for cached in cached_predictions:
                            predicted = decode_note_events(
                                onset_probs=cached["onset_probs"],
                                frame_probs=cached["frame_probs"],
                                hop_length=config.audio.hop_length,
                                sample_rate=config.audio.sample_rate,
                                min_midi=config.labels.min_midi,
                                onset_threshold=onset_threshold,
                                frame_threshold=frame_threshold,
                                max_onsets_per_frame=config.inference.max_onsets_per_frame,
                            )
                            predicted = postprocess_note_events(
                                predicted,
                                min_note_duration_seconds=min_note_duration,
                                merge_gap_seconds=merge_gap,
                                quantization_step_seconds=quantization_step,
                            )
                            truth = cached["truth"]
                            metric = compute_note_metrics(
                                predicted=predicted,
                                truth=truth,
                                onset_tolerance_seconds=0.05,
                                offset_tolerance_seconds=0.10,
                            )
                            metrics.append(metric)
                            last_predicted = predicted
                            last_truth = truth
                            last_audio_path = cached["audio_path"]

                        candidate_f1 = sum(item.f1 for item in metrics) / max(len(metrics), 1)
                        candidate_precision = sum(item.precision for item in metrics) / max(len(metrics), 1)
                        candidate_recall = sum(item.recall for item in metrics) / max(len(metrics), 1)
                        if candidate_f1 > average_f1:
                            average_f1 = candidate_f1
                            average_precision = candidate_precision
                            average_recall = candidate_recall
                            best_local_predictions = last_predicted
                            best_local_truth = last_truth
                            best_local_audio_path = last_audio_path
                            best_local_settings = {
                                "quantization_step_seconds": quantization_step,
                                "merge_gap_seconds": merge_gap,
                                "min_note_duration_seconds": min_note_duration,
                            }
            row_values.append(average_f1)
            if best_result is None or average_f1 > best_result["f1"]:
                best_result = {
                    "onset_threshold": onset_threshold,
                    "frame_threshold": frame_threshold,
                    "f1": average_f1,
                    "precision": average_precision,
                    "recall": average_recall,
                    **(best_local_settings or {}),
                }
                best_predictions = best_local_predictions
                best_truth = best_local_truth
                best_sample_path = best_local_audio_path
        heatmap.append(row_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "note_metrics_summary.json"
    summary_path.write_text(json.dumps(best_result, indent=2), encoding="utf-8")
    heatmap_path = plot_threshold_heatmap(
        onset_thresholds=onset_thresholds,
        frame_thresholds=frame_thresholds,
        values=heatmap,
        output_path=args.output_dir / "note_f1_heatmap.png",
        title="Validation Note F1 Sweep",
    )
    piano_roll_path = plot_piano_roll_comparison(
        predicted=best_predictions or [],
        truth=best_truth or [],
        output_path=args.output_dir / "piano_roll_comparison.png",
        title=f"Best sample comparison: {Path(best_sample_path or '').name}",
    )
    print(f"summary={summary_path}")
    print(f"heatmap={heatmap_path}")
    print(f"piano_roll={piano_roll_path}")
    print(json.dumps(best_result, indent=2))


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if __name__ == "__main__":
    main()
