from __future__ import annotations

from dataclasses import dataclass

from audio_project.schemas import NoteEvent


@dataclass(slots=True)
class NoteMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def compute_note_metrics(
    predicted: list[NoteEvent],
    truth: list[NoteEvent],
    onset_tolerance_seconds: float = 0.05,
    offset_tolerance_seconds: float | None = None,
) -> NoteMetrics:
    matched_truth = set()
    true_positives = 0

    sorted_predicted = sorted(predicted, key=lambda item: (item.start_time, item.pitch, item.end_time))
    sorted_truth = sorted(truth, key=lambda item: (item.start_time, item.pitch, item.end_time))

    for pred in sorted_predicted:
        match_index = None
        best_score = None
        for index, target in enumerate(sorted_truth):
            if index in matched_truth or pred.pitch != target.pitch:
                continue
            onset_error = abs(pred.start_time - target.start_time)
            if onset_error > onset_tolerance_seconds:
                continue
            if offset_tolerance_seconds is not None:
                offset_error = abs(pred.end_time - target.end_time)
                if offset_error > offset_tolerance_seconds:
                    continue
            score = onset_error + abs(pred.end_time - target.end_time)
            if best_score is None or score < best_score:
                best_score = score
                match_index = index
        if match_index is not None:
            matched_truth.add(match_index)
            true_positives += 1

    false_positives = len(sorted_predicted) - true_positives
    false_negatives = len(sorted_truth) - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return NoteMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
