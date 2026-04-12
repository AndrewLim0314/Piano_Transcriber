from __future__ import annotations

from audio_project.schemas import NoteEvent


def postprocess_note_events(
    note_events: list[NoteEvent],
    min_note_duration_seconds: float,
    merge_gap_seconds: float,
    quantization_step_seconds: float | None = None,
) -> list[NoteEvent]:
    processed = merge_nearby_events(
        note_events,
        min_gap_seconds=merge_gap_seconds,
        min_note_duration_seconds=min_note_duration_seconds,
    )
    if quantization_step_seconds is not None and quantization_step_seconds > 0:
        processed = quantize_note_events(processed, quantization_step_seconds)
    return processed


def merge_nearby_events(
    note_events: list[NoteEvent],
    min_gap_seconds: float,
    min_note_duration_seconds: float,
) -> list[NoteEvent]:
    merged: list[NoteEvent] = []
    for event in sorted(note_events, key=lambda item: (item.pitch, item.start_time, item.end_time)):
        duration = event.end_time - event.start_time
        if duration < min_note_duration_seconds:
            continue
        if not merged or merged[-1].pitch != event.pitch:
            merged.append(event)
            continue

        previous = merged[-1]
        if event.start_time <= previous.end_time + min_gap_seconds:
            merged[-1] = NoteEvent(
                pitch=previous.pitch,
                start_time=min(previous.start_time, event.start_time),
                end_time=max(previous.end_time, event.end_time),
                velocity=max(previous.velocity, event.velocity),
            )
        else:
            merged.append(event)

    return sorted(merged, key=lambda item: (item.start_time, item.pitch))


def quantize_note_events(note_events: list[NoteEvent], step_seconds: float) -> list[NoteEvent]:
    quantized: list[NoteEvent] = []
    for event in note_events:
        start_time = round(event.start_time / step_seconds) * step_seconds
        end_time = round(event.end_time / step_seconds) * step_seconds
        if end_time <= start_time:
            end_time = start_time + step_seconds
        quantized.append(
            NoteEvent(
                pitch=event.pitch,
                start_time=start_time,
                end_time=end_time,
                velocity=event.velocity,
            )
        )
    return sorted(quantized, key=lambda item: (item.start_time, item.pitch))
