from __future__ import annotations

import math
from pathlib import Path

import mido
import torch

from audio_project.schemas import NoteEvent


def load_note_events(midi_path: Path) -> list[NoteEvent]:
    midi = mido.MidiFile(str(midi_path))
    active_notes: dict[int, list[tuple[float, int]]] = {}
    note_events: list[NoteEvent] = []
    current_time = 0.0

    for message in midi:
        current_time += message.time
        if not hasattr(message, "note"):
            continue

        if message.type == "note_on" and message.velocity > 0:
            active_notes.setdefault(message.note, []).append((current_time, message.velocity))
            continue

        if message.type in {"note_off", "note_on"}:
            stack = active_notes.get(message.note)
            if not stack:
                continue
            start_time, velocity = stack.pop(0)
            end_time = max(current_time, start_time + 0.01)
            note_events.append(
                NoteEvent(
                    pitch=message.note,
                    start_time=start_time,
                    end_time=end_time,
                    velocity=velocity,
                )
            )

    note_events.sort(key=lambda event: (event.start_time, event.pitch))
    return note_events


def note_events_to_targets(
    note_events: list[NoteEvent],
    num_frames: int,
    sample_rate: int,
    hop_length: int,
    min_midi: int,
    max_midi: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pitches = max_midi - min_midi + 1
    onset_targets = torch.zeros(num_frames, num_pitches, dtype=torch.float32)
    frame_targets = torch.zeros(num_frames, num_pitches, dtype=torch.float32)
    frames_per_second = sample_rate / hop_length

    for note in note_events:
        if note.pitch < min_midi or note.pitch > max_midi:
            continue

        pitch_index = note.pitch - min_midi
        start_frame = max(0, min(num_frames - 1, int(round(note.start_time * frames_per_second))))
        end_frame = max(start_frame + 1, int(math.ceil(note.end_time * frames_per_second)))
        end_frame = min(num_frames, end_frame)

        onset_targets[start_frame, pitch_index] = 1.0
        frame_targets[start_frame:end_frame, pitch_index] = 1.0

    return onset_targets, frame_targets
