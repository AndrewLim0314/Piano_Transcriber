from __future__ import annotations

import json
from pathlib import Path

import mido

from audio_project.schemas import NoteEvent


def note_events_to_dicts(note_events: list[NoteEvent]) -> list[dict[str, float | int]]:
    return [
        {
            "pitch": event.pitch,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "velocity": event.velocity,
        }
        for event in note_events
    ]


def export_note_events_to_json(note_events: list[NoteEvent], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(note_events_to_dicts(note_events), indent=2), encoding="utf-8")
    return output_path


def export_note_events_to_midi(
    note_events: list[NoteEvent],
    output_path: Path,
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[float, str, int, int]] = []
    for note in note_events:
        events.append((note.start_time, "note_on", note.pitch, note.velocity))
        events.append((note.end_time, "note_off", note.pitch, 0))
    events.sort(key=lambda item: (item[0], 0 if item[1] == "note_off" else 1))

    current_tick = 0
    for event_time, event_type, pitch, velocity in events:
        tick = int(round(mido.second2tick(event_time, ticks_per_beat, tempo)))
        delta = max(0, tick - current_tick)
        current_tick = tick
        track.append(mido.Message(event_type, note=pitch, velocity=velocity, time=delta))

    midi.save(str(output_path))
    return output_path
