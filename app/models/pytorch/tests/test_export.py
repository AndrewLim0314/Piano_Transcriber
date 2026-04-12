from __future__ import annotations

from pathlib import Path

import mido

from audio_project.inference.export import export_note_events_to_json, export_note_events_to_midi
from audio_project.schemas import NoteEvent


def test_export_note_events_to_json(tmp_path: Path) -> None:
    path = export_note_events_to_json(
        [NoteEvent(pitch=60, start_time=0.5, end_time=1.0)],
        tmp_path / "notes.json",
    )

    assert path.exists()
    assert '"pitch": 60' in path.read_text(encoding="utf-8")


def test_export_note_events_to_midi(tmp_path: Path) -> None:
    path = export_note_events_to_midi(
        [NoteEvent(pitch=60, start_time=0.5, end_time=1.0, velocity=80)],
        tmp_path / "notes.mid",
    )

    midi = mido.MidiFile(str(path))
    note_messages = [message for track in midi.tracks for message in track if hasattr(message, "note")]

    assert path.exists()
    assert len(note_messages) >= 2
