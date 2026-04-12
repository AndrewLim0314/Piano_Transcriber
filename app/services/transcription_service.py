import sys
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional

import pretty_midi
from music21 import clef as m21clef, converter

from app.config import get_settings

settings = get_settings()

# Make the audio_project package importable without requiring a separate install step.
_AUDIO_PROJECT_SRC = Path(__file__).resolve().parent.parent / "models" / "pytorch" / "src"
if str(_AUDIO_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIO_PROJECT_SRC))

try:
    from audio_project.inference.service import TranscriptionService as NeuralTranscriptionService
    from audio_project.inference.export import export_note_events_to_midi
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

from app.services.audio_processor import AudioProcessor


def _note_event_to_dict(event) -> dict:
    """Bridge NoteEvent (neural) → dict format stored in the DB and returned by the API."""
    return {
        "pitch": pretty_midi.note_number_to_name(event.pitch),
        "midi_number": event.pitch,
        "start_time": event.start_time,
        "end_time": event.end_time,
        "duration": event.end_time - event.start_time,
        "velocity": event.velocity,
    }


class TranscriptionService:
    def __init__(self):
        self._neural_service: Optional["NeuralTranscriptionService"] = None
        self.audio_processor = AudioProcessor()
        self._init_neural_service()

    def _init_neural_service(self):
        if not _NEURAL_AVAILABLE:
            print("audio_project not importable (torch missing?); using librosa fallback")
            return
        checkpoint_path: Optional[Path] = None
        if settings.neural_model_checkpoint:
            checkpoint_path = Path(settings.neural_model_checkpoint)
        try:
            self._neural_service = NeuralTranscriptionService(checkpoint_path=checkpoint_path)
            print(f"Neural TranscriptionService loaded (checkpoint={checkpoint_path})")
        except Exception as exc:
            print(f"Neural model failed to load ({exc}); using librosa fallback")

    def process_transcription(self, file_id: str, audio_path: str) -> Dict:
        """Complete transcription pipeline: Audio → MIDI → MusicXML → PDF"""
        output_dir = settings.output_dir
        midi_path = os.path.join(output_dir, f"{file_id}.mid")
        musicxml_path = os.path.join(output_dir, f"{file_id}.musicxml")
        pdf_path = os.path.join(output_dir, f"{file_id}.pdf")

        if self._neural_service is not None:
            # Neural path: polyphonic, onset+frame model
            note_events = self._neural_service.transcribe_file(Path(audio_path))
            export_note_events_to_midi(note_events, Path(midi_path))
            notes = [_note_event_to_dict(e) for e in note_events]
        else:
            # Fallback: simple monophonic librosa pipeline
            notes = self.audio_processor.simple_transcribe(audio_path)
            self._notes_dict_to_midi(notes, midi_path)

        self._midi_to_musicxml(midi_path, musicxml_path)
        pdf_result = self._musicxml_to_pdf_musescore(musicxml_path, pdf_path)

        return {
            "notes": notes,
            "midi_path": midi_path,
            "musicxml_path": musicxml_path,
            "pdf_path": pdf_result,
        }

    # ── fallback helpers ──────────────────────────────────────────────────────

    def _notes_dict_to_midi(self, notes: list, output_path: str) -> str:
        """Convert librosa notes list to MIDI (fallback path only)."""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        for n in notes:
            note = pretty_midi.Note(
                velocity=n["velocity"],
                pitch=n["midi_number"],
                start=n["start_time"],
                end=n["start_time"] + n["duration"],
            )
            instrument.notes.append(note)
        midi.instruments.append(instrument)
        midi.write(output_path)
        return output_path

    def _midi_to_musicxml(self, midi_path: str, output_path: str) -> str:
        from music21 import stream as m21stream

        score = converter.parse(midi_path)

        # Snap raw millisecond timestamps to the nearest musical rhythm.
        # (4,) covers whole → 16th notes; (3,) adds triplets.
        # Note: only use triplets when notes genuinely land on triplet grid;
        # off-grid timing + triplet divisor can produce malformed tuplets in
        # MusicXML that some renderers reject. The default stays (4, 3) but
        # can be narrowed to (4,) if a specific export fails.
        score = score.quantize(
            quarterLengthDivisors=(4, 3),
            processOffsets=True,
            processDurations=True,
            inPlace=False,
        )

        key = score.analyze("key")

        # After quantize(), overlapping notes in the same voice cause MuseScore
        # to render them sequentially (adding extra beats). Fix: flatten each
        # part to absolute-offset notes, rebuild measures with makeNotation()
        # so overlapping notes are separated into distinct Voices, then
        # renumber voices from 0-indexed to 1-indexed (MusicXML requires ≥1).
        #
        # Grand staff split: notes ≥ C4 (MIDI 60) → treble clef;
        # notes below C4 → bass clef (only added when such notes exist).
        SPLIT_MIDI = 60  # middle C

        new_score = m21stream.Score()
        for part in score.parts:
            all_notes = list(part.flatten().notesAndRests)

            treble_notes = [n for n in all_notes if not hasattr(n, "pitch") or n.pitch.midi >= SPLIT_MIDI]
            bass_notes   = [n for n in all_notes if hasattr(n, "pitch") and n.pitch.midi < SPLIT_MIDI]

            for staff_notes, staff_clef in [
                (treble_notes, m21clef.TrebleClef()),
                (bass_notes,   m21clef.BassClef()),
            ]:
                if not staff_notes:
                    continue
                flat = m21stream.Part(staff_notes)
                notated = flat.makeNotation()
                for measure in notated.getElementsByClass("Measure"):
                    for i, voice in enumerate(measure.getElementsByClass("Voice"), 1):
                        voice.id = str(i)
                    # Replace auto-assigned clef with the correct one for this staff
                    for existing_clef in measure.getElementsByClass("Clef"):
                        measure.remove(existing_clef)
                    measure.insert(0, staff_clef)
                notated.insert(0, key)
                new_score.append(notated)

        new_score.write("musicxml", fp=output_path)
        return output_path

    def _musicxml_to_pdf_musescore(self, musicxml_path: str, output_path: str) -> Optional[str]:
        try:
            subprocess.run(
                ["mscore", musicxml_path, "-o", output_path],
                check=True, capture_output=True, timeout=30,
            )
            return output_path
        except subprocess.CalledProcessError as exc:
            print(f"MuseScore conversion failed: {exc}")
            return None
        except FileNotFoundError:
            print("MuseScore not installed")
            return None
        except subprocess.TimeoutExpired:
            print("MuseScore timeout")
            return None
