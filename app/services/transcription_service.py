import subprocess
import os
import pretty_midi
from music21 import converter
from typing import Dict, Optional
from app.config import get_settings
from app.services.audio_processor import AudioProcessor

settings = get_settings()


class TranscriptionService:
    def __init__(self):
        self.audio_processor = AudioProcessor()

    def process_transcription(
            self,
            file_id: str,
            audio_path: str  # ← Changed: This is MP3/audio file
    ) -> Dict:
        """Complete transcription pipeline: Audio → MIDI → MusicXML → PDF"""

        output_dir = settings.output_dir

        # Step 1: Transcribe audio to notes
        notes = self.audio_processor.simple_transcribe(audio_path)

        # Step 2: Convert notes to MIDI
        midi_path = os.path.join(output_dir, f"{file_id}.mid")
        self.notes_to_midi(notes, midi_path)

        # Step 3: Generate MusicXML from MIDI
        musicxml_path = os.path.join(output_dir, f"{file_id}.musicxml")
        self.midi_to_musicxml(midi_path, musicxml_path)

        # Step 4: Generate PDF from MIDI (using MuseScore)
        pdf_path = os.path.join(output_dir, f"{file_id}.pdf")
        pdf_result = self.midi_to_pdf_musescore(midi_path, pdf_path)

        return {
            "notes": notes,
            "midi_path": midi_path,
            "musicxml_path": musicxml_path,
            "pdf_path": pdf_result  # Can be None if MuseScore fails
        }

    def notes_to_midi(self, notes: list, output_path: str) -> str:
        """Convert notes list to MIDI file"""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Piano

        for n in notes:
            note = pretty_midi.Note(
                velocity=n['velocity'],
                pitch=n['midi_number'],
                start=n['start_time'],
                end=n['start_time'] + n['duration']
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)
        midi.write(output_path)
        return output_path

    def midi_to_musicxml(self, midi_path: str, output_path: str) -> str:
        """Convert MIDI to MusicXML using music21"""
        score = converter.parse(midi_path)
        score.write('musicxml', fp=output_path)
        return output_path

    def midi_to_pdf_musescore(self, midi_path: str, output_path: str) -> Optional[str]:
        """Convert MIDI to PDF using MuseScore command line"""
        try:
            subprocess.run([
                'musescore3',  # or 'mscore' or 'musescore'
                midi_path,
                '-o', output_path
            ], check=True, capture_output=True, timeout=30)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"MuseScore conversion failed: {e}")
            return None
        except FileNotFoundError:
            print("MuseScore not installed")
            return None
        except subprocess.TimeoutExpired:
            print("MuseScore timeout")
            return None