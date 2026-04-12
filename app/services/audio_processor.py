import librosa
import numpy as np
import pretty_midi
from typing import List, Tuple


class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def extract_pitch(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pitch using librosa"""
        # Use piptrack for pitch detection
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        return pitches, magnitudes

    def detect_onset(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Detect note onsets"""
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return onset_times

    def extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """Extract tempo (BPM)"""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)

    def pitch_to_midi(self, frequency: float) -> int:
        """Convert frequency to MIDI note number"""
        if frequency <= 0:
            return 0
        return int(round(69 + 12 * np.log2(frequency / 440.0)))

    def simple_transcribe(self, file_path: str) -> List[dict]:
        """
        Simple monophonic transcription
        For production, use specialized models like:
        - basic-pitch (Spotify)
        - crepe
        - mt3 (Google)
        """
        y, sr = self.load_audio(file_path)

        # Extract pitches and onsets
        onset_times = self.detect_onset(y, sr)
        pitches, magnitudes = self.extract_pitch(y, sr)

        notes = []

        # Simple note extraction (monophonic)
        for i, onset_time in enumerate(onset_times):
            # Get frame index
            frame_idx = librosa.time_to_frames(onset_time, sr=sr)

            if frame_idx < pitches.shape[1]:
                # Get pitch with highest magnitude at this frame
                pitch_idx = magnitudes[:, frame_idx].argmax()
                frequency = pitches[pitch_idx, frame_idx]

                if frequency > 0:
                    midi_note = self.pitch_to_midi(frequency)

                    # Estimate duration until next onset
                    if i < len(onset_times) - 1:
                        duration = onset_times[i + 1] - onset_time
                    else:
                        duration = 0.5  # Default duration

                    notes.append({
                        "pitch": pretty_midi.note_number_to_name(midi_note),
                        "midi_number": midi_note,
                        "start_time": float(onset_time),
                        "duration": float(duration),
                        "velocity": 80
                    })

        return notes