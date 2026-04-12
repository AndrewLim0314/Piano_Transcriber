"""Generate updated showcase sheet music for known melodies.

Runs each melody through the full pipeline:
  synthesize audio → neural model → MIDI → MusicXML (quantized, grand staff) → PDF
"""
from __future__ import annotations

import array
import math
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

import torch

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PYTORCH_DIR = SCRIPT_DIR.parent
SRC_DIR     = PYTORCH_DIR / "src"
ARTIFACTS   = PYTORCH_DIR / "artifacts"
SHOWCASE    = ARTIFACTS / "showcase"
CHECKPOINT  = ARTIFACTS / "run_v3" / "music_transcription_epoch_020.pt"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_project.config import ProjectConfig
from audio_project.inference.export import export_note_events_to_midi
from audio_project.inference.service import TranscriptionService

# ── audio helpers (copied from generate_synthetic_piano_data) ─────────────────

def midi_to_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def synthesize_audio(
    notes: list[tuple[int, float, float, int]],
    duration: float,
    sample_rate: int,
) -> torch.Tensor:
    num_samples = int(duration * sample_rate)
    timeline = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    audio = torch.zeros(num_samples, dtype=torch.float32)

    harmonics = [
        (1.0, 1.000, 2.5),
        (2.0, 0.500, 3.5),
        (3.0, 0.260, 5.0),
        (4.0, 0.140, 7.0),
        (5.0, 0.080, 9.5),
        (6.0, 0.045, 12.0),
        (7.0, 0.022, 15.0),
        (8.0, 0.010, 18.0),
    ]

    for pitch, start_time, end_time, velocity in notes:
        start_idx = int(start_time * sample_rate)
        end_idx   = min(num_samples, int(end_time * sample_rate))
        if end_idx <= start_idx:
            continue
        local_time = timeline[start_idx:end_idx] - start_time
        frequency  = midi_to_frequency(pitch)
        amplitude  = velocity / 127.0
        attack     = torch.clamp(local_time / 0.005, 0.0, 1.0)

        note_wave = torch.zeros_like(local_time)
        for rel_freq, h_amp, h_decay in harmonics:
            partial_freq = frequency * rel_freq
            h_env = attack * (torch.exp(-h_decay * local_time) + 0.06 * torch.exp(-0.4 * local_time))
            note_wave += h_amp * h_env * torch.sin(2.0 * math.pi * partial_freq * local_time)

        audio[start_idx:end_idx] += amplitude * note_wave
        transient_length = min(end_idx - start_idx, int(0.005 * sample_rate))
        if transient_length > 0:
            transient = torch.hann_window(transient_length, periodic=False)
            audio[start_idx : start_idx + transient_length] += 0.04 * amplitude * transient

    peak = audio.abs().max().item()
    if peak > 0:
        audio = 0.8 * audio / peak
    return (audio + 0.001 * torch.randn_like(audio)).clamp(-1.0, 1.0)


def write_wav(audio: torch.Tensor, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio * 32767.0).to(torch.int16).tolist()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array.array("h", pcm).tobytes())


# ── note sequences ─────────────────────────────────────────────────────────────
# Each note: (midi_pitch, start_sec, end_sec, velocity)
# BPM=120 → quarter note = 0.5 s, eighth = 0.25 s

def seven_nation_army() -> tuple[list[tuple[int, float, float, int]], float]:
    """White Stripes riff, two repetitions."""
    # E4 E4 G4 E4 D4 C4 B3 — then up a step
    riff: list[tuple[int, float, float, int]] = [
        (64, 0.000, 0.350, 90),   # E4  dotted quarter
        (64, 0.500, 0.750, 90),   # E4  quarter
        (67, 0.750, 1.000, 90),   # G4  quarter
        (64, 1.000, 1.250, 85),   # E4  quarter
        (62, 1.375, 1.625, 85),   # D4  quarter
        (60, 1.625, 2.250, 85),   # C4  dotted quarter
        (59, 2.500, 3.000, 80),   # B3  half
    ]
    offset = 3.5
    riff2 = [(p, s + offset, e + offset, v) for p, s, e, v in riff]
    all_notes = riff + riff2
    duration = max(e for _, _, e, _ in all_notes) + 0.5
    return all_notes, duration


def iron_man() -> tuple[list[tuple[int, float, float, int]], float]:
    """Iron Man intro riff (simplified melodic line), two repetitions.
    Notes snapped to 16th-note grid (0.125 s) to avoid triplet encoding."""
    riff: list[tuple[int, float, float, int]] = [
        (59, 0.000, 0.125, 95),   # B3  16th
        (62, 0.125, 0.250, 95),   # D4  16th
        (64, 0.250, 0.750, 95),   # E4  half
        (64, 0.875, 1.125, 90),   # E4  8th
        (64, 1.125, 1.375, 90),   # E4  8th
        (63, 1.375, 1.625, 88),   # Eb4 8th
        (61, 1.625, 2.125, 85),   # Db4 quarter
        (59, 2.250, 3.000, 80),   # B3  dotted quarter
    ]
    offset = 3.25
    riff2 = [(p, s + offset, e + offset, v) for p, s, e, v in riff]
    all_notes = riff + riff2
    duration = max(e for _, _, e, _ in all_notes) + 0.5
    return all_notes, duration


def smoke_on_the_water() -> tuple[list[tuple[int, float, float, int]], float]:
    """Smoke on the Water main riff, two repetitions."""
    # D4-F4-Ab4  D4-F4-Bb4-Ab4  D4-F4-Ab4-G4
    riff: list[tuple[int, float, float, int]] = [
        (62, 0.000, 0.375, 90),   # D4
        (65, 0.500, 0.875, 90),   # F4
        (68, 1.000, 1.625, 90),   # Ab4
        (62, 2.000, 2.375, 90),   # D4
        (65, 2.500, 2.875, 90),   # F4
        (70, 3.000, 3.250, 90),   # Bb4
        (68, 3.375, 3.875, 85),   # Ab4
        (62, 4.250, 4.625, 88),   # D4
        (65, 4.750, 5.125, 88),   # F4
        (68, 5.250, 5.875, 88),   # Ab4
        (67, 6.000, 6.750, 85),   # G4
    ]
    offset = 7.25
    riff2 = [(p, s + offset, e + offset, v) for p, s, e, v in riff]
    all_notes = riff + riff2
    duration = max(e for _, _, e, _ in all_notes) + 0.5
    return all_notes, duration


# ── note-event post-processing ────────────────────────────────────────────────

def snap_to_grid(note_events: list, grid_seconds: float = 0.125):
    """Snap NoteEvent start/end times to the nearest grid point.

    The neural model outputs float timestamps that can land slightly off a
    musical grid, causing music21 to encode them as exotic tuplets (6:5, 3:2
    whole-note triplets, etc.) that crash MuseScore with exit 40.
    Snapping to a 16th-note grid at 120 BPM (0.125 s) prevents this.
    """
    from audio_project.schemas import NoteEvent

    snapped = []
    for ev in note_events:
        start = round(ev.start_time / grid_seconds) * grid_seconds
        end   = round(ev.end_time   / grid_seconds) * grid_seconds
        if end <= start:
            end = start + grid_seconds
        snapped.append(NoteEvent(pitch=ev.pitch, start_time=start, end_time=end, velocity=ev.velocity))
    return snapped


# ── sheet music helpers ────────────────────────────────────────────────────────

def midi_to_musicxml(midi_path: Path, output_path: Path) -> Path:
    from music21 import clef as m21clef, converter, stream as m21stream

    score = converter.parse(str(midi_path))
    score = score.quantize(
        quarterLengthDivisors=(4,),
        processOffsets=True,
        processDurations=True,
        inPlace=False,
    )
    key = score.analyze("key")

    SPLIT_MIDI = 60
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
            flat    = m21stream.Part(staff_notes)
            notated = flat.makeNotation()
            for measure in notated.getElementsByClass("Measure"):
                for i, voice in enumerate(measure.getElementsByClass("Voice"), 1):
                    voice.id = str(i)
                for existing_clef in measure.getElementsByClass("Clef"):
                    measure.remove(existing_clef)
                measure.insert(0, staff_clef)
            notated.insert(0, key)
            new_score.append(notated)

    new_score.write("musicxml", fp=str(output_path))
    return output_path


def musicxml_to_pdf(musicxml_path: Path, pdf_path: Path) -> Optional[Path]:
    try:
        subprocess.run(
            ["mscore", str(musicxml_path), "-o", str(pdf_path)],
            check=True, capture_output=True, timeout=60,
        )
        return pdf_path
    except subprocess.CalledProcessError as exc:
        print(f"  MuseScore failed (exit {exc.returncode}): {exc.stderr.decode()[:200]}")
        return None
    except FileNotFoundError:
        print("  mscore not found — PDF step skipped")
        return None
    except subprocess.TimeoutExpired:
        print("  MuseScore timed out")
        return None


# ── main ───────────────────────────────────────────────────────────────────────

def process(
    name: str,
    notes: list[tuple[int, float, float, int]],
    duration: float,
    service: TranscriptionService,
    sample_rate: int,
) -> None:
    print(f"\n── {name} ──")
    SHOWCASE.mkdir(parents=True, exist_ok=True)
    tmp_wav  = SHOWCASE / f"{name}.wav"
    midi_out = SHOWCASE / f"{name}.mid"
    xml_out  = SHOWCASE / f"{name}.musicxml"
    pdf_out  = SHOWCASE / f"{name}.pdf"

    print("  synthesizing audio...")
    audio = synthesize_audio(notes, duration, sample_rate)
    write_wav(audio, sample_rate, tmp_wav)

    print("  transcribing...")
    note_events = service.transcribe_file(tmp_wav)
    note_events = snap_to_grid(note_events)
    print(f"  {len(note_events)} note events detected")
    export_note_events_to_midi(note_events, midi_out)

    print("  converting MIDI → MusicXML...")
    midi_to_musicxml(midi_out, xml_out)

    print("  converting MusicXML → PDF...")
    result = musicxml_to_pdf(xml_out, pdf_out)
    if result:
        print(f"  PDF: {pdf_out}")
    else:
        print("  PDF skipped (MuseScore unavailable)")

    tmp_wav.unlink(missing_ok=True)


def main() -> None:
    print(f"Loading checkpoint: {CHECKPOINT}")
    config  = ProjectConfig()
    service = TranscriptionService(config=config, checkpoint_path=CHECKPOINT)

    sample_rate = config.audio.sample_rate

    melodies = [
        ("7_nation_army",    *seven_nation_army()),
        ("iron_man",         *iron_man()),
        ("smoke_on_water",   *smoke_on_the_water()),
    ]

    for name, notes, duration in melodies:
        process(name, notes, duration, service, sample_rate)

    print("\nDone. Outputs in:", SHOWCASE)


if __name__ == "__main__":
    main()
