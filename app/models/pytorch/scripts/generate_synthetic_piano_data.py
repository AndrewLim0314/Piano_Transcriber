from __future__ import annotations

import argparse
import array
import json
import math
import random
import wave
from pathlib import Path

import mido
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic piano transcription dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root output directory.")
    parser.add_argument("--train-count", type=int, default=48, help="Number of training examples.")
    parser.add_argument("--valid-count", type=int, default=8, help="Number of validation examples.")
    parser.add_argument("--duration", type=float, default=4.0, help="Clip duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Audio sample rate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--style",
        choices=["random", "structured", "monophonic"],
        default="random",
        help="Synthetic composition style.",
    )
    return parser.parse_args()


def midi_to_frequency(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def write_wav(audio: torch.Tensor, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio * 32767.0).to(torch.int16).tolist()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(array.array("h", pcm).tobytes())


def create_note_schedule(duration: float, rng: random.Random) -> list[tuple[int, float, float, int]]:
    notes: list[tuple[int, float, float, int]] = []
    current_time = 0.0
    while current_time < duration - 0.35:
        chord_size = rng.choice([1, 1, 2, 2, 3])
        start_time = current_time + rng.uniform(0.05, 0.2)
        note_duration = rng.uniform(0.2, 0.7)
        pitches = sorted({rng.randint(48, 84) for _ in range(chord_size)})
        velocity = rng.randint(60, 110)
        for pitch in pitches:
            end_time = min(duration - 0.05, start_time + note_duration)
            notes.append((pitch, start_time, end_time, velocity))
        current_time = start_time + rng.uniform(0.18, 0.45)
    return notes


def create_structured_note_schedule(duration: float, rng: random.Random) -> list[tuple[int, float, float, int]]:
    notes: list[tuple[int, float, float, int]] = []
    beat_duration = 0.5
    subdivision = beat_duration / 2.0
    total_steps = int(duration / subdivision)
    root = rng.choice([48, 50, 52, 53, 55, 57, 60, 62])
    palette = rng.choice(
        [
            [0, 4, 7, 12, 16],       # major arpeggio
            [0, 3, 7, 12, 15],       # minor arpeggio
            [0, 2, 4, 7, 9, 12],     # pentatonic-like
            [0, 5, 7, 12, 17],       # open fifths
        ]
    )
    motif_length = rng.choice([4, 8])
    motif = [root + rng.choice(palette) for _ in range(motif_length)]
    previous_pitch = None

    step = 0
    while step < total_steps:
        if rng.random() < 0.15:
            step += 1
            continue

        start_time = step * subdivision
        duration_steps = rng.choice([1, 1, 2, 2, 4])
        end_time = min(duration, (step + duration_steps) * subdivision)
        velocity = rng.randint(70, 110)

        if step % 8 == 0 and rng.random() < 0.35:
            chord_root = rng.choice([root, root + 5, root + 7, root + 12])
            chord_intervals = rng.choice([[0, 4, 7], [0, 3, 7], [0, 7]])
            chord = [max(45, min(84, chord_root + interval)) for interval in chord_intervals]
            for pitch in chord[: rng.choice([2, 3])]:
                notes.append((pitch, start_time, end_time, velocity))
        else:
            pitch = motif[step % motif_length]
            if previous_pitch is not None and rng.random() < 0.25:
                pitch = previous_pitch + rng.choice([-2, -1, 1, 2])
            pitch = max(45, min(84, pitch))
            notes.append((pitch, start_time, end_time, velocity))
            previous_pitch = pitch

        step += duration_steps

    return notes


def create_monophonic_note_schedule(duration: float, rng: random.Random) -> list[tuple[int, float, float, int]]:
    notes: list[tuple[int, float, float, int]] = []
    beat_duration = 0.5
    subdivision = beat_duration / 2.0
    total_steps = int(duration / subdivision)
    root = rng.choice([48, 50, 52, 53, 55, 57, 60, 62])
    scale = rng.choice(
        [
            [0, 2, 4, 5, 7, 9, 11, 12],
            [0, 2, 3, 5, 7, 8, 10, 12],
            [0, 3, 5, 7, 10, 12],
        ]
    )
    contour = [root + rng.choice(scale) for _ in range(rng.choice([4, 8]))]
    previous_pitch = contour[0]
    step = 0

    while step < total_steps:
        if rng.random() < 0.15:
            step += 1
            continue

        start_time = step * subdivision
        duration_steps = rng.choice([1, 1, 1, 2])
        articulation_gap = rng.choice([1, 1, 2])
        end_time = min(duration, (step + duration_steps) * subdivision - 0.03)
        velocity = rng.randint(80, 110)

        pitch = contour[step % len(contour)]
        if rng.random() < 0.4:
            pitch = previous_pitch + rng.choice([-2, -1, 1, 2])
        pitch = max(48, min(79, pitch))
        notes.append((pitch, start_time, end_time, velocity))
        previous_pitch = pitch
        step += duration_steps + articulation_gap

    return notes


def synthesize_audio(
    notes: list[tuple[int, float, float, int]],
    duration: float,
    sample_rate: int,
) -> torch.Tensor:
    num_samples = int(duration * sample_rate)
    timeline = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    audio = torch.zeros(num_samples, dtype=torch.float32)

    # Piano harmonic series: (relative_freq_multiplier, amplitude, decay_rate).
    # Higher harmonics are quieter and decay faster, giving a bright attack
    # that mellows into a warm sustain — characteristic of real piano strings.
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
        end_idx = min(num_samples, int(end_time * sample_rate))
        if end_idx <= start_idx:
            continue
        local_time = timeline[start_idx:end_idx] - start_time
        frequency = midi_to_frequency(pitch)
        amplitude = velocity / 127.0
        # Sharp 5 ms attack
        attack = torch.clamp(local_time / 0.005, 0.0, 1.0)

        note_wave = torch.zeros_like(local_time)
        for rel_freq, h_amp, h_decay in harmonics:
            partial_freq = frequency * rel_freq
            # Per-harmonic envelope: fast initial decay + slow sustain tail
            h_env = attack * (torch.exp(-h_decay * local_time) + 0.06 * torch.exp(-0.4 * local_time))
            note_wave += h_amp * h_env * torch.sin(2.0 * math.pi * partial_freq * local_time)

        audio[start_idx:end_idx] += amplitude * note_wave

        # Percussive transient (hammer strike)
        transient_length = min(end_idx - start_idx, int(0.005 * sample_rate))
        if transient_length > 0:
            transient = torch.hann_window(transient_length, periodic=False)
            audio[start_idx : start_idx + transient_length] += 0.04 * amplitude * transient

    peak = audio.abs().max().item()
    if peak > 0:
        audio = 0.8 * audio / peak
    noise = 0.001 * torch.randn_like(audio)
    return (audio + noise).clamp(-1.0, 1.0)


def write_midi(
    notes: list[tuple[int, float, float, int]],
    sample_rate: int,
    path: Path,
) -> None:
    del sample_rate
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[float, str, int, int]] = []
    for pitch, start_time, end_time, velocity in notes:
        events.append((start_time, "note_on", pitch, velocity))
        events.append((end_time, "note_off", pitch, 0))
    events.sort(key=lambda item: (item[0], 0 if item[1] == "note_off" else 1))

    current_tick = 0
    for event_time, kind, pitch, velocity in events:
        tick = int(round(mido.second2tick(event_time, midi.ticks_per_beat, tempo)))
        delta = max(0, tick - current_tick)
        current_tick = tick
        track.append(mido.Message(kind, note=pitch, velocity=velocity, time=delta))

    midi.save(str(path))


def generate_example(
    index: int,
    split: str,
    output_dir: Path,
    duration: float,
    sample_rate: int,
    rng: random.Random,
    style: str,
) -> dict[str, str | float]:
    notes = (
        create_monophonic_note_schedule(duration=duration, rng=rng)
        if style == "monophonic"
        else create_structured_note_schedule(duration=duration, rng=rng)
        if style == "structured"
        else create_note_schedule(duration=duration, rng=rng)
    )
    audio = synthesize_audio(notes=notes, duration=duration, sample_rate=sample_rate)
    audio_path = output_dir / split / "audio" / f"{split}_{index:04d}.wav"
    midi_path = output_dir / split / "midi" / f"{split}_{index:04d}.mid"
    write_wav(audio=audio, sample_rate=sample_rate, path=audio_path)
    write_midi(notes=notes, sample_rate=sample_rate, path=midi_path)
    return {
        "audio_path": str(audio_path.resolve()),
        "midi_path": str(midi_path.resolve()),
        "duration": duration,
        "split": split,
        "dataset": f"synthetic-piano-{style}-v2",
    }


def write_manifest(rows: list[dict[str, str | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    manifests_dir = output_dir / "manifests"

    train_rows = [
        generate_example(
            index=index,
            split="train",
            output_dir=output_dir,
            duration=args.duration,
            sample_rate=args.sample_rate,
            rng=rng,
            style=args.style,
        )
        for index in range(args.train_count)
    ]
    valid_rows = [
        generate_example(
            index=index,
            split="valid",
            output_dir=output_dir,
            duration=args.duration,
            sample_rate=args.sample_rate,
            rng=rng,
            style=args.style,
        )
        for index in range(args.valid_count)
    ]

    write_manifest(train_rows, manifests_dir / "train.jsonl")
    write_manifest(valid_rows, manifests_dir / "valid.jsonl")
    print(f"train_manifest={manifests_dir / 'train.jsonl'} count={len(train_rows)}")
    print(f"valid_manifest={manifests_dir / 'valid.jsonl'} count={len(valid_rows)}")


if __name__ == "__main__":
    main()
