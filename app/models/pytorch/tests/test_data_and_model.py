from __future__ import annotations

import json
from pathlib import Path
import wave

import torch

from audio_project.config import ProjectConfig
from audio_project.data.audio import AudioPreprocessor
from audio_project.data.dataset import AudioManifestDataset, collate_batch
from audio_project.data.midi import note_events_to_targets
from audio_project.inference.postprocess import merge_nearby_events
from audio_project.inference.service import decode_note_events, TranscriptionService
from audio_project.models.onset_frame_model import OnsetFrameModel
from audio_project.schemas import NoteEvent
from audio_project.training.train import compute_losses


class StubPreprocessor:
    def load_features(self, audio_path: Path) -> torch.Tensor:
        del audio_path
        return torch.randn(12, 140)


def test_manifest_paths_resolve_relative_to_manifest(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "train.jsonl"
    audio_dir = tmp_path / "audio"
    midi_dir = tmp_path / "midi"
    audio_dir.mkdir()
    midi_dir.mkdir()
    audio_path = audio_dir / "sample.wav"
    midi_path = midi_dir / "sample.mid"
    audio_path.write_bytes(b"")
    midi_path.write_bytes(b"")
    manifest.write_text(
        json.dumps({"audio_path": "audio/sample.wav", "midi_path": "midi/sample.mid"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("audio_project.data.dataset.load_note_events", lambda _: [])

    dataset = AudioManifestDataset(
        manifest_path=manifest,
        preprocessor=StubPreprocessor(),
        min_midi=21,
        max_midi=108,
    )

    assert dataset.samples[0].audio_path == audio_path
    assert dataset.samples[0].midi_path == midi_path


def test_note_events_to_targets_marks_onsets_and_frames() -> None:
    onset_targets, frame_targets = note_events_to_targets(
        note_events=[NoteEvent(pitch=60, start_time=0.0, end_time=0.1)],
        num_frames=10,
        sample_rate=100,
        hop_length=10,
        min_midi=21,
        max_midi=108,
    )

    pitch_index = 60 - 21
    assert onset_targets[0, pitch_index].item() == 1.0
    assert frame_targets[:1, pitch_index].sum().item() >= 1.0


def test_collate_batch_pads_features_and_targets() -> None:
    batch = [
        (torch.randn(10, 140), torch.zeros(10, 88), torch.zeros(10, 88)),
        (torch.randn(6, 140), torch.zeros(6, 88), torch.zeros(6, 88)),
    ]

    result = collate_batch(batch)

    assert result.features.shape == (2, 10, 140)
    assert result.feature_lengths.tolist() == [10, 6]
    assert result.onset_targets.shape == (2, 10, 88)
    assert result.frame_targets.shape == (2, 10, 88)
    assert result.frame_mask.shape == (2, 10)


def test_onset_frame_model_forward_returns_expected_shapes() -> None:
    config = ProjectConfig()
    model = OnsetFrameModel(
        audio_config=config.audio,
        num_pitches=config.labels.num_pitches,
        config=config.model,
    )
    features = torch.randn(2, 15, config.audio.n_mels + config.audio.n_chroma)
    lengths = torch.tensor([15, 12], dtype=torch.long)

    outputs = model(features, lengths)

    assert outputs["onset_logits"].shape == (2, 15, config.labels.num_pitches)
    assert outputs["frame_logits"].shape == (2, 15, config.labels.num_pitches)
    assert outputs["lengths"].tolist() == [15, 12]


def test_compute_losses_returns_finite_value() -> None:
    config = ProjectConfig()
    model = OnsetFrameModel(
        audio_config=config.audio,
        num_pitches=config.labels.num_pitches,
        config=config.model,
    )
    features = torch.randn(2, 18, config.audio.n_mels + config.audio.n_chroma)
    feature_lengths = torch.tensor([18, 16], dtype=torch.long)
    outputs = model(features, feature_lengths)
    onset_targets = torch.zeros(2, 18, config.labels.num_pitches)
    frame_targets = torch.zeros(2, 18, config.labels.num_pitches)
    frame_mask = torch.ones(2, 18, dtype=torch.bool)

    loss = compute_losses(
        config=config,
        onset_logits=outputs["onset_logits"],
        frame_logits=outputs["frame_logits"],
        onset_targets=onset_targets,
        frame_targets=frame_targets,
        frame_mask=frame_mask,
    )

    assert torch.isfinite(loss)


def test_decode_note_events_creates_note_boundaries() -> None:
    onset_probs = torch.zeros(5, 88)
    frame_probs = torch.zeros(5, 88)
    pitch_index = 60 - 21
    onset_probs[1, pitch_index] = 0.9
    frame_probs[1:4, pitch_index] = 0.9

    events = decode_note_events(
        onset_probs=onset_probs,
        frame_probs=frame_probs,
        hop_length=160,
        sample_rate=16000,
        min_midi=21,
    )

    assert len(events) == 1
    assert events[0].pitch == 60
    assert events[0].end_time > events[0].start_time


def test_decode_note_events_limits_dense_onsets_per_frame() -> None:
    onset_probs = torch.full((3, 88), 0.9)
    frame_probs = torch.zeros(3, 88)
    frame_probs[:, :10] = 0.9

    events = decode_note_events(
        onset_probs=onset_probs,
        frame_probs=frame_probs,
        hop_length=160,
        sample_rate=16000,
        min_midi=21,
        onset_threshold=0.5,
        frame_threshold=0.5,
        max_onsets_per_frame=4,
    )

    assert len(events) <= 4


def test_load_waveform_handles_real_wav_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "example.wav"
    sample_rate = 8_000
    samples = [0, 4096, -4096, 2048, -2048, 0] * 200
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frame_bytes = b"".join(int(sample).to_bytes(2, byteorder="little", signed=True) for sample in samples)
        handle.writeframes(frame_bytes)

    preprocessor = AudioPreprocessor(ProjectConfig().audio)
    waveform = preprocessor.load_waveform(audio_path)
    features = preprocessor.waveform_to_features(waveform)

    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
    assert features.shape[1] == preprocessor.feature_dim


def test_merge_overlapping_events_collapses_chunk_duplicates() -> None:
    events = [
        NoteEvent(pitch=60, start_time=0.50, end_time=0.90),
        NoteEvent(pitch=60, start_time=0.82, end_time=1.10),
        NoteEvent(pitch=64, start_time=1.20, end_time=1.30),
    ]

    merged = merge_nearby_events(
        events,
        min_gap_seconds=0.2,
        min_note_duration_seconds=0.01,
    )

    assert len(merged) == 2
    assert merged[0].pitch == 60
    assert merged[0].start_time == 0.50
    assert merged[0].end_time == 1.10


def test_transcribe_waveform_chunks_longer_audio(monkeypatch) -> None:
    config = ProjectConfig()
    config.inference.chunk_duration_seconds = 1.0
    config.inference.chunk_overlap_seconds = 0.25
    service = TranscriptionService(config=config)

    calls: list[float] = []

    def fake_chunk(self, waveform: torch.Tensor, time_offset_seconds: float) -> list[NoteEvent]:
        del waveform
        calls.append(time_offset_seconds)
        return [NoteEvent(pitch=60, start_time=time_offset_seconds, end_time=time_offset_seconds + 0.5)]

    monkeypatch.setattr(TranscriptionService, "_transcribe_chunk", fake_chunk)
    waveform = torch.zeros(1, int(2.2 * config.audio.sample_rate))

    events = service.transcribe_waveform(waveform, config.audio.sample_rate)

    assert len(calls) >= 2
    assert events[0].pitch == 60
