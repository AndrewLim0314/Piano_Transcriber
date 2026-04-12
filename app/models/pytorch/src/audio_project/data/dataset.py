from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from audio_project.data.audio import AudioPreprocessor
from audio_project.data.midi import load_note_events, note_events_to_targets
from audio_project.schemas import MusicBatch, MusicSample


class AudioManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        preprocessor: AudioPreprocessor,
        min_midi: int,
        max_midi: int,
    ) -> None:
        self.preprocessor = preprocessor
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.samples = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: Path) -> list[MusicSample]:
        samples: list[MusicSample] = []
        manifest_root = manifest_path.parent
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                audio_path = Path(row["audio_path"])
                midi_path = Path(row["midi_path"])
                if not audio_path.is_absolute():
                    audio_path = manifest_root / audio_path
                if not midi_path.is_absolute():
                    midi_path = manifest_root / midi_path
                samples.append(
                    MusicSample(
                        audio_path=audio_path,
                        midi_path=midi_path,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        features = self.preprocessor.load_features(sample.audio_path)
        note_events = load_note_events(sample.midi_path)
        onset_targets, frame_targets = note_events_to_targets(
            note_events=note_events,
            num_frames=features.size(0),
            sample_rate=self.preprocessor.config.sample_rate,
            hop_length=self.preprocessor.config.hop_length,
            min_midi=self.min_midi,
            max_midi=self.max_midi,
        )
        return features, onset_targets, frame_targets


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> MusicBatch:
    feature_tensors = [item[0] for item in batch]
    feature_lengths = torch.tensor([tensor.size(0) for tensor in feature_tensors], dtype=torch.long)
    padded_features = pad_sequence(feature_tensors, batch_first=True)
    padded_onsets = pad_sequence([item[1] for item in batch], batch_first=True)
    padded_frames = pad_sequence([item[2] for item in batch], batch_first=True)
    time_steps = padded_features.size(1)
    frame_mask = torch.arange(time_steps).unsqueeze(0) < feature_lengths.unsqueeze(1)

    return MusicBatch(
        features=padded_features,
        feature_lengths=feature_lengths,
        onset_targets=padded_onsets,
        frame_targets=padded_frames,
        frame_mask=frame_mask,
    )
