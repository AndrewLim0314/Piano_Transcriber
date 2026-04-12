from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    n_mels: int = 128
    n_chroma: int = 12
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256


@dataclass(slots=True)
class ModelConfig:
    feature_mode: str = "hybrid"
    decoder_mode: str = "onset_conditioned"
    mel_conv_channels: int = 128
    chroma_conv_channels: int = 32
    fusion_channels: int = 160
    encoder_hidden_size: int = 384
    encoder_layers: int = 3
    onset_conditioning_hidden_size: int = 256
    dropout: float = 0.2


@dataclass(slots=True)
class LabelConfig:
    min_midi: int = 21
    max_midi: int = 108

    @property
    def num_pitches(self) -> int:
        return self.max_midi - self.min_midi + 1


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    onset_loss_weight: float = 6.0
    frame_loss_weight: float = 1.0
    gradient_clip_norm: float = 1.0
    focal_gamma: float = 2.0
    device: str = "cpu"
    checkpoint_dir: Path = Path("artifacts/checkpoints")


@dataclass(slots=True)
class DataConfig:
    train_manifest: Path = Path("data/train.jsonl")
    valid_manifest: Path = Path("data/valid.jsonl")
    test_manifest: Path = Path("data/test.jsonl")


@dataclass(slots=True)
class InferenceConfig:
    chunk_duration_seconds: float = 12.0
    chunk_overlap_seconds: float = 1.0
    onset_threshold: float = 0.80
    frame_threshold: float = 0.5
    min_note_duration_seconds: float = 0.04
    max_onsets_per_frame: int = 6


@dataclass(slots=True)
class ProjectConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
