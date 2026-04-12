from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

@dataclass(slots=True)
class MusicSample:
    audio_path: Path
    midi_path: Path


@dataclass(slots=True)
class NoteEvent:
    pitch: int
    start_time: float
    end_time: float
    velocity: int = 64


@dataclass(slots=True)
class MusicBatch:
    features: torch.Tensor
    feature_lengths: torch.Tensor
    onset_targets: torch.Tensor
    frame_targets: torch.Tensor
    frame_mask: torch.Tensor
