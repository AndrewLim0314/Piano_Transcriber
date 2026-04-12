from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class TranscriptionModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return model outputs and updated sequence lengths."""
