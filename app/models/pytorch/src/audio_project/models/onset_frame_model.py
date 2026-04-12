from __future__ import annotations

import torch
from torch import nn

from audio_project.config import AudioConfig, ModelConfig
from audio_project.models.base import TranscriptionModel


class OnsetFrameModel(TranscriptionModel):
    def __init__(self, audio_config: AudioConfig, num_pitches: int, config: ModelConfig) -> None:
        super().__init__()
        if config.feature_mode not in {"mel", "hybrid"}:
            raise ValueError(f"Unsupported feature mode: {config.feature_mode}")
        if config.decoder_mode not in {"baseline", "onset_conditioned"}:
            raise ValueError(f"Unsupported decoder mode: {config.decoder_mode}")
        self.feature_mode = config.feature_mode
        self.decoder_mode = config.decoder_mode
        self.mel_dim = audio_config.n_mels
        self.chroma_dim = audio_config.n_chroma
        self.mel_conv = nn.Sequential(
            nn.Conv1d(self.mel_dim, config.mel_conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.mel_conv_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.mel_conv_channels, config.mel_conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.mel_conv_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.mel_conv_channels, config.mel_conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(config.mel_conv_channels),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        if self.feature_mode == "hybrid":
            self.chroma_conv = nn.Sequential(
                nn.Conv1d(self.chroma_dim, config.chroma_conv_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(config.chroma_conv_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(config.chroma_conv_channels, config.chroma_conv_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(config.chroma_conv_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(config.chroma_conv_channels, config.chroma_conv_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(config.chroma_conv_channels),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(
                    config.mel_conv_channels + config.chroma_conv_channels,
                    config.fusion_channels,
                    kernel_size=1,
                ),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            encoder_input_size = config.fusion_channels
        else:
            self.chroma_conv = None
            self.fusion = None
            encoder_input_size = config.mel_conv_channels
        self.encoder = nn.GRU(
            input_size=encoder_input_size,
            hidden_size=config.encoder_hidden_size,
            num_layers=config.encoder_layers,
            dropout=config.dropout if config.encoder_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        hidden_dim = config.encoder_hidden_size * 2
        if self.decoder_mode == "onset_conditioned":
            self.onset_projection = nn.Sequential(
                nn.Linear(hidden_dim, config.onset_conditioning_hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.onset_head = nn.Linear(config.onset_conditioning_hidden_size, num_pitches)
            self.frame_projection = nn.Sequential(
                nn.Linear(hidden_dim + num_pitches, config.onset_conditioning_hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            self.frame_head = nn.Linear(config.onset_conditioning_hidden_size, num_pitches)
        else:
            self.onset_projection = None
            self.frame_projection = None
            self.onset_head = nn.Linear(hidden_dim, num_pitches)
            self.frame_head = nn.Linear(hidden_dim, num_pitches)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        mel_features = features[..., : self.mel_dim].transpose(1, 2)
        mel_encoded = self.mel_conv(mel_features)
        if self.feature_mode == "hybrid":
            chroma_features = features[..., self.mel_dim :].transpose(1, 2)
            chroma_encoded = self.chroma_conv(chroma_features)
            encoded_input = self.fusion(torch.cat([mel_encoded, chroma_encoded], dim=1))
        else:
            encoded_input = mel_encoded
        x = encoded_input.transpose(1, 2)
        encoded, _ = self.encoder(x)
        if self.decoder_mode == "onset_conditioned":
            onset_hidden = self.onset_projection(encoded)
            onset_logits = self.onset_head(onset_hidden)
            frame_input = torch.cat([encoded, onset_logits.sigmoid()], dim=-1)
            frame_hidden = self.frame_projection(frame_input)
            frame_logits = self.frame_head(frame_hidden)
        else:
            onset_logits = self.onset_head(encoded)
            frame_logits = self.frame_head(encoded)
        return {
            "onset_logits": onset_logits,
            "frame_logits": frame_logits,
            "lengths": feature_lengths,
        }
