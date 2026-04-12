from __future__ import annotations

import array
import wave
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from audio_project.config import AudioConfig


class AudioPreprocessor:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
        )
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.chroma_filterbank = self._build_chroma_filterbank()

    @property
    def feature_dim(self) -> int:
        return self.config.n_mels + self.config.n_chroma

    def load_waveform(self, audio_path: Path) -> torch.Tensor:
        waveform, sample_rate = self._load_audio_file(audio_path)
        return self.prepare_waveform(waveform, sample_rate)

    def prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(torch.float32)
        peak = waveform.abs().max().item()
        if peak > 0:
            waveform = waveform / peak
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )
        return waveform.contiguous()

    def waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_features = self.mel_transform(waveform)
        mel_features = self.amplitude_to_db(mel_features)
        mel_features = mel_features.squeeze(0).transpose(0, 1)

        spectrogram = self.spectrogram_transform(waveform).squeeze(0).transpose(0, 1)
        chroma_features = spectrogram @ self.chroma_filterbank.to(spectrogram.device)
        chroma_features = torch.log1p(chroma_features)

        mel_features = self._normalize_features(mel_features)
        chroma_features = self._normalize_features(chroma_features)
        return torch.cat([mel_features, chroma_features], dim=-1)

    def load_features(self, audio_path: Path) -> torch.Tensor:
        waveform = self.load_waveform(audio_path)
        return self.waveform_to_features(waveform)

    def _load_audio_file(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        if audio_path.suffix.lower() == ".wav":
            return self._load_wav_file(audio_path)

        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform, sample_rate
        except Exception:  # noqa: BLE001
            return self._load_with_soundfile(audio_path)

    def _load_wav_file(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        with wave.open(str(audio_path), "rb") as handle:
            sample_rate = handle.getframerate()
            num_channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            num_frames = handle.getnframes()
            raw_audio = handle.readframes(num_frames)

        if sample_width != 2:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        pcm = array.array("h")
        pcm.frombytes(raw_audio)
        waveform = torch.tensor(pcm, dtype=torch.int16).to(torch.float32) / 32768.0
        waveform = waveform.view(-1, num_channels).transpose(0, 1).contiguous()
        return waveform, sample_rate

    def _load_with_soundfile(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(audio).transpose(0, 1).contiguous()
        return waveform, sample_rate

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - features.mean()) / (features.std() + 1e-6)

    def _build_chroma_filterbank(self) -> torch.Tensor:
        num_frequency_bins = (self.config.n_fft // 2) + 1
        frequencies = torch.linspace(0.0, self.config.sample_rate / 2.0, num_frequency_bins)
        chroma = torch.zeros(num_frequency_bins, self.config.n_chroma, dtype=torch.float32)

        valid = frequencies > 0
        midi = 69.0 + 12.0 * torch.log2(frequencies[valid] / 440.0)
        chroma_index = torch.remainder(torch.round(midi), self.config.n_chroma).to(torch.long)
        bin_distance = torch.abs(midi - torch.round(midi))
        weights = torch.exp(-0.5 * (bin_distance / 0.35) ** 2)

        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        chroma[valid_indices, chroma_index] = weights
        column_sums = chroma.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return chroma / column_sums
