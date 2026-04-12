from __future__ import annotations

from pathlib import Path

import torch

from audio_project.config import ProjectConfig
from audio_project.data.audio import AudioPreprocessor
from audio_project.inference.postprocess import postprocess_note_events
from audio_project.models.onset_frame_model import OnsetFrameModel
from audio_project.schemas import NoteEvent


class TranscriptionService:
    def __init__(self, config: ProjectConfig | None = None, checkpoint_path: Path | None = None) -> None:
        self.config = config or ProjectConfig()
        self.preprocessor = AudioPreprocessor(self.config.audio)
        self.model = OnsetFrameModel(
            audio_config=self.config.audio,
            num_pitches=self.config.labels.num_pitches,
            config=self.config.model,
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def transcribe_file(self, audio_path: Path) -> list[NoteEvent]:
        waveform = self.preprocessor.load_waveform(audio_path)
        return self.transcribe_waveform(waveform, self.config.audio.sample_rate)

    @torch.inference_mode()
    def transcribe_waveform(self, waveform: torch.Tensor, sample_rate: int) -> list[NoteEvent]:
        prepared = self.preprocessor.prepare_waveform(waveform, sample_rate)
        return self._transcribe_prepared_waveform(prepared)

    def _transcribe_prepared_waveform(self, waveform: torch.Tensor) -> list[NoteEvent]:
        chunked_events: list[NoteEvent] = []
        sample_rate = self.config.audio.sample_rate
        chunk_samples = max(1, int(self.config.inference.chunk_duration_seconds * sample_rate))
        overlap_samples = max(0, int(self.config.inference.chunk_overlap_seconds * sample_rate))
        step_samples = max(1, chunk_samples - overlap_samples)
        total_samples = waveform.size(1)

        for start_sample in range(0, total_samples, step_samples):
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk = waveform[:, start_sample:end_sample]
            if chunk.numel() == 0:
                continue
            events = self._transcribe_chunk(
                chunk,
                time_offset_seconds=start_sample / sample_rate,
            )
            chunked_events.extend(events)
            if end_sample >= total_samples:
                break

        return postprocess_note_events(
            chunked_events,
            min_note_duration_seconds=self.config.inference.min_note_duration_seconds,
            merge_gap_seconds=self.config.inference.chunk_overlap_seconds / 2.0,
        )

    def predict_file(self, audio_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self.preprocessor.load_waveform(audio_path)
        prepared = self.preprocessor.prepare_waveform(waveform, self.config.audio.sample_rate)
        return self._predict_prepared_waveform(prepared)

    def _predict_prepared_waveform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.preprocessor.waveform_to_features(waveform).unsqueeze(0)
        lengths = torch.tensor([features.size(1)], dtype=torch.long)
        outputs = self.model(features, lengths)
        onset_probs = outputs["onset_logits"].sigmoid().squeeze(0).cpu()
        frame_probs = outputs["frame_logits"].sigmoid().squeeze(0).cpu()
        return onset_probs, frame_probs

    def decode_predictions(
        self,
        onset_probs: torch.Tensor,
        frame_probs: torch.Tensor,
        time_offset_seconds: float = 0.0,
    ) -> list[NoteEvent]:
        events = decode_note_events(
            onset_probs=onset_probs,
            frame_probs=frame_probs,
            hop_length=self.config.audio.hop_length,
            sample_rate=self.config.audio.sample_rate,
            min_midi=self.config.labels.min_midi,
            onset_threshold=self.config.inference.onset_threshold,
            frame_threshold=self.config.inference.frame_threshold,
            max_onsets_per_frame=self.config.inference.max_onsets_per_frame,
        )
        shifted = [
            NoteEvent(
                pitch=event.pitch,
                start_time=event.start_time + time_offset_seconds,
                end_time=event.end_time + time_offset_seconds,
                velocity=event.velocity,
            )
            for event in events
        ]
        return postprocess_note_events(
            shifted,
            min_note_duration_seconds=self.config.inference.min_note_duration_seconds,
            merge_gap_seconds=self.config.inference.chunk_overlap_seconds / 2.0,
        )

    def _transcribe_chunk(self, waveform: torch.Tensor, time_offset_seconds: float) -> list[NoteEvent]:
        onset_probs, frame_probs = self._predict_prepared_waveform(waveform)
        return self.decode_predictions(
            onset_probs=onset_probs,
            frame_probs=frame_probs,
            time_offset_seconds=time_offset_seconds,
        )


def decode_note_events(
    onset_probs: torch.Tensor,
    frame_probs: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    min_midi: int,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    max_onsets_per_frame: int = 6,
) -> list[NoteEvent]:
    note_events: list[NoteEvent] = []
    seconds_per_frame = hop_length / sample_rate
    active_notes: dict[int, float] = {}
    num_frames, num_pitches = onset_probs.shape

    for frame_index in range(num_frames):
        current_time = frame_index * seconds_per_frame
        frame_onsets = onset_probs[frame_index]
        candidate_indices = torch.nonzero(frame_onsets >= onset_threshold, as_tuple=False).squeeze(1).tolist()
        if len(candidate_indices) > max_onsets_per_frame:
            candidate_indices = sorted(
                candidate_indices,
                key=lambda index: frame_onsets[index].item(),
                reverse=True,
            )[:max_onsets_per_frame]
        candidate_set = set(candidate_indices)
        for pitch_index in range(num_pitches):
            pitch = min_midi + pitch_index
            onset_active = pitch_index in candidate_set and _is_local_onset_peak(
                onset_probs=onset_probs,
                frame_index=frame_index,
                pitch_index=pitch_index,
            )
            frame_active = frame_probs[frame_index, pitch_index].item() >= frame_threshold
            is_active = pitch in active_notes

            if onset_active and is_active:
                note_events.append(
                    NoteEvent(
                        pitch=pitch,
                        start_time=active_notes[pitch],
                        end_time=max(current_time, active_notes[pitch] + seconds_per_frame),
                    )
                )
                active_notes[pitch] = current_time
                continue

            if onset_active and not is_active:
                active_notes[pitch] = current_time
                continue

            if is_active and not frame_active:
                start_time = active_notes.pop(pitch)
                note_events.append(
                    NoteEvent(
                        pitch=pitch,
                        start_time=start_time,
                        end_time=max(current_time, start_time + seconds_per_frame),
                    )
                )

    final_time = num_frames * seconds_per_frame
    for pitch, start_time in active_notes.items():
        note_events.append(
            NoteEvent(
                pitch=pitch,
                start_time=start_time,
                end_time=max(final_time, start_time + seconds_per_frame),
            )
        )

    return sorted(note_events, key=lambda event: (event.start_time, event.pitch))


def _is_local_onset_peak(onset_probs: torch.Tensor, frame_index: int, pitch_index: int) -> bool:
    value = onset_probs[frame_index, pitch_index].item()
    previous_value = onset_probs[frame_index - 1, pitch_index].item() if frame_index > 0 else 0.0
    next_value = onset_probs[frame_index + 1, pitch_index].item() if frame_index + 1 < onset_probs.size(0) else value
    return value > previous_value and value >= next_value
