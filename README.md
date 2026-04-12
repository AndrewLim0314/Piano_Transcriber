# Piano Transcriber

Converts a piano audio recording into sheet music. Upload an audio file, get back a PDF score, MusicXML, and MIDI.

---

## Pipeline

```
Audio file (WAV/MP3/FLAC)
        │
        ▼
  File validation & storage
        │
        ▼
  Audio preprocessing
  ├─ Resample to 16 kHz mono
  ├─ 128-band Mel spectrogram (dB)
  └─ 12-band chroma features
        │ 140-dim feature vector per 16 ms frame
        ▼
  OnsetFrameModel (PyTorch)
  ├─ Mel Conv1D branch  (3 layers, 128 ch)
  ├─ Chroma Conv1D branch (3 layers, 32 ch)
  ├─ Fusion Conv1D → 160 ch
  ├─ Bidirectional GRU (3 layers, hidden 384 → 768-dim output)
  ├─ Onset head  → 88 per-pitch onset probabilities
  └─ Frame head  → 88 per-pitch sustain probabilities
        │            (conditioned on onset predictions)
        ▼
  Decoding
  ├─ Onset threshold 0.80, frame threshold 0.50
  ├─ Local peak detection (no double-firing)
  └─ Postprocess: merge gaps, drop notes < 40 ms
        │ NoteEvent list (pitch, start, end, velocity)
        ▼
  MIDI export (.mid)
        │
        ▼
  MusicXML generation (music21)
  ├─ Quantize to 16th-note grid
  ├─ Key detection
  ├─ Grand staff split at middle C
  │   ├─ Notes ≥ C4 → treble clef
  │   └─ Notes < C4 → bass clef
  └─ Voice renumbering (MusicXML requires voices ≥ 1)
        │
        ▼
  PDF rendering (MuseScore CLI)
        │
        ▼
  MIDI + MusicXML + PDF returned via API
```

---

## Model Architecture

The neural model (`OnsetFrameModel`) uses two parallel convolutional branches — one for timbral content (mel spectrogram) and one for harmonic/pitch content (chroma) — that merge before a shared recurrent encoder.

```
Mel (128-dim) ──→ Conv1D × 3 (128 ch, kernels 5-5-3) ─┐
                                                         ├─→ Fusion Conv1D (160 ch)
Chroma (12-dim) → Conv1D × 3  (32 ch, kernels 7-7-3) ─┘
                                                         │
                                                         ▼
                                      Bidirectional GRU, 3 layers
                                      hidden=384 per direction → 768-dim
                                                         │
                              ┌──────────────────────────┤
                              ▼                          ▼
                       Onset head                  Frame head
                  Linear(768→256→88)     Linear(768 + 88 → 256 → 88)
                  onset probability       sustain probability
                  per pitch per frame     conditioned on onset output
```

The onset-conditioning on the frame head enforces consistency: a note can only sustain if it was first detected as an onset.

**Config:**
- Sample rate: 16 kHz
- FFT: n_fft=1024, hop=256 (16 ms per frame)
- Piano range: MIDI 21–108 (88 keys)
- Onset threshold: 0.80 (tuned for F1=0.924 on validation set)
- Inference: 12-second chunks with 1-second overlap

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transcription/upload` | Upload audio file, returns job ID |
| GET | `/transcription/{id}` | Poll status, get download URLs |
| GET | `/transcription/` | List all transcriptions |
| DELETE | `/transcription/{id}` | Delete a transcription |
| GET | `/health/` | Health check |

Responses include URLs for `.mid`, `.musicxml`, and `.pdf` outputs plus a structured note list (pitch name, MIDI number, start time, end time, duration, velocity).

Processing runs as a background task. Status values: `pending` → `processing` → `completed` / `failed`.

---

## Outputs

Each transcription produces three files:

- **MIDI** — raw note events from the model, standard `.mid`
- **MusicXML** — quantized, key-detected, grand staff notation
- **PDF** — typeset sheet music rendered by MuseScore

---

## Showcase

Example transcriptions of known melodies using the synthetic piano synthesizer:

| Melody | Notes detected | Files |
|--------|---------------|-------|
| 7 Nation Army | 10 | `artifacts/showcase/7_nation_army.pdf` |
| Iron Man | 11 | `artifacts/showcase/iron_man.pdf` |
| Smoke on the Water | 22 | `artifacts/showcase/smoke_on_water.pdf` |

Piano roll comparisons and training curves are in `artifacts/showcase/`.

---

## Project Structure

```
app/
  routers/          — FastAPI route handlers
  services/
    transcription_service.py  — full pipeline orchestration
    audio_processor.py        — librosa fallback (monophonic)
  models/
    pytorch/
      src/audio_project/
        models/onset_frame_model.py   — neural network
        data/audio.py                 — preprocessing
        inference/service.py          — transcription + decoding
        inference/export.py           — MIDI export
        config.py                     — all hyperparameters
      artifacts/
        run_v3/                       — training checkpoints
        showcase/                     — example outputs
      scripts/
        generate_synthetic_piano_data.py
        evaluate_checkpoint.py
        generate_showcase.py
  utils/file_handler.py
  db/database.py
  config.py
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: point to a specific checkpoint
echo "NEURAL_MODEL_CHECKPOINT=./app/models/pytorch/artifacts/run_v3/music_transcription_epoch_020.pt" >> .env

# Run
uvicorn app.main:app --reload
```

MuseScore must be installed for PDF output (`mscore` on PATH). PDF is skipped gracefully if not available.

If no checkpoint is configured, the model runs with random weights. The librosa-based monophonic fallback activates if PyTorch is unavailable.
