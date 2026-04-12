from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from os import path as os_path
from pathlib import Path


SPLIT_MAP = {
    "train": "train",
    "validation": "valid",
    "valid": "valid",
    "test": "test",
}


@dataclass(slots=True)
class MaestroRecord:
    split: str
    audio_path: Path
    midi_path: Path
    duration: float
    canonical_composer: str
    canonical_title: str
    year: int


def normalize_split(raw_split: str) -> str:
    split = raw_split.strip().lower()
    if split not in SPLIT_MAP:
        raise ValueError(f"Unsupported split value: {raw_split!r}")
    return SPLIT_MAP[split]


def read_maestro_metadata(metadata_csv: Path, dataset_root: Path) -> list[MaestroRecord]:
    records: list[MaestroRecord] = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = normalize_split(row["split"])
            records.append(
                MaestroRecord(
                    split=split,
                    audio_path=(dataset_root / row["audio_filename"]).resolve(),
                    midi_path=(dataset_root / row["midi_filename"]).resolve(),
                    duration=float(row["duration"]),
                    canonical_composer=row["canonical_composer"],
                    canonical_title=row["canonical_title"],
                    year=int(row["year"]),
                )
            )
    return records


def summarize_records(records: list[MaestroRecord]) -> dict[str, dict[str, float]]:
    summary = {
        "train": {"count": 0, "hours": 0.0},
        "valid": {"count": 0, "hours": 0.0},
        "test": {"count": 0, "hours": 0.0},
    }
    for record in records:
        summary[record.split]["count"] += 1
        summary[record.split]["hours"] += record.duration / 3600.0
    return summary


def write_maestro_manifests(
    records: list[MaestroRecord],
    output_dir: Path,
    relative_paths: bool = False,
    validate_files: bool = True,
    dataset_name: str = "maestro-v3.0.0",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        "train": output_dir / "train.jsonl",
        "valid": output_dir / "valid.jsonl",
        "test": output_dir / "test.jsonl",
    }

    handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in manifest_paths.items()
    }
    try:
        for record in records:
            if validate_files:
                if not record.audio_path.exists():
                    raise FileNotFoundError(f"Missing audio file: {record.audio_path}")
                if not record.midi_path.exists():
                    raise FileNotFoundError(f"Missing MIDI file: {record.midi_path}")

            manifest_path = manifest_paths[record.split]
            row = {
                "audio_path": _format_path(record.audio_path, manifest_path.parent, relative_paths),
                "midi_path": _format_path(record.midi_path, manifest_path.parent, relative_paths),
                "duration": record.duration,
                "split": record.split,
                "canonical_composer": record.canonical_composer,
                "canonical_title": record.canonical_title,
                "year": record.year,
                "dataset": dataset_name,
            }
            handles[record.split].write(json.dumps(row) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    return manifest_paths


def prepare_maestro_dataset(
    dataset_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    relative_paths: bool = False,
    validate_files: bool = True,
    dataset_name: str = "maestro-v3.0.0",
) -> tuple[dict[str, Path], dict[str, dict[str, float]]]:
    records = read_maestro_metadata(metadata_csv=metadata_csv, dataset_root=dataset_root)
    manifest_paths = write_maestro_manifests(
        records=records,
        output_dir=output_dir,
        relative_paths=relative_paths,
        validate_files=validate_files,
        dataset_name=dataset_name,
    )
    summary = summarize_records(records)
    return manifest_paths, summary


def _format_path(path: Path, manifest_dir: Path, relative_paths: bool) -> str:
    if relative_paths:
        return os_path.relpath(path, start=manifest_dir)
    return str(path)
