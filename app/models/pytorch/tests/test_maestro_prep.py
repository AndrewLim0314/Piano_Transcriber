from __future__ import annotations

import json
from pathlib import Path

from audio_project.data.maestro import prepare_maestro_dataset


def test_prepare_maestro_dataset_writes_split_manifests(tmp_path: Path) -> None:
    dataset_root = tmp_path / "maestro-v3.0.0"
    dataset_root.mkdir()

    audio_a = dataset_root / "2004" / "a.wav"
    midi_a = dataset_root / "2004" / "a.mid"
    audio_b = dataset_root / "2005" / "b.wav"
    midi_b = dataset_root / "2005" / "b.mid"
    audio_c = dataset_root / "2006" / "c.wav"
    midi_c = dataset_root / "2006" / "c.mid"

    for path in (audio_a, midi_a, audio_b, midi_b, audio_c, midi_c):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    metadata_csv = dataset_root / "maestro-v3.0.0.csv"
    metadata_csv.write_text(
        "\n".join(
            [
                "canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration",
                "Composer A,Piece A,train,2004,2004/a.mid,2004/a.wav,10.0",
                "Composer B,Piece B,validation,2005,2005/b.mid,2005/b.wav,20.0",
                "Composer C,Piece C,test,2006,2006/c.mid,2006/c.wav,30.0",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "manifests"
    manifest_paths, summary = prepare_maestro_dataset(
        dataset_root=dataset_root,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
    )

    assert manifest_paths["train"].exists()
    assert manifest_paths["valid"].exists()
    assert manifest_paths["test"].exists()
    assert summary["train"]["count"] == 1
    assert summary["valid"]["count"] == 1
    assert summary["test"]["count"] == 1

    train_row = json.loads(manifest_paths["train"].read_text(encoding="utf-8").strip())
    valid_row = json.loads(manifest_paths["valid"].read_text(encoding="utf-8").strip())
    test_row = json.loads(manifest_paths["test"].read_text(encoding="utf-8").strip())

    assert train_row["audio_path"] == str(audio_a.resolve())
    assert valid_row["midi_path"] == str(midi_b.resolve())
    assert test_row["split"] == "test"
