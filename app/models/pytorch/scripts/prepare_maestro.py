from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_project.data.maestro import prepare_maestro_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MAESTRO manifests for training.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the extracted MAESTRO directory.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Path to maestro-v3.0.0.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train/valid/test manifests.")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Write paths relative to each manifest directory instead of absolute paths.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip checking whether every audio and MIDI file exists.",
    )
    parser.add_argument(
        "--dataset-name",
        default="maestro-v3.0.0",
        help="Dataset label to embed in each manifest row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_paths, summary = prepare_maestro_dataset(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        relative_paths=args.relative_paths,
        validate_files=not args.skip_validation,
        dataset_name=args.dataset_name,
    )

    for split in ("train", "valid", "test"):
        stats = summary[split]
        print(
            f"{split}: count={int(stats['count'])} "
            f"hours={stats['hours']:.2f} "
            f"manifest={manifest_paths[split]}"
        )


if __name__ == "__main__":
    main()
