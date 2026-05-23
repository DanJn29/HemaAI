from __future__ import annotations

import argparse
import json

from app.ml.training.model_selection import write_best_model_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best trained HemaAI ML model and write best_model.json.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset-variants",
        nargs="+",
        default=["strict", "default"],
        choices=["strict", "default"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = write_best_model_metadata(
        output_dir=args.output_dir,
        dataset_variants=tuple(args.dataset_variants),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
