from __future__ import annotations

import argparse
import json

from app.core.db import get_session_factory, initialize_engine
from app.ml.generation.dataset_builder import SyntheticDatasetBuilder
from app.ml.generation.generator import SyntheticGenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a medically plausible synthetic CBC dataset for HemaAI.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--max-attempt-multiplier", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the synthetic dataset exports should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initialize_engine()
    session_factory = get_session_factory()

    config = SyntheticGenerationConfig(
        seed=args.seed,
        samples_per_class=args.samples_per_class,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )
    with session_factory() as session:
        builder = SyntheticDatasetBuilder(session=session, config=config)
        bundle = builder.build()
        builder.export(args.output_dir, bundle)
        print(json.dumps(bundle.summary, indent=2))


if __name__ == "__main__":
    main()
