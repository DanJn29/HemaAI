from __future__ import annotations

import argparse
import json

from app.ml.training.pipeline import TrainingConfig, TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiclass ML models on a synthetic HemaAI CBC dataset.",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset-variant", choices=["strict", "default"], default="default")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        choices=["raw_only", "hybrid"],
        default=["raw_only", "hybrid"],
    )
    parser.add_argument("--include-rule-score-experiment", action="store_true")
    parser.add_argument("--with-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TrainingPipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        dataset_variant=args.dataset_variant,
        config=TrainingConfig(seed=args.seed),
    )
    results = pipeline.run(
        feature_modes=args.feature_modes,
        include_rule_score_experiment=args.include_rule_score_experiment,
        with_shap=args.with_shap,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
