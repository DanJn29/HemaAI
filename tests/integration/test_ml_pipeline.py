import json
from pathlib import Path

import pandas as pd

from app.ml.generation.dataset_builder import SyntheticDatasetBuilder
from app.ml.generation.generator import SyntheticGenerationConfig
from app.ml.training.pipeline import TrainingConfig, TrainingPipeline


def test_synthetic_dataset_and_training_pipeline_smoke(db_session, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"

    builder = SyntheticDatasetBuilder(
        session=db_session,
        config=SyntheticGenerationConfig(seed=42, samples_per_class=16),
    )
    bundle = builder.build()
    builder.export(dataset_dir, bundle)

    pipeline = TrainingPipeline(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        dataset_variant="default",
        config=TrainingConfig(
            seed=42,
            random_forest_estimators=20,
            catboost_iterations=20,
        ),
    )
    results = pipeline.run(
        feature_modes=["raw_only", "hybrid"],
        include_rule_score_experiment=True,
    )

    assert "rule_engine" in results
    assert "raw_only" in results["experiments"]
    assert "hybrid" in results["experiments"]
    assert "hybrid_rule_scores" in results["experiments"]
    assert (dataset_dir / "train_dataset_strict.csv").exists()
    assert (dataset_dir / "train_dataset_default.csv").exists()
    assert (dataset_dir / "dataset_summary.json").exists()
    assert (dataset_dir / "dataset_diagnostics.json").exists()
    assert (model_dir / "default_comparison.json").exists()
    assert (model_dir / "default" / "raw_only" / "metrics.json").exists()
    assert (model_dir / "default" / "hybrid" / "metrics.json").exists()

    default_train = pd.read_csv(dataset_dir / "train_dataset_default.csv")
    default_test = pd.read_csv(dataset_dir / "default_test.csv")
    assert set(default_train["quality_label"].unique()).issubset({"GOOD", "AMBIGUOUS"})
    assert "AMBIGUOUS" in set(default_train["quality_label"].unique())
    assert "AMBIGUOUS" in set(default_test["quality_label"].unique())

    summary = json.loads((dataset_dir / "dataset_summary.json").read_text())
    diagnostics = json.loads((dataset_dir / "dataset_diagnostics.json").read_text())
    assert int(summary["global_counts"]["ambiguous_cases"]) > 0
    assert int(summary["global_counts"]["bad_cases"]) > 0
    assert float(summary["hm_ambiguous_share"]) < 0.50
    assert float(summary["hm_bad_share"]) < 0.50
    assert int(diagnostics["non_malignancy_ambiguous_class_count"]) >= 3
    assert float(results["rule_engine"]["accuracy"]) < 1.0
