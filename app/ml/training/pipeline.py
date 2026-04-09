from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

from app.ml.training.features import FeatureSet, build_feature_set, build_preprocessor


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    logistic_max_iter: int = 2000
    random_forest_estimators: int = 200
    catboost_iterations: int = 200
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.1
    shap_sample_size: int = 200


class TrainingPipeline:
    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        output_dir: str | Path,
        dataset_variant: str,
        config: TrainingConfig | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.dataset_variant = dataset_variant
        self.config = config or TrainingConfig()

    def run(
        self,
        *,
        feature_modes: list[str],
        include_rule_score_experiment: bool = False,
        with_shap: bool = False,
    ) -> dict[str, Any]:
        split_frames = self._load_split_frames(self.dataset_variant)
        baseline_metrics = self._evaluate_rule_engine(split_frames["test"])
        results: dict[str, Any] = {
            "dataset_variant": self.dataset_variant,
            "rule_engine": baseline_metrics,
            "experiments": {},
        }

        for feature_mode in feature_modes:
            experiment_key = feature_mode
            results["experiments"][experiment_key] = self._run_experiment(
                split_frames=split_frames,
                feature_mode=feature_mode,
                include_rule_scores=False,
                with_shap=with_shap,
                output_subdir=self.output_dir / self.dataset_variant / experiment_key,
            )

        if include_rule_score_experiment:
            experiment_key = "hybrid_rule_scores"
            results["experiments"][experiment_key] = self._run_experiment(
                split_frames=split_frames,
                feature_mode="hybrid",
                include_rule_scores=True,
                with_shap=with_shap,
                output_subdir=self.output_dir / self.dataset_variant / experiment_key,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_dir / f"{self.dataset_variant}_comparison.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    def _run_experiment(
        self,
        *,
        split_frames: dict[str, pd.DataFrame],
        feature_mode: str,
        include_rule_scores: bool,
        with_shap: bool,
        output_subdir: Path,
    ) -> dict[str, Any]:
        output_subdir.mkdir(parents=True, exist_ok=True)
        feature_sets = {
            split_name: build_feature_set(
                frame,
                feature_mode=feature_mode,
                include_rule_scores=include_rule_scores,
            )
            for split_name, frame in split_frames.items()
        }

        preprocessor = build_preprocessor(feature_sets["train"])
        X_train = preprocessor.fit_transform(feature_sets["train"].frame)
        X_validation = preprocessor.transform(feature_sets["validation"].frame)
        X_test = preprocessor.transform(feature_sets["test"].frame)
        feature_names = list(preprocessor.get_feature_names_out())

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(feature_sets["train"].target)
        y_validation = label_encoder.transform(feature_sets["validation"].target)
        y_test = label_encoder.transform(feature_sets["test"].target)

        models = {
            "logistic_regression": LogisticRegression(
                max_iter=self.config.logistic_max_iter,
                random_state=self.config.seed,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=self.config.random_forest_estimators,
                random_state=self.config.seed,
                n_jobs=-1,
            ),
            "catboost": CatBoostClassifier(
                iterations=self.config.catboost_iterations,
                depth=self.config.catboost_depth,
                learning_rate=self.config.catboost_learning_rate,
                loss_function="MultiClass",
                random_seed=self.config.seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        }

        experiment_metrics: dict[str, Any] = {}
        predictions_output: dict[str, pd.DataFrame] = {}
        for model_name, model in models.items():
            if model_name == "catboost":
                model.fit(X_train, y_train, eval_set=(X_validation, y_validation), verbose=False)
            else:
                model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_test)
            predicted_labels = self._decode_predictions(
                np.argmax(probabilities, axis=1),
                label_encoder,
            )
            metrics = self._evaluate_predictions(
                y_true=y_test,
                y_pred=np.argmax(probabilities, axis=1),
                probabilities=probabilities,
                label_encoder=label_encoder,
            )
            experiment_metrics[model_name] = metrics

            artifact_path = output_subdir / f"{model_name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "preprocessor": preprocessor,
                    "label_encoder": label_encoder,
                    "feature_names": feature_names,
                    "feature_mode": feature_mode,
                    "include_rule_scores": include_rule_scores,
                },
                artifact_path,
            )

            predictions_output[model_name] = pd.DataFrame(
                {
                    "case_id": split_frames["test"]["case_id"],
                    "intended_label": feature_sets["test"].target,
                    "predicted_label": predicted_labels,
                    "top3_labels": [
                        json.dumps(labels)
                        for labels in self._topk_labels(probabilities, label_encoder, k=3)
                    ],
                }
            )
            predictions_output[model_name].to_csv(
                output_subdir / f"{model_name}_predictions.csv",
                index=False,
            )

            self._export_feature_importance(
                output_subdir=output_subdir,
                model_name=model_name,
                model=model,
                feature_names=feature_names,
                label_encoder=label_encoder,
            )
            self._export_metrics_artifacts(
                output_subdir=output_subdir,
                model_name=model_name,
                metrics=metrics,
            )

            if with_shap and model_name == "catboost":
                self._maybe_export_shap(
                    output_subdir=output_subdir,
                    model=model,
                    features=X_validation,
                    feature_names=feature_names,
                )

        experiment_summary = {
            "feature_mode": feature_mode,
            "include_rule_scores": include_rule_scores,
            "models": experiment_metrics,
        }
        (output_subdir / "metrics.json").write_text(
            json.dumps(experiment_summary, indent=2),
            encoding="utf-8",
        )
        return experiment_summary

    def _load_split_frames(self, dataset_variant: str) -> dict[str, pd.DataFrame]:
        return {
            split_name: pd.read_csv(self.dataset_dir / f"{dataset_variant}_{split_name}.csv")
            for split_name in ("train", "validation", "test")
        }

    def _evaluate_rule_engine(self, test_frame: pd.DataFrame) -> dict[str, Any]:
        labels = sorted(test_frame["intended_label"].unique())
        y_true = test_frame["intended_label"].tolist()
        y_pred = test_frame["rule_top1_label"].tolist()
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        top3_accuracy = float(
            np.mean(
                [
                    row["intended_label"] in json.loads(row["rule_top3_labels"])
                    for _, row in test_frame.iterrows()
                ]
            )
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "top3_accuracy": top3_accuracy,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "per_class": {
                label: report[label]
                for label in labels
            },
        }

    def _evaluate_predictions(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probabilities: np.ndarray,
        label_encoder: LabelEncoder,
    ) -> dict[str, Any]:
        labels = list(range(len(label_encoder.classes_)))
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "top3_accuracy": float(
                top_k_accuracy_score(
                    y_true,
                    probabilities,
                    k=min(3, probabilities.shape[1]),
                    labels=labels,
                )
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "per_class": {
                label_encoder.inverse_transform([label])[0]: report[str(label)]
                for label in labels
            },
        }

    @staticmethod
    def _decode_predictions(
        encoded_predictions: np.ndarray,
        label_encoder: LabelEncoder,
    ) -> list[str]:
        return list(label_encoder.inverse_transform(encoded_predictions))

    @staticmethod
    def _topk_labels(
        probabilities: np.ndarray,
        label_encoder: LabelEncoder,
        *,
        k: int,
    ) -> list[list[str]]:
        topk_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]
        return [
            list(label_encoder.inverse_transform(indices))
            for indices in topk_indices
        ]

    def _export_feature_importance(
        self,
        *,
        output_subdir: Path,
        model_name: str,
        model: Any,
        feature_names: list[str],
        label_encoder: LabelEncoder,
    ) -> None:
        if model_name == "logistic_regression":
            rows: list[dict[str, object]] = []
            for class_index, class_name in enumerate(label_encoder.classes_):
                for feature_name, coefficient in zip(feature_names, model.coef_[class_index], strict=True):
                    rows.append(
                        {
                            "class_name": class_name,
                            "feature": feature_name,
                            "coefficient": float(coefficient),
                            "abs_coefficient": float(abs(coefficient)),
                        }
                    )
            pd.DataFrame(rows).sort_values(
                ["class_name", "abs_coefficient"],
                ascending=[True, False],
            ).to_csv(output_subdir / f"{model_name}_feature_importance.csv", index=False)
            return

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": [float(value) for value in importances],
            }
        ).sort_values("importance", ascending=False).to_csv(
            output_subdir / f"{model_name}_feature_importance.csv",
            index=False,
        )

    def _export_metrics_artifacts(
        self,
        *,
        output_subdir: Path,
        model_name: str,
        metrics: dict[str, Any],
    ) -> None:
        pd.DataFrame(metrics["per_class"]).T.reset_index(names="class_name").to_csv(
            output_subdir / f"{model_name}_per_class_metrics.csv",
            index=False,
        )
        pd.DataFrame(metrics["confusion_matrix"]).to_csv(
            output_subdir / f"{model_name}_confusion_matrix.csv",
            index=False,
        )

    def _maybe_export_shap(
        self,
        *,
        output_subdir: Path,
        model: CatBoostClassifier,
        features: np.ndarray,
        feature_names: list[str],
    ) -> None:
        try:
            import shap
        except ImportError:
            (output_subdir / "shap_status.json").write_text(
                json.dumps({"status": "skipped", "reason": "shap not installed"}, indent=2),
                encoding="utf-8",
            )
            return

        sample_size = min(len(features), self.config.shap_sample_size)
        if sample_size == 0:
            return
        sample = features[:sample_size]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            values = np.mean([np.abs(class_values) for class_values in shap_values], axis=0)
        else:
            values = np.abs(shap_values)
        mean_abs = values.mean(axis=0)
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False).to_csv(
            output_subdir / "catboost_shap_importance.csv",
            index=False,
        )
