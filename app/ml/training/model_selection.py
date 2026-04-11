from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from numbers import Real
from pathlib import Path
from typing import Any


DEFAULT_DATASET_VARIANTS = ("strict", "default")
DATASET_VARIANT_PRIORITY = ("default", "strict")
SUPPORTED_FEATURE_MODES = {"raw_only", "hybrid"}
REQUIRED_METRIC_FIELDS = ("f1_macro", "accuracy")
METRIC_TOLERANCE = 1e-12
ARTIFACT_FORMAT_VERSION = 1
SELECTION_RULE = (
    "prefer valid default deployable candidates first, otherwise fall back to valid "
    "strict candidates; within the chosen dataset variant rank by highest validation "
    "f1_macro, then validation accuracy, then test f1_macro, then test accuracy, "
    "then prefer hybrid over raw_only on ties, then lexical model-name tie-break"
)


class ModelSelectionError(RuntimeError):
    """Raised when no valid deployable model candidate can be selected."""


@dataclass(slots=True)
class ModelCandidate:
    dataset_variant: str
    feature_mode: str
    model_name: str
    include_rule_scores: bool
    model_path: Path
    comparison_path: Path
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]


@dataclass(slots=True)
class CandidateLoadResult:
    candidates: list[ModelCandidate]
    rejection_reasons: list[str]


def write_best_model_metadata(
    *,
    output_dir: str | Path,
    dataset_variants: tuple[str, ...] = DEFAULT_DATASET_VARIANTS,
    tolerance: float = METRIC_TOLERANCE,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    candidate = select_best_model(
        output_dir=output_path,
        dataset_variants=dataset_variants,
        tolerance=tolerance,
    )
    if not candidate.model_path.exists():
        raise FileNotFoundError(
            f"Selected model artifact was not found at {candidate.model_path}."
        )
    metadata = {
        "selection_rule": SELECTION_RULE,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_name": candidate.model_name,
        "dataset_variant": candidate.dataset_variant,
        "feature_mode": candidate.feature_mode,
        "include_rule_scores": candidate.include_rule_scores,
        "model_path": str(candidate.model_path.relative_to(output_path)),
        "comparison_path": str(candidate.comparison_path.relative_to(output_path)),
        "validation_metrics": candidate.validation_metrics,
        "test_metrics": candidate.test_metrics,
    }
    (output_path / "best_model.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def select_best_model(
    *,
    output_dir: str | Path,
    dataset_variants: tuple[str, ...] = DEFAULT_DATASET_VARIANTS,
    tolerance: float = METRIC_TOLERANCE,
) -> ModelCandidate:
    output_path = Path(output_dir)
    candidates_by_variant: dict[str, list[ModelCandidate]] = {}
    rejection_reasons_by_variant: dict[str, list[str]] = {}
    for dataset_variant in dataset_variants:
        load_result = _load_model_candidates_for_variant(
            output_path=output_path,
            dataset_variant=dataset_variant,
        )
        candidates_by_variant[dataset_variant] = load_result.candidates
        rejection_reasons_by_variant[dataset_variant] = load_result.rejection_reasons

    for dataset_variant in _ordered_dataset_variants(dataset_variants):
        variant_candidates = candidates_by_variant.get(dataset_variant, [])
        if not variant_candidates:
            continue
        best = variant_candidates[0]
        for candidate in variant_candidates[1:]:
            if _candidate_is_better(candidate, best, tolerance=tolerance):
                best = candidate
        return best

    raise ModelSelectionError(
        _build_selection_error_message(
            output_path=output_path,
            dataset_variants=dataset_variants,
            rejection_reasons_by_variant=rejection_reasons_by_variant,
        )
    )


def load_model_candidates(
    *,
    output_dir: str | Path,
    dataset_variants: tuple[str, ...] = DEFAULT_DATASET_VARIANTS,
) -> list[ModelCandidate]:
    output_path = Path(output_dir)
    candidates: list[ModelCandidate] = []
    for dataset_variant in dataset_variants:
        candidates.extend(
            _load_model_candidates_for_variant(
                output_path=output_path,
                dataset_variant=dataset_variant,
            ).candidates
        )
    return candidates


def _candidate_is_better(
    candidate: ModelCandidate,
    incumbent: ModelCandidate,
    *,
    tolerance: float,
) -> bool:
    for metric_scope, metric_name in (
        ("validation_metrics", "f1_macro"),
        ("validation_metrics", "accuracy"),
        ("test_metrics", "f1_macro"),
        ("test_metrics", "accuracy"),
    ):
        outcome = _compare_metric(
            getattr(candidate, metric_scope).get(metric_name, 0.0),
            getattr(incumbent, metric_scope).get(metric_name, 0.0),
            tolerance=tolerance,
        )
        if outcome != 0:
            return outcome > 0

    if _metrics_truly_identical(candidate, incumbent, tolerance=tolerance):
        if candidate.feature_mode != incumbent.feature_mode:
            if candidate.feature_mode == "hybrid":
                return True
            if incumbent.feature_mode == "hybrid":
                return False

    if candidate.model_name != incumbent.model_name:
        return candidate.model_name < incumbent.model_name
    return False


def _metrics_truly_identical(
    candidate: ModelCandidate,
    incumbent: ModelCandidate,
    *,
    tolerance: float,
) -> bool:
    return all(
        abs(lhs - rhs) <= tolerance
        for lhs, rhs in (
            (
                candidate.validation_metrics.get("f1_macro", 0.0),
                incumbent.validation_metrics.get("f1_macro", 0.0),
            ),
            (
                candidate.validation_metrics.get("accuracy", 0.0),
                incumbent.validation_metrics.get("accuracy", 0.0),
            ),
            (
                candidate.test_metrics.get("f1_macro", 0.0),
                incumbent.test_metrics.get("f1_macro", 0.0),
            ),
            (
                candidate.test_metrics.get("accuracy", 0.0),
                incumbent.test_metrics.get("accuracy", 0.0),
            ),
        )
    )


def _compare_metric(candidate_value: float, incumbent_value: float, *, tolerance: float) -> int:
    if abs(candidate_value - incumbent_value) <= tolerance:
        return 0
    return 1 if candidate_value > incumbent_value else -1


def _ordered_dataset_variants(dataset_variants: tuple[str, ...]) -> tuple[str, ...]:
    prioritized = [variant for variant in DATASET_VARIANT_PRIORITY if variant in dataset_variants]
    prioritized.extend(variant for variant in dataset_variants if variant not in prioritized)
    return tuple(prioritized)


def _load_model_candidates_for_variant(
    *,
    output_path: Path,
    dataset_variant: str,
) -> CandidateLoadResult:
    comparison_path = output_path / f"{dataset_variant}_comparison.json"
    if not comparison_path.exists():
        return CandidateLoadResult(
            candidates=[],
            rejection_reasons=[
                f"{dataset_variant}: missing comparison artifact at {comparison_path}"
            ],
        )

    try:
        comparison_payload = comparison_path.read_text(encoding="utf-8")
    except OSError as exc:
        return CandidateLoadResult(
            candidates=[],
            rejection_reasons=[
                f"{dataset_variant}: comparison artifact {comparison_path.name} is unreadable ({exc})"
            ],
        )

    try:
        comparison = json.loads(comparison_payload)
    except JSONDecodeError as exc:
        return CandidateLoadResult(
            candidates=[],
            rejection_reasons=[
                f"{dataset_variant}: comparison artifact {comparison_path.name} is malformed JSON ({exc})"
            ],
        )

    if not isinstance(comparison, dict):
        return CandidateLoadResult(
            candidates=[],
            rejection_reasons=[
                f"{dataset_variant}: comparison artifact {comparison_path.name} does not contain a JSON object"
            ],
        )

    experiments = comparison.get("experiments")
    if not isinstance(experiments, dict):
        return CandidateLoadResult(
            candidates=[],
            rejection_reasons=[
                f"{dataset_variant}: comparison artifact {comparison_path.name} is missing a readable 'experiments' object"
            ],
        )

    candidates: list[ModelCandidate] = []
    rejection_reasons: list[str] = []
    saw_supported_feature_mode = False
    saw_deployable_experiment = False
    saw_non_deployable_rule_scores = False

    for feature_mode, experiment in experiments.items():
        if feature_mode not in SUPPORTED_FEATURE_MODES:
            continue
        saw_supported_feature_mode = True
        if not isinstance(experiment, dict):
            rejection_reasons.append(
                f"{dataset_variant}: experiment '{feature_mode}' in {comparison_path.name} is malformed"
            )
            continue
        include_rule_scores = bool(experiment.get("include_rule_scores", False))
        if include_rule_scores:
            saw_non_deployable_rule_scores = True
            rejection_reasons.append(
                f"{dataset_variant}: experiment '{feature_mode}' in {comparison_path.name} is non-deployable because include_rule_scores=true"
            )
            continue
        saw_deployable_experiment = True
        models = experiment.get("models")
        if not isinstance(models, dict) or not models:
            rejection_reasons.append(
                f"{dataset_variant}: experiment '{feature_mode}' in {comparison_path.name} has no readable model entries"
            )
            continue
        for model_name, model_metrics in models.items():
            candidate, rejection_reason = _build_candidate(
                output_path=output_path,
                dataset_variant=dataset_variant,
                feature_mode=feature_mode,
                model_name=model_name,
                model_metrics=model_metrics,
                include_rule_scores=include_rule_scores,
                comparison_path=comparison_path,
            )
            if candidate is not None:
                candidates.append(candidate)
            elif rejection_reason is not None:
                rejection_reasons.append(rejection_reason)

    if not candidates:
        if not saw_supported_feature_mode:
            rejection_reasons.append(
                f"{dataset_variant}: comparison artifact {comparison_path.name} contains no supported deployable feature modes"
            )
        elif not saw_deployable_experiment and saw_non_deployable_rule_scores:
            rejection_reasons.append(
                f"{dataset_variant}: only non-deployable include_rule_scores=true experiments were found in {comparison_path.name}"
            )
        elif not saw_deployable_experiment:
            rejection_reasons.append(
                f"{dataset_variant}: comparison artifact {comparison_path.name} contains no deployable experiments"
            )

    return CandidateLoadResult(
        candidates=candidates,
        rejection_reasons=rejection_reasons,
    )


def _build_candidate(
    *,
    output_path: Path,
    dataset_variant: str,
    feature_mode: str,
    model_name: str,
    model_metrics: Any,
    include_rule_scores: bool,
    comparison_path: Path,
) -> tuple[ModelCandidate | None, str | None]:
    candidate_label = (
        f"{dataset_variant}/{feature_mode}/{model_name} in {comparison_path.name}"
    )
    if not isinstance(model_metrics, dict):
        return None, f"{candidate_label}: model metrics entry is malformed"

    validation_metrics = model_metrics.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        return None, f"{candidate_label}: missing readable validation_metrics block"

    test_metrics = _extract_test_metrics(model_metrics)

    missing_validation = _missing_required_metrics(validation_metrics)
    if missing_validation:
        return (
            None,
            f"{candidate_label}: missing required validation metrics: {', '.join(missing_validation)}",
        )

    missing_test = _missing_required_metrics(test_metrics)
    if missing_test:
        return (
            None,
            f"{candidate_label}: missing required test metrics: {', '.join(missing_test)}",
        )

    model_path = output_path / dataset_variant / feature_mode / f"{model_name}.joblib"
    if not model_path.exists():
        return (
            None,
            f"{candidate_label}: missing model artifact at {model_path}",
        )

    return (
        ModelCandidate(
            dataset_variant=dataset_variant,
            feature_mode=feature_mode,
            model_name=model_name,
            include_rule_scores=include_rule_scores,
            model_path=model_path,
            comparison_path=comparison_path,
            validation_metrics=dict(validation_metrics),
            test_metrics=test_metrics,
        ),
        None,
    )


def _extract_test_metrics(model_metrics: dict[str, Any]) -> dict[str, Any]:
    explicit_test_metrics = model_metrics.get("test_metrics")
    if isinstance(explicit_test_metrics, dict):
        return dict(explicit_test_metrics)
    return {
        key: value
        for key, value in model_metrics.items()
        if key != "validation_metrics"
    }


def _missing_required_metrics(metrics: dict[str, Any]) -> list[str]:
    return [
        metric_name
        for metric_name in REQUIRED_METRIC_FIELDS
        if not _is_numeric_metric(metrics.get(metric_name))
    ]


def _is_numeric_metric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _build_selection_error_message(
    *,
    output_path: Path,
    dataset_variants: tuple[str, ...],
    rejection_reasons_by_variant: dict[str, list[str]],
) -> str:
    details: list[str] = []
    for dataset_variant in _ordered_dataset_variants(dataset_variants):
        reasons = rejection_reasons_by_variant.get(dataset_variant, [])
        if not reasons:
            reasons = [f"{dataset_variant}: no valid deployable candidates were found"]
        details.extend(reasons)
    joined_reasons = " | ".join(details)
    return (
        f"No valid deployable model candidates were found in {output_path}. "
        f"Selection considered variants in order {', '.join(_ordered_dataset_variants(dataset_variants))}. "
        f"Reasons: {joined_reasons}"
    )
