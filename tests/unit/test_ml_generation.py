from collections import Counter

from app.ml.generation.dataset_builder import QualityMixController, SyntheticDatasetBuilder
from app.ml.generation.evaluator import RuleEngineContext, RuleEngineEvaluator
from app.ml.generation.generator import SyntheticCaseGenerator, SyntheticGenerationConfig
from app.ml.generation.profiles import CLASS_PROFILES


def test_generate_value_for_state_respects_requested_deviation_state(db_session) -> None:
    context = RuleEngineContext.from_session(db_session)
    generator = SyntheticCaseGenerator(
        session=db_session,
        config=SyntheticGenerationConfig(seed=123, samples_per_class=8),
        context=context,
    )
    reference_range = context.get_reference_range(
        indicator_code="HGB",
        sex="female",
        age=28,
    )

    for state_code in [
        "normal",
        "mild_low",
        "moderate_low",
        "severe_low",
        "mild_high",
        "moderate_high",
        "severe_high",
    ]:
        value = generator.generate_value_for_state(
            reference_range=reference_range,
            deviation_state=state_code,
            indicator_code="HGB",
        )
        actual_state, _ = context.interpreter.interpret(reference_range, value)
        assert actual_state.code == state_code
        assert value >= 0


def test_generate_value_for_state_applies_indicator_caps(db_session) -> None:
    context = RuleEngineContext.from_session(db_session)
    generator = SyntheticCaseGenerator(
        session=db_session,
        config=SyntheticGenerationConfig(seed=999, samples_per_class=8),
        context=context,
    )
    reference_range = context.get_reference_range(
        indicator_code="WBC",
        sex="male",
        age=52,
    )

    severe_high_value = generator.generate_value_for_state(
        reference_range=reference_range,
        deviation_state="severe_high",
        indicator_code="WBC",
    )
    assert severe_high_value <= 50


def test_rule_engine_evaluator_labels_a_strong_iron_case_good(db_session) -> None:
    context = RuleEngineContext.from_session(db_session)
    generator = SyntheticCaseGenerator(
        session=db_session,
        config=SyntheticGenerationConfig(seed=7, samples_per_class=8),
        context=context,
    )
    evaluator = RuleEngineEvaluator(db_session, context=context)

    case = generator.generate_case(
        "iron_deficiency_anemia",
        1,
        signal_strength="strong",
        archetype="canonical",
        sex="female",
        age=28,
    )
    evaluation = evaluator.evaluate_case(case)

    assert evaluation.top1_label == "iron_deficiency_anemia"
    assert evaluation.quality_label == "GOOD"
    assert evaluation.actual_deviation_states["HGB"].endswith("low")
    assert evaluation.actual_deviation_states["RDW"].endswith("high")


def test_overlap_archetype_uses_only_clinically_nearby_neighbors(db_session) -> None:
    context = RuleEngineContext.from_session(db_session)
    generator = SyntheticCaseGenerator(
        session=db_session,
        config=SyntheticGenerationConfig(seed=24, samples_per_class=8),
        context=context,
    )

    for label in [
        "bacterial_infection",
        "viral_infection",
        "allergic_or_parasitic_pattern",
        "thrombocytopenia_pattern",
        "hematologic_malignancy_suspicion",
    ]:
        profile = CLASS_PROFILES[label]
        case = generator.generate_case(label, 1, archetype="overlap")
        assert case.overlap_source in profile.overlap_neighbors
        if case.overlap_source is not None:
            assert set(case.borrowed_indicators).issubset(
                set(profile.borrowed_indicators_for(case.overlap_source))
            )


def test_bad_protection_controller_reduces_conflicted_weight() -> None:
    config = SyntheticGenerationConfig(seed=42, samples_per_class=8)
    controller = QualityMixController(config)
    base_weight = controller.current_conflicted_weights["viral_infection"]

    for _ in range(config.rolling_bad_window):
        controller.record(label="viral_infection", quality_label="BAD")

    assert controller.current_conflicted_weights["viral_infection"] < base_weight
    assert controller.current_conflicted_weights["viral_infection"] >= min(base_weight, config.min_conflicted_weight)


def test_dataset_builder_creates_balanced_nontrivial_variants(db_session) -> None:
    builder = SyntheticDatasetBuilder(
        session=db_session,
        config=SyntheticGenerationConfig(seed=42, samples_per_class=8),
    )

    bundle = builder.build()
    strict_counts = Counter(row["intended_label"] for row in bundle.train_dataset_strict)
    default_counts = Counter(row["intended_label"] for row in bundle.train_dataset_default)
    default_quality_counts = Counter(row["quality_label"] for row in bundle.train_dataset_default)
    default_test_quality_counts = Counter(
        row["quality_label"] for row in bundle.splits["default"]["test"]
    )

    assert bundle.train_dataset_strict
    assert bundle.train_dataset_default
    assert bundle.ambiguous_cases
    assert bundle.bad_cases
    assert all(row["quality_label"] == "GOOD" for row in bundle.train_dataset_strict)
    assert all(row["quality_label"] in {"GOOD", "AMBIGUOUS"} for row in bundle.train_dataset_default)
    assert default_quality_counts["AMBIGUOUS"] > 0
    assert default_test_quality_counts["AMBIGUOUS"] > 0
    assert set(strict_counts.values()) == {8}
    assert set(default_counts.values()) == {8}
    assert bundle.summary["global_counts"]["bad_cases"] > 0
    assert bundle.summary["global_counts"]["ambiguous_cases"] > 0
    assert "per_class" in bundle.diagnostics


def test_dataset_builder_spreads_ambiguity_beyond_malignancy(db_session) -> None:
    builder = SyntheticDatasetBuilder(
        session=db_session,
        config=SyntheticGenerationConfig(seed=42, samples_per_class=16),
    )

    bundle = builder.build()
    quality_counts = bundle.summary["quality_counts_per_class"]
    non_malignancy_labels = [
        label for label in quality_counts if label != "hematologic_malignancy_suspicion"
    ]
    ambiguous_classes = [
        label for label in non_malignancy_labels if quality_counts[label]["AMBIGUOUS"] > 0
    ]
    bad_classes = [
        label for label in non_malignancy_labels if quality_counts[label]["BAD"] > 0
    ]

    assert len(ambiguous_classes) >= 3
    assert len(bad_classes) >= 1
    assert bundle.summary["hm_ambiguous_share"] < 0.80
    assert bundle.diagnostics["non_malignancy_ambiguous_class_count"] >= 3
