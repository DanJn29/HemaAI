from decimal import Decimal

from app.models.deviation_state import DeviationState
from app.models.reference_range import ReferenceRange
from app.services.deviation_interpreter import DeviationInterpreter


def build_state_map() -> dict[str, DeviationState]:
    return {
        "severe_low": DeviationState(id=1, code="severe_low", name="Severe Low", severity_rank=1),
        "moderate_low": DeviationState(id=2, code="moderate_low", name="Moderate Low", severity_rank=2),
        "mild_low": DeviationState(id=3, code="mild_low", name="Mild Low", severity_rank=3),
        "normal": DeviationState(id=4, code="normal", name="Normal", severity_rank=4),
        "mild_high": DeviationState(id=5, code="mild_high", name="Mild High", severity_rank=5),
        "moderate_high": DeviationState(id=6, code="moderate_high", name="Moderate High", severity_rank=6),
        "severe_high": DeviationState(id=7, code="severe_high", name="Severe High", severity_rank=7),
    }


def build_reference_range(
    *,
    age_min: int = 18,
    age_max: int = 40,
    normal_min: str = "10",
    normal_max: str = "20",
    moderate_low_threshold: str = "8",
    severe_low_threshold: str = "5",
    moderate_high_threshold: str = "24",
    severe_high_threshold: str = "30",
) -> ReferenceRange:
    return ReferenceRange(
        id=1,
        indicator_id=1,
        sex="male",
        age_min=age_min,
        age_max=age_max,
        normal_min=Decimal(normal_min),
        normal_max=Decimal(normal_max),
        mild_low_threshold=Decimal(normal_min),
        moderate_low_threshold=Decimal(moderate_low_threshold),
        severe_low_threshold=Decimal(severe_low_threshold),
        mild_high_threshold=Decimal(normal_max),
        moderate_high_threshold=Decimal(moderate_high_threshold),
        severe_high_threshold=Decimal(severe_high_threshold),
    )


def test_interpret_returns_normal_for_inclusive_boundaries() -> None:
    interpreter = DeviationInterpreter(build_state_map())
    reference_range = build_reference_range()

    low_boundary_state, _ = interpreter.interpret(reference_range, Decimal("10"))
    high_boundary_state, _ = interpreter.interpret(reference_range, Decimal("20"))

    assert low_boundary_state.code == "normal"
    assert high_boundary_state.code == "normal"


def test_interpret_classifies_low_severity_bands() -> None:
    interpreter = DeviationInterpreter(build_state_map())
    reference_range = build_reference_range()

    mild_state, _ = interpreter.interpret(reference_range, Decimal("9"))
    moderate_state, _ = interpreter.interpret(reference_range, Decimal("8"))
    severe_state, normalized_score = interpreter.interpret(reference_range, Decimal("5"))

    assert mild_state.code == "mild_low"
    assert moderate_state.code == "moderate_low"
    assert severe_state.code == "severe_low"
    assert normalized_score < 0


def test_interpret_classifies_high_severity_bands() -> None:
    interpreter = DeviationInterpreter(build_state_map())
    reference_range = build_reference_range()

    mild_state, _ = interpreter.interpret(reference_range, Decimal("21"))
    moderate_state, _ = interpreter.interpret(reference_range, Decimal("24"))
    severe_state, normalized_score = interpreter.interpret(reference_range, Decimal("30"))

    assert mild_state.code == "mild_high"
    assert moderate_state.code == "moderate_high"
    assert severe_state.code == "severe_high"
    assert normalized_score > 0


def test_interpret_respects_age_bucket_specific_boundaries() -> None:
    interpreter = DeviationInterpreter(build_state_map())
    young_adult_range = build_reference_range(age_min=18, age_max=40, normal_max="96", moderate_high_threshold="100", severe_high_threshold="105")
    elderly_range = build_reference_range(age_min=66, age_max=120, normal_max="98", moderate_high_threshold="102", severe_high_threshold="107")

    young_state, young_score = interpreter.interpret(young_adult_range, Decimal("97"))
    elderly_state, elderly_score = interpreter.interpret(elderly_range, Decimal("97"))

    assert young_state.code == "mild_high"
    assert elderly_state.code == "normal"
    assert young_score > 0
    assert elderly_score == 0.0
