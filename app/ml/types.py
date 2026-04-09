from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(slots=True)
class SyntheticCase:
    case_id: str
    intended_label: str
    sex: str
    age: int
    age_bucket: str
    signal_strength: str
    archetype: str
    overlap_source: str | None
    raw_values: dict[str, Decimal]
    intended_deviation_states: dict[str, str]
    borrowed_indicators: tuple[str, ...] = ()


@dataclass(slots=True)
class SyntheticCaseEvaluation:
    case: SyntheticCase
    actual_deviation_states: dict[str, str]
    normalized_scores: dict[str, float]
    pattern_flags: dict[str, bool]
    rule_scores: dict[str, float]
    top1_label: str
    top3_labels: list[str]
    quality_label: str


@dataclass(slots=True)
class DatasetBundle:
    all_cases: list[dict[str, object]]
    good_cases: list[dict[str, object]]
    ambiguous_cases: list[dict[str, object]]
    bad_cases: list[dict[str, object]]
    train_dataset_strict: list[dict[str, object]]
    train_dataset_default: list[dict[str, object]]
    splits: dict[str, dict[str, list[dict[str, object]]]] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
