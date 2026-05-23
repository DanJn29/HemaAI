from dataclasses import dataclass, field
from decimal import Decimal

from app.models.deviation_state import DeviationState
from app.models.disease import Disease
from app.models.enums import ExplanationSourceType
from app.models.indicator import Indicator


@dataclass(slots=True)
class InterpretedValue:
    indicator: Indicator
    raw_value: Decimal
    deviation_state: DeviationState
    normalized_score: float


@dataclass(slots=True)
class ScoreExplanation:
    source_type: ExplanationSourceType
    source_id: int | None
    explanation_text: str
    score_effect: float


@dataclass(slots=True)
class DiseaseScoreCard:
    disease: Disease
    total_score: float = 0.0
    explanations: list[ScoreExplanation] = field(default_factory=list)
    matched_pattern_bonus: float = 0.0

    def add_explanation(self, explanation: ScoreExplanation) -> None:
        self.total_score += explanation.score_effect
        self.explanations.append(explanation)
        if explanation.source_type == ExplanationSourceType.PATTERN_RULE and explanation.score_effect > 0:
            self.matched_pattern_bonus = max(self.matched_pattern_bonus, explanation.score_effect)
