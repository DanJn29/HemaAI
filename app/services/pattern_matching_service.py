from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models.enums import ExplanationSourceType, PatternMatchMode
from app.models.pattern_rule import PatternRule
from app.models.pattern_rule_condition import PatternRuleCondition
from app.services.deviation_interpreter import DeviationInterpreter
from app.services.dto import DiseaseScoreCard, InterpretedValue, ScoreExplanation


class PatternMatchingService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def load_pattern_rules(self) -> list[PatternRule]:
        stmt = (
            select(PatternRule)
            .options(
                joinedload(PatternRule.disease),
                selectinload(PatternRule.conditions).joinedload(PatternRuleCondition.indicator),
                selectinload(PatternRule.conditions).joinedload(PatternRuleCondition.deviation_state),
            )
            .order_by(PatternRule.id)
        )
        return list(self.session.scalars(stmt).all())

    def apply_patterns(
        self,
        interpreted_values: Iterable[InterpretedValue],
        pattern_rules: Iterable[PatternRule],
        scorecards: dict[int, DiseaseScoreCard],
    ) -> dict[int, DiseaseScoreCard]:
        interpreted_by_indicator = {item.indicator.id: item for item in interpreted_values}
        for pattern_rule in pattern_rules:
            if self._matches(pattern_rule, interpreted_by_indicator):
                scorecard = scorecards.setdefault(
                    pattern_rule.disease_id,
                    DiseaseScoreCard(disease=pattern_rule.disease),
                )
                scorecard.add_explanation(
                    ScoreExplanation(
                        source_type=ExplanationSourceType.PATTERN_RULE,
                        source_id=pattern_rule.id,
                        explanation_text=pattern_rule.rule_description,
                        score_effect=float(pattern_rule.bonus_weight),
                    )
                )
        return scorecards

    def _matches(
        self,
        pattern_rule: PatternRule,
        interpreted_by_indicator: dict[int, InterpretedValue],
    ) -> bool:
        for condition in pattern_rule.conditions:
            interpreted = interpreted_by_indicator.get(condition.indicator_id)
            if interpreted is None:
                return False
            if condition.match_mode == PatternMatchMode.EXACT:
                if interpreted.deviation_state.id != condition.deviation_state_id:
                    return False
                continue
            family = DeviationInterpreter.get_family(interpreted.deviation_state.code)
            if family != condition.deviation_family:
                return False
        return True
