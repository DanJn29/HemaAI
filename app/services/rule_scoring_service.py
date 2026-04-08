from collections.abc import Iterable

from sqlalchemy import select, tuple_
from sqlalchemy.orm import Session, joinedload

from app.models.enums import ExplanationSourceType, IndicatorRuleRelationType
from app.models.indicator_rule import IndicatorRule
from app.services.dto import DiseaseScoreCard, InterpretedValue, ScoreExplanation


class RuleScoringService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def load_rules(self, interpreted_values: Iterable[InterpretedValue]) -> list[IndicatorRule]:
        pairs = [(item.indicator.id, item.deviation_state.id) for item in interpreted_values]
        if not pairs:
            return []
        stmt = (
            select(IndicatorRule)
            .where(tuple_(IndicatorRule.indicator_id, IndicatorRule.deviation_state_id).in_(pairs))
            .options(
                joinedload(IndicatorRule.indicator),
                joinedload(IndicatorRule.deviation_state),
                joinedload(IndicatorRule.disease),
            )
        )
        return list(self.session.scalars(stmt).all())

    def apply_rules(
        self,
        interpreted_values: Iterable[InterpretedValue],
        rules: Iterable[IndicatorRule],
    ) -> dict[int, DiseaseScoreCard]:
        interpreted_lookup = {
            (item.indicator.id, item.deviation_state.id): item for item in interpreted_values
        }
        scorecards: dict[int, DiseaseScoreCard] = {}
        for rule in rules:
            interpreted = interpreted_lookup[(rule.indicator_id, rule.deviation_state_id)]
            scorecard = scorecards.setdefault(rule.disease_id, DiseaseScoreCard(disease=rule.disease))
            effect = float(rule.weight)
            relation_verb = "supported"
            if rule.relation_type == IndicatorRuleRelationType.CONTRADICT:
                effect *= -1
                relation_verb = "contradicted"
            text = rule.evidence_note or (
                f"{interpreted.indicator.code} {interpreted.deviation_state.code} "
                f"{relation_verb} {rule.disease.name}."
            )
            scorecard.add_explanation(
                ScoreExplanation(
                    source_type=ExplanationSourceType.INDICATOR_RULE,
                    source_id=rule.id,
                    explanation_text=text,
                    score_effect=effect,
                )
            )
        return scorecards

