from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
import json

from sqlalchemy.orm import Session

from app.models.reference_range import ReferenceRange
from app.repositories.catalog_repository import CatalogRepository
from app.services.analysis_orchestrator import AnalysisOrchestrator
from app.services.deviation_interpreter import DeviationInterpreter
from app.services.dto import DiseaseScoreCard, InterpretedValue
from app.services.pattern_matching_service import PatternMatchingService
from app.services.reference_range_service import ReferenceRangeService
from app.services.rule_scoring_service import RuleScoringService
from app.ml.types import SyntheticCase, SyntheticCaseEvaluation


@dataclass(slots=True)
class RuleEngineContext:
    catalog_repository: CatalogRepository
    reference_range_service: ReferenceRangeService
    rule_scoring_service: RuleScoringService
    pattern_matching_service: PatternMatchingService
    analysis_orchestrator: AnalysisOrchestrator
    indicators_by_code: dict[str, object]
    diseases_by_code: dict[str, object]
    deviation_state_map: dict[str, object]
    pattern_rules: list[object]
    interpreter: DeviationInterpreter
    reference_cache: dict[tuple[int, str, int], ReferenceRange]

    @classmethod
    def from_session(cls, session: Session) -> "RuleEngineContext":
        catalog_repository = CatalogRepository(session)
        reference_range_service = ReferenceRangeService(session)
        rule_scoring_service = RuleScoringService(session)
        pattern_matching_service = PatternMatchingService(session)
        analysis_orchestrator = AnalysisOrchestrator(session)

        indicators_by_code = {
            indicator.code: indicator for indicator in catalog_repository.list_indicators()
        }
        diseases_by_code = {
            disease.code: disease for disease in catalog_repository.list_active_diseases()
        }
        deviation_state_map = catalog_repository.get_deviation_state_map()
        interpreter = DeviationInterpreter(deviation_state_map)
        pattern_rules = pattern_matching_service.load_pattern_rules()

        reference_cache: dict[tuple[int, str, int], ReferenceRange] = {}
        for indicator in indicators_by_code.values():
            for sex in ("male", "female"):
                for age in range(18, 121):
                    reference_cache[(indicator.id, sex, age)] = reference_range_service.get_for_indicator(
                        indicator_id=indicator.id,
                        sex=sex,
                        age=age,
                    )

        return cls(
            catalog_repository=catalog_repository,
            reference_range_service=reference_range_service,
            rule_scoring_service=rule_scoring_service,
            pattern_matching_service=pattern_matching_service,
            analysis_orchestrator=analysis_orchestrator,
            indicators_by_code=indicators_by_code,
            diseases_by_code=diseases_by_code,
            deviation_state_map=deviation_state_map,
            pattern_rules=pattern_rules,
            interpreter=interpreter,
            reference_cache=reference_cache,
        )

    def get_reference_range(
        self,
        *,
        indicator_code: str,
        sex: str,
        age: int,
    ) -> ReferenceRange:
        indicator = self.indicators_by_code[indicator_code]
        return self.reference_cache[(indicator.id, sex, age)]


class RuleEngineEvaluator:
    def __init__(self, session: Session, context: RuleEngineContext | None = None) -> None:
        self.session = session
        self.context = context or RuleEngineContext.from_session(session)

    def evaluate_case(self, case: SyntheticCase) -> SyntheticCaseEvaluation:
        interpreted_values = self._interpret_values(
            sex=case.sex,
            age=case.age,
            raw_values=case.raw_values,
        )
        rules = self.context.rule_scoring_service.load_rules(interpreted_values)
        scorecards = self.context.rule_scoring_service.apply_rules(interpreted_values, rules)
        matched_rules = self.context.pattern_matching_service.get_matched_rules(
            interpreted_values,
            self.context.pattern_rules,
        )
        scorecards = self.context.pattern_matching_service.apply_patterns(
            interpreted_values,
            self.context.pattern_rules,
            scorecards,
        )

        actual_deviation_states = {
            item.indicator.code: item.deviation_state.code
            for item in interpreted_values
        }
        normalized_scores = {
            item.indicator.code: item.normalized_score
            for item in interpreted_values
        }
        matched_codes = {pattern_rule.code for pattern_rule in matched_rules}
        pattern_flags = {
            pattern_rule.code: pattern_rule.code in matched_codes
            for pattern_rule in self.context.pattern_rules
        }

        top_labels = self._top_labels(scorecards)
        top1_label = top_labels[0]
        quality_label = self._quality_label(case.intended_label, top_labels)
        rule_scores = {
            disease_code: round(scorecards.get(disease.id, DiseaseScoreCard(disease=disease)).total_score, 4)
            for disease_code, disease in self.context.diseases_by_code.items()
        }

        return SyntheticCaseEvaluation(
            case=case,
            actual_deviation_states=actual_deviation_states,
            normalized_scores=normalized_scores,
            pattern_flags=pattern_flags,
            rule_scores=rule_scores,
            top1_label=top1_label,
            top3_labels=top_labels[:3],
            quality_label=quality_label,
        )

    def serialise_evaluation(self, evaluation: SyntheticCaseEvaluation) -> dict[str, object]:
        row: dict[str, object] = {
            "case_id": evaluation.case.case_id,
            "intended_label": evaluation.case.intended_label,
            "sex": evaluation.case.sex,
            "age": evaluation.case.age,
            "age_bucket": evaluation.case.age_bucket,
            "signal_strength": evaluation.case.signal_strength,
            "archetype": evaluation.case.archetype,
            "overlap_source": evaluation.case.overlap_source,
            "rule_top1_label": evaluation.top1_label,
            "rule_top3_labels": json.dumps(evaluation.top3_labels),
            "quality_label": evaluation.quality_label,
        }
        for indicator_code, raw_value in evaluation.case.raw_values.items():
            row[indicator_code] = float(raw_value)
            row[f"intended_state_{indicator_code}"] = evaluation.case.intended_deviation_states[indicator_code]
            row[f"deviation_state_{indicator_code}"] = evaluation.actual_deviation_states[indicator_code]
            row[f"normalized_score_{indicator_code}"] = evaluation.normalized_scores[indicator_code]
        for pattern_code, matched in evaluation.pattern_flags.items():
            row[f"pattern_{pattern_code}"] = int(matched)
        for disease_code, score in evaluation.rule_scores.items():
            row[f"rule_score_{disease_code}"] = score
        return row

    def _interpret_values(
        self,
        *,
        sex: str,
        age: int,
        raw_values: dict[str, Decimal],
    ) -> list[InterpretedValue]:
        interpreted_values: list[InterpretedValue] = []
        for indicator_code, raw_value in raw_values.items():
            indicator = self.context.indicators_by_code[indicator_code]
            reference_range = self.context.get_reference_range(
                indicator_code=indicator_code,
                sex=sex,
                age=age,
            )
            deviation_state, normalized_score = self.context.interpreter.interpret(
                reference_range,
                raw_value,
            )
            interpreted_values.append(
                InterpretedValue(
                    indicator=indicator,
                    raw_value=raw_value,
                    deviation_state=deviation_state,
                    normalized_score=normalized_score,
                )
            )
        return interpreted_values

    def _top_labels(
        self,
        scorecards: dict[int, DiseaseScoreCard],
    ) -> list[str]:
        non_normal_scorecards = [
            scorecard
            for scorecard in scorecards.values()
            if scorecard.disease.code != "normal"
        ]
        if self.context.analysis_orchestrator.should_fallback_to_normal(non_normal_scorecards):
            return ["normal"]
        persisted = self.context.analysis_orchestrator.select_persisted_scorecards(
            non_normal_scorecards,
        )
        return [scorecard.disease.code for scorecard in persisted]

    @staticmethod
    def _quality_label(intended_label: str, top_labels: Iterable[str]) -> str:
        ranked = list(top_labels)
        if ranked and intended_label == ranked[0]:
            return "GOOD"
        if intended_label in ranked[:3]:
            return "AMBIGUOUS"
        return "BAD"
