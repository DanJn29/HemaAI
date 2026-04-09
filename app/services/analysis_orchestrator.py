from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import DomainValidationError, NotFoundError
from app.models.analysis_case import AnalysisCase
from app.models.analysis_result import AnalysisResult
from app.models.analysis_result_explanation import AnalysisResultExplanation
from app.models.analysis_value import AnalysisValue
from app.models.enums import AnalysisResultSource
from app.repositories.analysis_repository import AnalysisRepository
from app.repositories.catalog_repository import CatalogRepository
from app.schemas.analysis import (
    AnalysisCreateRequest,
    AnalysisResponse,
    ExplanationResponse,
    HypothesisResponse,
    IndicatorInterpretationResponse,
    PatientSummary,
)
from app.services.deviation_interpreter import DeviationInterpreter
from app.services.dto import DiseaseScoreCard, InterpretedValue
from app.services.pattern_matching_service import PatternMatchingService
from app.services.reference_range_service import ReferenceRangeService
from app.services.rule_scoring_service import RuleScoringService


class AnalysisOrchestrator:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.catalog_repository = CatalogRepository(session)
        self.analysis_repository = AnalysisRepository(session)
        self.reference_range_service = ReferenceRangeService(session)
        self.rule_scoring_service = RuleScoringService(session)
        self.pattern_matching_service = PatternMatchingService(session)

    def create_analysis(self, payload: AnalysisCreateRequest) -> AnalysisResponse:
        with self.session.begin():
            case = AnalysisCase(
                sex=payload.sex,
                age=payload.age,
                patient_code=payload.patient_code,
                notes=payload.notes,
            )
            self.analysis_repository.add_case(case)
            interpreted_values = self._create_or_refresh_values(case, payload)
            scorecards = self._score_case(interpreted_values)
            self._replace_results(case, scorecards)

        return self.get_analysis(case.id)

    def recompute_analysis(self, analysis_case_id: int) -> AnalysisResponse:
        with self.session.begin():
            case = self.analysis_repository.get_case_for_update(analysis_case_id)
            if case is None:
                raise NotFoundError(f"Analysis case {analysis_case_id} was not found.")

            deviation_state_map = self.catalog_repository.get_deviation_state_map()
            interpreter = DeviationInterpreter(deviation_state_map)
            interpreted_values: list[InterpretedValue] = []
            for analysis_value in case.values:
                reference_range = self.reference_range_service.get_for_indicator(
                    indicator_id=analysis_value.indicator_id,
                    sex=case.sex,
                    age=case.age,
                )
                deviation_state, normalized_score = interpreter.interpret(
                    reference_range=reference_range,
                    raw_value=analysis_value.raw_value,
                )
                analysis_value.deviation_state_id = deviation_state.id
                analysis_value.deviation_state = deviation_state
                analysis_value.normalized_score = Decimal(str(normalized_score))
                interpreted_values.append(
                    InterpretedValue(
                        indicator=analysis_value.indicator,
                        raw_value=analysis_value.raw_value,
                        deviation_state=deviation_state,
                        normalized_score=normalized_score,
                    )
                )

            scorecards = self._score_case(interpreted_values)
            self._replace_results(case, scorecards)

        return self.get_analysis(analysis_case_id)

    def get_analysis(self, analysis_case_id: int) -> AnalysisResponse:
        case = self.analysis_repository.get_case(analysis_case_id)
        if case is None:
            raise NotFoundError(f"Analysis case {analysis_case_id} was not found.")
        return self._serialize_case(case)

    def _create_or_refresh_values(
        self,
        case: AnalysisCase,
        payload: AnalysisCreateRequest,
    ) -> list[InterpretedValue]:
        indicators = self.catalog_repository.get_indicators_by_codes(
            [value.indicator_code for value in payload.values]
        )
        indicator_map = {indicator.code: indicator for indicator in indicators}
        missing_codes = sorted(
            {
                value.indicator_code
                for value in payload.values
                if value.indicator_code not in indicator_map
            }
        )
        if missing_codes:
            missing = ", ".join(missing_codes)
            raise DomainValidationError(f"Unknown indicator codes: {missing}")

        deviation_state_map = self.catalog_repository.get_deviation_state_map()
        interpreter = DeviationInterpreter(deviation_state_map)

        interpreted_values: list[InterpretedValue] = []
        for item in payload.values:
            indicator = indicator_map[item.indicator_code]
            reference_range = self.reference_range_service.get_for_indicator(
                indicator_id=indicator.id,
                sex=payload.sex,
                age=payload.age,
            )
            deviation_state, normalized_score = interpreter.interpret(reference_range, item.raw_value)
            analysis_value = AnalysisValue(
                analysis_case_id=case.id,
                indicator_id=indicator.id,
                raw_value=item.raw_value,
                deviation_state_id=deviation_state.id,
                normalized_score=Decimal(str(normalized_score)),
            )
            self.analysis_repository.add_value(analysis_value)
            analysis_value.indicator = indicator
            analysis_value.deviation_state = deviation_state
            interpreted_values.append(
                InterpretedValue(
                    indicator=indicator,
                    raw_value=item.raw_value,
                    deviation_state=deviation_state,
                    normalized_score=normalized_score,
                )
            )
        return interpreted_values

    def _score_case(self, interpreted_values: list[InterpretedValue]) -> dict[int, DiseaseScoreCard]:
        rules = self.rule_scoring_service.load_rules(interpreted_values)
        scorecards = self.rule_scoring_service.apply_rules(interpreted_values, rules)
        pattern_rules = self.pattern_matching_service.load_pattern_rules()
        return self.pattern_matching_service.apply_patterns(interpreted_values, pattern_rules, scorecards)

    def _replace_results(
        self,
        case: AnalysisCase,
        scorecards: dict[int, DiseaseScoreCard],
    ) -> None:
        for existing_result in list(case.results):
            self.session.delete(existing_result)
        self.session.flush()

        non_normal_scorecards = [
            scorecard
            for scorecard in scorecards.values()
            if scorecard.disease.code != "normal"
        ]
        if self.should_fallback_to_normal(non_normal_scorecards):
            normal_disease = self.catalog_repository.get_disease_by_code("normal")
            if normal_disease is None:
                raise DomainValidationError("The normal disease seed record is missing.")
            result = AnalysisResult(
                analysis_case=case,
                disease=normal_disease,
                total_score=Decimal("0"),
                rank_position=1,
                result_source=AnalysisResultSource.RULE_ENGINE.value,
            )
            self.session.add(result)
            self.session.flush()
            case.results = [result]
            return

        persisted_scorecards = self.select_persisted_scorecards(non_normal_scorecards)

        case.results = []
        for rank_position, scorecard in enumerate(persisted_scorecards, start=1):
            result = AnalysisResult(
                analysis_case=case,
                disease=scorecard.disease,
                total_score=Decimal(str(round(scorecard.total_score, 3))),
                rank_position=rank_position,
                result_source=AnalysisResultSource.RULE_ENGINE.value,
            )
            self.session.add(result)
            self.session.flush()
            for explanation in scorecard.explanations:
                result.explanations.append(
                    AnalysisResultExplanation(
                        analysis_result=result,
                        source_type=explanation.source_type,
                        source_id=explanation.source_id,
                        explanation_text=explanation.explanation_text,
                        score_effect=Decimal(str(round(explanation.score_effect, 3))),
                    )
                )

    def should_fallback_to_normal(self, non_normal_scorecards: list[DiseaseScoreCard]) -> bool:
        has_pathology = any(
            scorecard.total_score >= self.settings.min_pathology_score
            for scorecard in non_normal_scorecards
        )
        has_strong_pattern = any(
            scorecard.matched_pattern_bonus >= self.settings.strong_pattern_bonus_threshold
            for scorecard in non_normal_scorecards
        )
        return not has_pathology and not has_strong_pattern

    def select_persisted_scorecards(
        self,
        non_normal_scorecards: list[DiseaseScoreCard],
    ) -> list[DiseaseScoreCard]:
        persisted_scorecards = sorted(
            [
                scorecard
                for scorecard in non_normal_scorecards
                if scorecard.total_score >= self.settings.min_persisted_score
            ],
            key=lambda card: card.total_score,
            reverse=True,
        )[: self.settings.max_persisted_non_normal]
        if not persisted_scorecards and non_normal_scorecards:
            persisted_scorecards = sorted(
                non_normal_scorecards,
                key=lambda card: card.total_score,
                reverse=True,
            )[:1]
        return persisted_scorecards

    def _serialize_case(self, case: AnalysisCase) -> AnalysisResponse:
        top_results = sorted(case.results, key=lambda item: item.rank_position)[
            : self.settings.max_returned_hypotheses
        ]
        return AnalysisResponse(
            analysis_id=case.id,
            created_at=case.created_at,
            patient=PatientSummary(
                sex=case.sex,
                age=case.age,
                patient_code=case.patient_code,
                notes=case.notes,
            ),
            indicator_interpretation=[
                IndicatorInterpretationResponse(
                    indicator_code=value.indicator.code,
                    indicator_name=value.indicator.name,
                    raw_value=value.raw_value,
                    unit=value.indicator.unit,
                    deviation_state=value.deviation_state.code if value.deviation_state else "unknown",
                    normalized_score=float(value.normalized_score) if value.normalized_score is not None else None,
                )
                for value in case.values
            ],
            top_hypotheses=[
                HypothesisResponse(
                    rank=result.rank_position,
                    disease_code=result.disease.code,
                    disease_name=result.disease.name,
                    total_score=float(result.total_score),
                    confidence=float(result.confidence) if result.confidence is not None else None,
                    explanations=[
                        ExplanationResponse(
                            type=explanation.source_type.value,
                            text=explanation.explanation_text,
                            score_effect=float(explanation.score_effect),
                        )
                        for explanation in result.explanations
                    ],
                )
                for result in top_results
            ],
            disclaimer=self.settings.analysis_disclaimer,
        )
