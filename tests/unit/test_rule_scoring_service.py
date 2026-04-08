from decimal import Decimal

from app.models.deviation_state import DeviationState
from app.models.disease import Disease
from app.models.enums import IndicatorRuleRelationType
from app.models.indicator import Indicator
from app.models.indicator_rule import IndicatorRule
from app.services.analysis_orchestrator import AnalysisOrchestrator
from app.services.dto import DiseaseScoreCard, InterpretedValue
from app.services.rule_scoring_service import RuleScoringService


def test_rule_scoring_applies_support_and_contradiction() -> None:
    disease = Disease(id=1, code="iron_deficiency_anemia", name="Iron Deficiency Anemia", category="hematology", description="d", severity_level="moderate", is_active=True)
    indicator = Indicator(id=1, code="HGB", name="Hemoglobin", unit="g/L", description="d")
    mild_low = DeviationState(id=1, code="mild_low", name="Mild Low", severity_rank=3)
    normal = DeviationState(id=2, code="normal", name="Normal", severity_rank=4)

    interpreted_values = [
        InterpretedValue(indicator=indicator, raw_value=Decimal("118"), deviation_state=mild_low, normalized_score=-0.1),
        InterpretedValue(indicator=Indicator(id=2, code="MCV", name="MCV", unit="fL", description="d"), raw_value=Decimal("90"), deviation_state=normal, normalized_score=0.0),
    ]
    rules = [
        IndicatorRule(
            id=1,
            indicator_id=1,
            deviation_state_id=1,
            disease_id=1,
            relation_type=IndicatorRuleRelationType.SUPPORT,
            weight=Decimal("2.5"),
            disease=disease,
            indicator=indicator,
            deviation_state=mild_low,
            evidence_note="Low HGB supported iron deficiency anemia.",
        ),
        IndicatorRule(
            id=2,
            indicator_id=2,
            deviation_state_id=2,
            disease_id=1,
            relation_type=IndicatorRuleRelationType.CONTRADICT,
            weight=Decimal("1.0"),
            disease=disease,
            indicator=interpreted_values[1].indicator,
            deviation_state=normal,
            evidence_note="Normal MCV contradicted iron deficiency anemia.",
        ),
    ]

    service = RuleScoringService(session=None)
    scorecards = service.apply_rules(interpreted_values, rules)

    assert scorecards[disease.id].total_score == 1.5
    assert [item.score_effect for item in scorecards[disease.id].explanations] == [2.5, -1.0]


def test_normal_fallback_threshold_logic() -> None:
    orchestrator = AnalysisOrchestrator(session=None)
    weak_disease = Disease(id=1, code="bacterial_infection", name="Bacterial", category="infection", description="d", severity_level="moderate", is_active=True)
    weak_scorecard = DiseaseScoreCard(disease=weak_disease, total_score=2.9, matched_pattern_bonus=2.5)
    strong_scorecard = DiseaseScoreCard(disease=weak_disease, total_score=1.0, matched_pattern_bonus=3.0)

    assert orchestrator.should_fallback_to_normal([weak_scorecard]) is True
    assert orchestrator.should_fallback_to_normal([strong_scorecard]) is False

