from decimal import Decimal

from app.models.deviation_state import DeviationState
from app.models.disease import Disease
from app.models.enums import DeviationFamily, PatternMatchMode
from app.models.indicator import Indicator
from app.models.pattern_rule import PatternRule
from app.models.pattern_rule_condition import PatternRuleCondition
from app.services.dto import DiseaseScoreCard, InterpretedValue
from app.services.pattern_matching_service import PatternMatchingService


def test_pattern_matching_applies_family_bonus() -> None:
    disease = Disease(id=1, code="iron_deficiency_anemia", name="Iron Deficiency Anemia", category="hematology", description="d", severity_level="moderate", is_active=True)
    hgb = Indicator(id=1, code="HGB", name="Hemoglobin", unit="g/L", description="d")
    mcv = Indicator(id=2, code="MCV", name="MCV", unit="fL", description="d")
    mild_low = DeviationState(id=1, code="mild_low", name="Mild Low", severity_rank=3)
    moderate_low = DeviationState(id=2, code="moderate_low", name="Moderate Low", severity_rank=2)

    interpreted_values = [
        InterpretedValue(indicator=hgb, raw_value=Decimal("118"), deviation_state=mild_low, normalized_score=-0.1),
        InterpretedValue(indicator=mcv, raw_value=Decimal("76"), deviation_state=moderate_low, normalized_score=-0.2),
    ]
    pattern_rule = PatternRule(
        id=1,
        code="iron_pattern",
        name="Iron Pattern",
        disease_id=1,
        disease=disease,
        bonus_weight=Decimal("4.0"),
        rule_description="Pattern matched.",
        conditions=[
            PatternRuleCondition(id=1, pattern_rule_id=1, indicator_id=1, match_mode=PatternMatchMode.FAMILY, deviation_family=DeviationFamily.LOW, deviation_state_id=None),
            PatternRuleCondition(id=2, pattern_rule_id=1, indicator_id=2, match_mode=PatternMatchMode.FAMILY, deviation_family=DeviationFamily.LOW, deviation_state_id=None),
        ],
    )

    service = PatternMatchingService(session=None)
    scorecards = service.apply_patterns(interpreted_values, [pattern_rule], {})

    assert scorecards[disease.id].total_score == 4.0
    assert scorecards[disease.id].matched_pattern_bonus == 4.0


def test_pattern_matching_requires_all_conditions() -> None:
    disease = Disease(id=1, code="viral_infection", name="Viral", category="infection", description="d", severity_level="moderate", is_active=True)
    lym = Indicator(id=1, code="LYM", name="Lymphocytes", unit="10^9/L", description="d")
    normal = DeviationState(id=1, code="normal", name="Normal", severity_rank=4)
    severe_high = DeviationState(id=2, code="severe_high", name="Severe High", severity_rank=7)

    interpreted_values = [
        InterpretedValue(indicator=lym, raw_value=Decimal("7.0"), deviation_state=severe_high, normalized_score=0.5),
    ]
    pattern_rule = PatternRule(
        id=1,
        code="viral_pattern",
        name="Viral Pattern",
        disease_id=1,
        disease=disease,
        bonus_weight=Decimal("3.5"),
        rule_description="Pattern matched.",
        conditions=[
            PatternRuleCondition(id=1, pattern_rule_id=1, indicator_id=1, match_mode=PatternMatchMode.EXACT, deviation_state_id=1, deviation_family=None),
        ],
    )

    service = PatternMatchingService(session=None)
    scorecards = service.apply_patterns(interpreted_values, [pattern_rule], {})

    assert disease.id not in scorecards

