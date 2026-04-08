from collections.abc import Iterable
from decimal import Decimal

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_session_factory, initialize_engine
from app.models.deviation_state import DeviationState
from app.models.disease import Disease
from app.models.enums import (
    DeviationFamily,
    IndicatorRuleRelationType,
    PatternMatchMode,
    Sex,
)
from app.models.indicator import Indicator
from app.models.indicator_rule import IndicatorRule
from app.models.pattern_rule import PatternRule
from app.models.pattern_rule_condition import PatternRuleCondition
from app.models.reference_range import ReferenceRange

ADULT_AGE_MIN = 18
ADULT_AGE_MAX = 120

INDICATORS = [
    {"code": "WBC", "name": "White Blood Cell Count", "unit": "10^9/L", "description": "Total white blood cell count."},
    {"code": "RBC", "name": "Red Blood Cell Count", "unit": "10^12/L", "description": "Total red blood cell count."},
    {"code": "HGB", "name": "Hemoglobin", "unit": "g/L", "description": "Hemoglobin concentration."},
    {"code": "HCT", "name": "Hematocrit", "unit": "L/L", "description": "Packed cell volume."},
    {"code": "MCV", "name": "Mean Corpuscular Volume", "unit": "fL", "description": "Average red cell volume."},
    {"code": "MCH", "name": "Mean Corpuscular Hemoglobin", "unit": "pg", "description": "Average hemoglobin mass per red cell."},
    {"code": "MCHC", "name": "Mean Corpuscular Hemoglobin Concentration", "unit": "g/L", "description": "Average hemoglobin concentration in red cells."},
    {"code": "PLT", "name": "Platelet Count", "unit": "10^9/L", "description": "Platelet concentration."},
    {"code": "RDW", "name": "Red Cell Distribution Width", "unit": "%", "description": "Variation in red blood cell size."},
    {"code": "NEU", "name": "Neutrophils", "unit": "10^9/L", "description": "Absolute neutrophil count."},
    {"code": "LYM", "name": "Lymphocytes", "unit": "10^9/L", "description": "Absolute lymphocyte count."},
    {"code": "MONO", "name": "Monocytes", "unit": "10^9/L", "description": "Absolute monocyte count."},
    {"code": "EOS", "name": "Eosinophils", "unit": "10^9/L", "description": "Absolute eosinophil count."},
    {"code": "BASO", "name": "Basophils", "unit": "10^9/L", "description": "Absolute basophil count."},
]

DEVIATION_STATES = [
    {"code": "severe_low", "name": "Severe Low", "severity_rank": 1},
    {"code": "moderate_low", "name": "Moderate Low", "severity_rank": 2},
    {"code": "mild_low", "name": "Mild Low", "severity_rank": 3},
    {"code": "normal", "name": "Normal", "severity_rank": 4},
    {"code": "mild_high", "name": "Mild High", "severity_rank": 5},
    {"code": "moderate_high", "name": "Moderate High", "severity_rank": 6},
    {"code": "severe_high", "name": "Severe High", "severity_rank": 7},
]

DISEASES = [
    {
        "code": "normal",
        "name": "No Significant Pathologic Pattern",
        "category": "baseline",
        "description": "Fallback hypothesis indicating that no sufficiently strong pathological CBC pattern was detected.",
        "severity_level": "none",
        "is_active": True,
    },
    {
        "code": "iron_deficiency_anemia",
        "name": "Iron Deficiency Anemia",
        "category": "hematology",
        "description": "Demo hypothesis for microcytic anemia with iron deficiency features.",
        "severity_level": "moderate",
        "is_active": True,
    },
    {
        "code": "macrocytic_anemia",
        "name": "Macrocytic Anemia",
        "category": "hematology",
        "description": "Demo hypothesis for macrocytic anemia pattern support.",
        "severity_level": "moderate",
        "is_active": True,
    },
    {
        "code": "bacterial_infection",
        "name": "Bacterial Infection Pattern",
        "category": "infection",
        "description": "Demo hypothesis for neutrophil-predominant inflammatory pattern.",
        "severity_level": "moderate",
        "is_active": True,
    },
    {
        "code": "viral_infection",
        "name": "Viral Infection Pattern",
        "category": "infection",
        "description": "Demo hypothesis for lymphocyte-predominant inflammatory pattern.",
        "severity_level": "moderate",
        "is_active": True,
    },
    {
        "code": "allergic_or_parasitic_pattern",
        "name": "Allergic or Parasitic Pattern",
        "category": "immunology",
        "description": "Demo hypothesis for eosinophil-predominant pattern.",
        "severity_level": "low",
        "is_active": True,
    },
    {
        "code": "thrombocytopenia_pattern",
        "name": "Thrombocytopenia Pattern",
        "category": "hematology",
        "description": "Demo hypothesis for low platelet pattern support.",
        "severity_level": "moderate",
        "is_active": True,
    },
    {
        "code": "hematologic_malignancy_suspicion",
        "name": "Hematologic Malignancy Suspicion",
        "category": "hematology",
        "description": "Demo escalation hypothesis for multiple severe CBC abnormalities.",
        "severity_level": "high",
        "is_active": True,
    },
]

REFERENCE_RANGES = {
    Sex.MALE.value: {
        "WBC": (4.0, 10.0, 3.0, 2.0, 12.0, 15.0),
        "RBC": (4.5, 5.9, 4.0, 3.5, 6.2, 6.6),
        "HGB": (135.0, 175.0, 120.0, 100.0, 185.0, 195.0),
        "HCT": (0.41, 0.53, 0.36, 0.30, 0.55, 0.58),
        "MCV": (80.0, 96.0, 75.0, 70.0, 100.0, 105.0),
        "MCH": (27.0, 33.0, 24.0, 21.0, 34.0, 36.0),
        "MCHC": (320.0, 360.0, 300.0, 280.0, 370.0, 380.0),
        "PLT": (150.0, 400.0, 100.0, 50.0, 450.0, 600.0),
        "RDW": (11.5, 14.5, 10.5, 9.5, 15.5, 17.0),
        "NEU": (2.0, 7.5, 1.5, 1.0, 8.5, 12.0),
        "LYM": (1.0, 4.0, 0.8, 0.5, 4.5, 6.0),
        "MONO": (0.2, 0.8, 0.1, 0.05, 1.0, 1.5),
        "EOS": (0.0, 0.5, 0.0, 0.0, 0.7, 1.0),
        "BASO": (0.0, 0.2, 0.0, 0.0, 0.3, 0.5),
    },
    Sex.FEMALE.value: {
        "WBC": (4.0, 10.0, 3.0, 2.0, 12.0, 15.0),
        "RBC": (4.1, 5.1, 3.6, 3.1, 5.4, 5.8),
        "HGB": (120.0, 155.0, 110.0, 95.0, 165.0, 175.0),
        "HCT": (0.36, 0.46, 0.32, 0.28, 0.48, 0.52),
        "MCV": (80.0, 96.0, 75.0, 70.0, 100.0, 105.0),
        "MCH": (27.0, 33.0, 24.0, 21.0, 34.0, 36.0),
        "MCHC": (320.0, 360.0, 300.0, 280.0, 370.0, 380.0),
        "PLT": (150.0, 400.0, 100.0, 50.0, 450.0, 600.0),
        "RDW": (11.5, 14.5, 10.5, 9.5, 15.5, 17.0),
        "NEU": (2.0, 7.5, 1.5, 1.0, 8.5, 12.0),
        "LYM": (1.0, 4.0, 0.8, 0.5, 4.5, 6.0),
        "MONO": (0.2, 0.8, 0.1, 0.05, 1.0, 1.5),
        "EOS": (0.0, 0.5, 0.0, 0.0, 0.7, 1.0),
        "BASO": (0.0, 0.2, 0.0, 0.0, 0.3, 0.5),
    },
}


def decimalize(value: float) -> Decimal:
    return Decimal(str(value))


def upsert(session: Session, model: type, lookup: dict, defaults: dict):
    instance = session.scalar(select(model).filter_by(**lookup))
    if instance is None:
        instance = model(**lookup, **defaults)
        session.add(instance)
    else:
        for key, value in defaults.items():
            setattr(instance, key, value)
    session.flush()
    return instance


def seed_indicators(session: Session) -> dict[str, Indicator]:
    indicator_map: dict[str, Indicator] = {}
    for indicator_data in INDICATORS:
        indicator = upsert(
            session,
            Indicator,
            {"code": indicator_data["code"]},
            {
                "name": indicator_data["name"],
                "unit": indicator_data["unit"],
                "description": indicator_data["description"],
            },
        )
        indicator_map[indicator.code] = indicator
    return indicator_map


def seed_deviation_states(session: Session) -> dict[str, DeviationState]:
    deviation_state_map: dict[str, DeviationState] = {}
    for state_data in DEVIATION_STATES:
        state = upsert(
            session,
            DeviationState,
            {"code": state_data["code"]},
            {
                "name": state_data["name"],
                "severity_rank": state_data["severity_rank"],
            },
        )
        deviation_state_map[state.code] = state
    return deviation_state_map


def seed_diseases(session: Session) -> dict[str, Disease]:
    disease_map: dict[str, Disease] = {}
    for disease_data in DISEASES:
        disease = upsert(
            session,
            Disease,
            {"code": disease_data["code"]},
            {
                "name": disease_data["name"],
                "category": disease_data["category"],
                "description": disease_data["description"],
                "severity_level": disease_data["severity_level"],
                "is_active": disease_data["is_active"],
            },
        )
        disease_map[disease.code] = disease
    return disease_map


def seed_reference_ranges(session: Session, indicator_map: dict[str, Indicator]) -> None:
    for sex, ranges in REFERENCE_RANGES.items():
        for indicator_code, values in ranges.items():
            normal_min, normal_max, moderate_low, severe_low, moderate_high, severe_high = values
            upsert(
                session,
                ReferenceRange,
                {
                    "indicator_id": indicator_map[indicator_code].id,
                    "sex": sex,
                    "age_min": ADULT_AGE_MIN,
                    "age_max": ADULT_AGE_MAX,
                },
                {
                    "normal_min": decimalize(normal_min),
                    "normal_max": decimalize(normal_max),
                    "mild_low_threshold": decimalize(normal_min),
                    "moderate_low_threshold": decimalize(moderate_low),
                    "severe_low_threshold": decimalize(severe_low),
                    "mild_high_threshold": decimalize(normal_max),
                    "moderate_high_threshold": decimalize(moderate_high),
                    "severe_high_threshold": decimalize(severe_high),
                },
            )


def add_indicator_rule(
    session: Session,
    indicator_map: dict[str, Indicator],
    state_map: dict[str, DeviationState],
    disease_map: dict[str, Disease],
    *,
    indicator_code: str,
    state_code: str,
    disease_code: str,
    relation_type: IndicatorRuleRelationType,
    weight: float,
    evidence_note: str,
) -> None:
    upsert(
        session,
        IndicatorRule,
        {
            "indicator_id": indicator_map[indicator_code].id,
            "deviation_state_id": state_map[state_code].id,
            "disease_id": disease_map[disease_code].id,
            "relation_type": relation_type,
        },
        {
            "weight": decimalize(weight),
            "evidence_note": evidence_note,
        },
    )


def add_rule_series(
    session: Session,
    indicator_map: dict[str, Indicator],
    state_map: dict[str, DeviationState],
    disease_map: dict[str, Disease],
    *,
    indicator_code: str,
    disease_code: str,
    relation_type: IndicatorRuleRelationType,
    state_weights: dict[str, float],
    template: str,
) -> None:
    for state_code, weight in state_weights.items():
        add_indicator_rule(
            session,
            indicator_map,
            state_map,
            disease_map,
            indicator_code=indicator_code,
            state_code=state_code,
            disease_code=disease_code,
            relation_type=relation_type,
            weight=weight,
            evidence_note=template.format(state_code=state_code.replace("_", " "), indicator_code=indicator_code),
        )


def seed_indicator_rules(
    session: Session,
    indicator_map: dict[str, Indicator],
    state_map: dict[str, DeviationState],
    disease_map: dict[str, Disease],
) -> None:
    low_states = {"mild_low": 2.0, "moderate_low": 3.0, "severe_low": 4.0}
    lower_support = {"mild_low": 1.0, "moderate_low": 1.8, "severe_low": 2.6}
    high_states = {"mild_high": 2.0, "moderate_high": 3.0, "severe_high": 4.0}

    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="HGB",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights=low_states,
        template="{indicator_code} {state_code} supported iron deficiency anemia.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="HGB",
        disease_code="macrocytic_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_low": 1.5, "moderate_low": 2.5, "severe_low": 3.5},
        template="{indicator_code} {state_code} supported macrocytic anemia.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="RBC",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights=lower_support,
        template="{indicator_code} {state_code} reinforced a hypoproliferative anemia pattern.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="HCT",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_low": 1.0, "moderate_low": 1.5, "severe_low": 2.0},
        template="{indicator_code} {state_code} reinforced anemia support.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="MCV",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_low": 2.5, "moderate_low": 3.5, "severe_low": 4.5},
        template="{indicator_code} {state_code} strongly supported microcytic anemia.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="MCV",
        disease_code="macrocytic_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights=high_states,
        template="{indicator_code} {state_code} strongly supported macrocytosis.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="MCV",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        state_weights={"mild_high": 2.0, "moderate_high": 3.0, "severe_high": 4.0},
        template="{indicator_code} {state_code} contradicted iron deficiency anemia.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="MCV",
        disease_code="macrocytic_anemia",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        state_weights={"mild_low": 2.0, "moderate_low": 3.0, "severe_low": 4.0},
        template="{indicator_code} {state_code} contradicted macrocytic anemia.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="MCH",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_low": 1.5, "moderate_low": 2.0, "severe_low": 2.5},
        template="{indicator_code} {state_code} supported hypochromia in iron deficiency anemia.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="RDW",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_high": 1.0, "moderate_high": 1.5, "severe_high": 2.0},
        template="{indicator_code} {state_code} supported anisocytosis consistent with iron deficiency.",
    )
    add_indicator_rule(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="HGB",
        state_code="normal",
        disease_code="iron_deficiency_anemia",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        weight=2.0,
        evidence_note="Normal HGB contradicted clinically significant iron deficiency anemia.",
    )
    add_indicator_rule(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="HGB",
        state_code="normal",
        disease_code="macrocytic_anemia",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        weight=2.0,
        evidence_note="Normal HGB contradicted clinically significant macrocytic anemia.",
    )

    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="WBC",
        disease_code="bacterial_infection",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_high": 1.5, "moderate_high": 2.5, "severe_high": 3.5},
        template="{indicator_code} {state_code} supported a leukocytosis-driven bacterial pattern.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="NEU",
        disease_code="bacterial_infection",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_high": 2.0, "moderate_high": 3.0, "severe_high": 4.0},
        template="{indicator_code} {state_code} strongly supported neutrophil-predominant inflammation.",
    )
    add_indicator_rule(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="WBC",
        state_code="normal",
        disease_code="bacterial_infection",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        weight=1.0,
        evidence_note="Normal WBC weakened support for a bacterial infection pattern.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="NEU",
        disease_code="bacterial_infection",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        state_weights={"mild_low": 1.0, "moderate_low": 1.5, "severe_low": 2.0},
        template="{indicator_code} {state_code} contradicted a neutrophil-predominant bacterial pattern.",
    )

    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="WBC",
        disease_code="viral_infection",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_high": 1.0, "moderate_high": 1.5, "severe_high": 2.0},
        template="{indicator_code} {state_code} supported a reactive viral pattern.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="LYM",
        disease_code="viral_infection",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_high": 2.0, "moderate_high": 3.0, "severe_high": 4.0},
        template="{indicator_code} {state_code} strongly supported lymphocyte-predominant activation.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="LYM",
        disease_code="viral_infection",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        state_weights={"mild_low": 1.0, "moderate_low": 1.5, "severe_low": 2.0},
        template="{indicator_code} {state_code} contradicted a viral lymphocytic pattern.",
    )

    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="EOS",
        disease_code="allergic_or_parasitic_pattern",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights=high_states,
        template="{indicator_code} {state_code} supported eosinophil-predominant allergic or parasitic activity.",
    )

    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="PLT",
        disease_code="thrombocytopenia_pattern",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights=low_states,
        template="{indicator_code} {state_code} supported thrombocytopenia.",
    )
    add_indicator_rule(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="PLT",
        state_code="normal",
        disease_code="thrombocytopenia_pattern",
        relation_type=IndicatorRuleRelationType.CONTRADICT,
        weight=2.0,
        evidence_note="Normal platelets contradicted thrombocytopenia.",
    )

    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="WBC",
        disease_code="hematologic_malignancy_suspicion",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"moderate_high": 2.0, "severe_high": 3.0},
        template="{indicator_code} {state_code} increased concern for hematologic malignancy.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="HGB",
        disease_code="hematologic_malignancy_suspicion",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"moderate_low": 1.5, "severe_low": 2.5},
        template="{indicator_code} {state_code} increased concern for marrow pathology.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="PLT",
        disease_code="hematologic_malignancy_suspicion",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"moderate_low": 1.5, "severe_low": 2.5},
        template="{indicator_code} {state_code} increased concern for multilineage involvement.",
    )
    add_rule_series(
        session,
        indicator_map,
        state_map,
        disease_map,
        indicator_code="BASO",
        disease_code="hematologic_malignancy_suspicion",
        relation_type=IndicatorRuleRelationType.SUPPORT,
        state_weights={"mild_high": 1.0, "moderate_high": 1.5, "severe_high": 2.0},
        template="{indicator_code} {state_code} increased concern for atypical myeloid activity.",
    )


def seed_pattern_rules(
    session: Session,
    indicator_map: dict[str, Indicator],
    disease_map: dict[str, Disease],
) -> None:
    pattern_definitions = [
        {
            "code": "iron_deficiency_pattern",
            "name": "Iron Deficiency Pattern",
            "disease_code": "iron_deficiency_anemia",
            "bonus_weight": 4.0,
            "rule_description": "Pattern match: low HGB, low MCV, low MCH, and high RDW supported iron deficiency anemia.",
            "conditions": [
                {"indicator_code": "HGB", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.LOW},
                {"indicator_code": "MCV", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.LOW},
                {"indicator_code": "MCH", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.LOW},
                {"indicator_code": "RDW", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.HIGH},
            ],
        },
        {
            "code": "bacterial_pattern",
            "name": "Bacterial Infection Pattern",
            "disease_code": "bacterial_infection",
            "bonus_weight": 3.5,
            "rule_description": "Pattern match: high WBC with high neutrophils supported a bacterial infection hypothesis.",
            "conditions": [
                {"indicator_code": "WBC", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.HIGH},
                {"indicator_code": "NEU", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.HIGH},
            ],
        },
        {
            "code": "viral_pattern",
            "name": "Viral Infection Pattern",
            "disease_code": "viral_infection",
            "bonus_weight": 3.5,
            "rule_description": "Pattern match: compatible leukocyte elevation with lymphocytosis supported a viral infection hypothesis.",
            "conditions": [
                {"indicator_code": "WBC", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.HIGH},
                {"indicator_code": "LYM", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.HIGH},
            ],
        },
        {
            "code": "thrombocytopenia_pattern",
            "name": "Thrombocytopenia Pattern",
            "disease_code": "thrombocytopenia_pattern",
            "bonus_weight": 3.0,
            "rule_description": "Pattern match: platelet count in the low family supported thrombocytopenia.",
            "conditions": [
                {"indicator_code": "PLT", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.LOW},
            ],
        },
        {
            "code": "hematologic_alarm_pattern",
            "name": "Hematologic Alarm Pattern",
            "disease_code": "hematologic_malignancy_suspicion",
            "bonus_weight": 4.0,
            "rule_description": "Pattern match: concurrent leukocytosis, anemia, and thrombocytopenia increased suspicion for hematologic malignancy.",
            "conditions": [
                {"indicator_code": "WBC", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.HIGH},
                {"indicator_code": "HGB", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.LOW},
                {"indicator_code": "PLT", "match_mode": PatternMatchMode.FAMILY, "deviation_family": DeviationFamily.LOW},
            ],
        },
    ]

    for pattern_definition in pattern_definitions:
        pattern_rule = upsert(
            session,
            PatternRule,
            {"code": pattern_definition["code"]},
            {
                "name": pattern_definition["name"],
                "disease_id": disease_map[pattern_definition["disease_code"]].id,
                "bonus_weight": decimalize(pattern_definition["bonus_weight"]),
                "rule_description": pattern_definition["rule_description"],
            },
        )
        session.execute(
            delete(PatternRuleCondition).where(PatternRuleCondition.pattern_rule_id == pattern_rule.id)
        )
        session.flush()
        for condition in pattern_definition["conditions"]:
            session.add(
                PatternRuleCondition(
                    pattern_rule_id=pattern_rule.id,
                    indicator_id=indicator_map[condition["indicator_code"]].id,
                    match_mode=condition["match_mode"],
                    deviation_family=condition.get("deviation_family"),
                    deviation_state_id=None,
                )
            )
        session.flush()


def seed_database(session: Session) -> None:
    indicator_map = seed_indicators(session)
    deviation_state_map = seed_deviation_states(session)
    disease_map = seed_diseases(session)
    seed_reference_ranges(session, indicator_map)
    seed_indicator_rules(session, indicator_map, deviation_state_map, disease_map)
    seed_pattern_rules(session, indicator_map, disease_map)


def main() -> None:
    settings = get_settings()
    initialize_engine(settings.database_url)
    session_factory = get_session_factory()
    with session_factory() as session:
        with session.begin():
            seed_database(session)


if __name__ == "__main__":
    main()
