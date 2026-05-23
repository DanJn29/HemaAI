from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class IndicatorRuleRelationType(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"


class PatternMatchMode(str, Enum):
    EXACT = "exact"
    FAMILY = "family"


class DeviationFamily(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ExplanationSourceType(str, Enum):
    INDICATOR_RULE = "indicator_rule"
    PATTERN_RULE = "pattern_rule"
    ML_FEATURE = "ml_feature"


class AnalysisSourceType(str, Enum):
    MANUAL = "manual"


class AnalysisResultSource(str, Enum):
    RULE_ENGINE = "rule_engine"


def enum_values(enum_class: type[Enum]) -> list[str]:
    return [member.value for member in enum_class]
