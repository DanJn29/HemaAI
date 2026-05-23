from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class IndicatorInputLimit:
    indicator_code: str
    min_allowed: Decimal
    max_allowed: Decimal
    unit: str
    warning_low: Decimal | None = None
    warning_high: Decimal | None = None


INDICATOR_INPUT_LIMITS: dict[str, IndicatorInputLimit] = {
    "WBC": IndicatorInputLimit("WBC", Decimal("0.1"), Decimal("300"), "10^9/L"),
    "RBC": IndicatorInputLimit("RBC", Decimal("0.5"), Decimal("10"), "10^12/L"),
    "HGB": IndicatorInputLimit("HGB", Decimal("20"), Decimal("250"), "g/L"),
    "HCT": IndicatorInputLimit("HCT", Decimal("0.05"), Decimal("0.75"), "L/L"),
    "MCV": IndicatorInputLimit("MCV", Decimal("40"), Decimal("150"), "fL"),
    "MCH": IndicatorInputLimit("MCH", Decimal("10"), Decimal("60"), "pg"),
    "MCHC": IndicatorInputLimit("MCHC", Decimal("200"), Decimal("450"), "g/L"),
    "PLT": IndicatorInputLimit("PLT", Decimal("1"), Decimal("2000"), "10^9/L"),
    "RDW": IndicatorInputLimit("RDW", Decimal("5"), Decimal("40"), "%"),
    "NEU": IndicatorInputLimit("NEU", Decimal("0"), Decimal("100"), "10^9/L"),
    "LYM": IndicatorInputLimit("LYM", Decimal("0"), Decimal("100"), "10^9/L"),
    "MONO": IndicatorInputLimit("MONO", Decimal("0"), Decimal("100"), "10^9/L"),
    "EOS": IndicatorInputLimit("EOS", Decimal("0"), Decimal("100"), "10^9/L"),
    "BASO": IndicatorInputLimit("BASO", Decimal("0"), Decimal("100"), "10^9/L"),
}


def get_indicator_input_limit(indicator_code: str) -> IndicatorInputLimit | None:
    return INDICATOR_INPUT_LIMITS.get(indicator_code.upper())
