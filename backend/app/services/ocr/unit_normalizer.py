from decimal import Decimal


class UnitNormalizationResult:
    def __init__(self, value: Decimal, warnings: list[str] | None = None) -> None:
        self.value = value
        self.warnings = warnings or []


def normalize_indicator_value(indicator_code: str, value: Decimal, unit_text: str) -> UnitNormalizationResult:
    normalized_unit = _normalize_unit_text(unit_text)
    warnings: list[str] = []

    if indicator_code in {"HGB", "MCHC"}:
        if "g/dl" in normalized_unit or "g dl" in normalized_unit:
            warnings.append(f"{indicator_code} converted from g/dL to g/L. Please review.")
            return UnitNormalizationResult(value * Decimal("10"), warnings)
        if indicator_code == "HGB" and not normalized_unit:
            warnings.append("HGB unit was unclear. Please review the extracted value.")

    if indicator_code == "HCT":
        if "%" in normalized_unit or "percent" in normalized_unit:
            warnings.append("HCT converted from percent to fraction. Please review.")
            return UnitNormalizationResult(value / Decimal("100"), warnings)
        if value > Decimal("1"):
            warnings.append("HCT looked like a percent value and was converted to fraction. Please review.")
            return UnitNormalizationResult(value / Decimal("100"), warnings)
        if not normalized_unit:
            warnings.append("HCT unit was unclear. Please review the extracted value.")

    if indicator_code in {"NEU", "LYM", "MONO", "EOS", "BASO"} and "%" in normalized_unit:
        warnings.append(f"{indicator_code} appears to be a percent differential. Please review before analysis.")

    return UnitNormalizationResult(value, warnings)


def _normalize_unit_text(unit_text: str) -> str:
    return (
        unit_text.lower()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("／", "/")
        .replace("⁹", "9")
        .replace("¹", "1")
        .replace("²", "2")
        .replace("³", "3")
        .strip()
    )
