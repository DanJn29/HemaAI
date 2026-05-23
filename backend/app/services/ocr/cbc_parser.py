from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

from app.core.indicator_input_limits import INDICATOR_INPUT_LIMITS
from app.services.ocr.unit_normalizer import normalize_indicator_value


INDICATOR_CODES = tuple(INDICATOR_INPUT_LIMITS.keys())
ALLOWED_SEXES = {"male", "female"}

ALIASES: dict[str, tuple[str, ...]] = {
    "WBC": ("WBC", "Leukocytes", "White Blood Cells"),
    "RBC": ("RBC", "Erythrocytes", "Red Blood Cells"),
    "HGB": ("HGB", "Hb", "Hemoglobin"),
    "HCT": ("HCT", "Hematocrit"),
    "MCV": ("MCV",),
    "MCH": ("MCH",),
    "MCHC": ("MCHC",),
    "PLT": ("PLT", "Platelets", "Thrombocytes"),
    "RDW": ("RDW",),
    "NEU": ("NEU", "Neutrophils"),
    "LYM": ("LYM", "Lymphocytes"),
    "MONO": ("MONO", "Monocytes"),
    "EOS": ("EOS", "Eosinophils"),
    "BASO": ("BASO", "Basophils"),
}

NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:[\.,]\d+)?")
UNIT_PATTERN = re.compile(r"[%a-zA-Z/^\dµμ\*\.\s]+")


@dataclass(frozen=True, slots=True)
class PatientDemographics:
    age: int | None = None
    sex: str | None = None


@dataclass(frozen=True, slots=True)
class CBCParseResult:
    extracted_values: dict[str, float | None]
    patient: PatientDemographics
    raw_text: str
    warnings: list[str]


def parse_cbc_text(raw_text: str) -> CBCParseResult:
    extracted_values: dict[str, float | None] = {code: None for code in INDICATOR_CODES}
    warnings: list[str] = []
    patient, patient_warnings = parse_patient_demographics(raw_text)
    warnings.extend(patient_warnings)

    for line in raw_text.splitlines():
        for indicator_code, alias in _find_aliases(line):
            if extracted_values[indicator_code] is not None:
                continue

            value, unit_text = _extract_value_after_alias(line, alias)
            if value is None:
                continue

            normalized = normalize_indicator_value(indicator_code, value, unit_text)
            warnings.extend(_dedupe_new(normalized.warnings, warnings))

            limit = INDICATOR_INPUT_LIMITS[indicator_code]
            if normalized.value < limit.min_allowed or normalized.value > limit.max_allowed:
                warnings.append(
                    f"{indicator_code} value {normalized.value} is outside allowed input range "
                    f"{limit.min_allowed}-{limit.max_allowed} {limit.unit}. It was left empty."
                )
                extracted_values[indicator_code] = None
                continue

            extracted_values[indicator_code] = float(normalized.value)

    missing_codes = [code for code in INDICATOR_CODES if extracted_values[code] is None]
    if missing_codes:
        warnings.append("Some CBC indicators were not found and remain empty.")

    return CBCParseResult(
        extracted_values=extracted_values,
        patient=patient,
        raw_text=raw_text,
        warnings=warnings,
    )


def parse_patient_demographics(raw_text: str) -> tuple[PatientDemographics, list[str]]:
    warnings: list[str] = []
    normalized_text = " ".join(raw_text.replace("\r", "\n").split())

    age: int | None = None
    sex: str | None = None

    patterns = (
        re.compile(
            r"age\s*/\s*sex\s*:?\s*(?P<age>\d{1,3})\s*/?\s*(?P<sex>male|female|m|f)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"age\s*:?\s*(?P<age>\d{1,3}).{0,24}?sex\s*:?\s*(?P<sex>male|female|m|f)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<age>\d{1,3})\s*/\s*(?P<sex>male|female|m|f)\b",
            re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.search(normalized_text)
        if match:
            age = int(match.group("age"))
            sex = _normalize_sex(match.group("sex"))
            break

    if age is not None and (age < 18 or age > 120):
        warnings.append(f"Extracted age {age} is outside supported range 18-120 and was left empty.")
        age = None

    return PatientDemographics(age=age, sex=sex), warnings


def _find_aliases(text: str) -> list[tuple[str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    alias_items = [
        (indicator_code, alias)
        for indicator_code, aliases in ALIASES.items()
        for alias in aliases
    ]
    for indicator_code, alias in sorted(alias_items, key=lambda item: len(item[1]), reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            matches.append((match.start(), len(alias), indicator_code, match.group(0)))
    if not matches:
        return []
    matches.sort(key=lambda item: (item[0], -item[1]))
    first_start = matches[0][0]
    return [(indicator_code, alias) for start, _, indicator_code, alias in matches if start == first_start]


def _normalize_sex(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized == "m":
        return "male"
    if normalized == "f":
        return "female"
    if normalized in ALLOWED_SEXES:
        return normalized
    return None


def _extract_value_after_alias(text: str, alias: str) -> tuple[Decimal | None, str]:
    alias_match = re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text, re.IGNORECASE)
    if alias_match is None:
        return None, ""

    tail = text[alias_match.end() :]
    number_match = NUMBER_PATTERN.search(tail)
    if number_match is None:
        return None, ""

    raw_number = number_match.group(0).replace(",", ".")
    try:
        value = Decimal(raw_number)
    except InvalidOperation:
        return None, ""

    unit_tail = tail[number_match.end() : number_match.end() + 32]
    unit_match = UNIT_PATTERN.match(unit_tail)
    unit_text = unit_match.group(0).strip() if unit_match else ""
    return value, unit_text


def _dedupe_new(candidates: list[str], existing: list[str]) -> list[str]:
    return [candidate for candidate in candidates if candidate not in existing]
