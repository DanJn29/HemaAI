import pytest

from app.services.ocr.cbc_parser import parse_cbc_text
from app.services.ocr.unit_normalizer import normalize_indicator_value
from decimal import Decimal


def test_parse_cbc_text_extracts_common_aliases_and_units() -> None:
    result = parse_cbc_text(
        "\n".join(
            [
                "WBC 6.2 10^9/L",
                "Hemoglobin 14.2 g/dL",
                "HCT 42 %",
                "Platelets 240",
                "Neutrophils 4.1 10^9/L",
            ]
        )
    )

    assert result.extracted_values["WBC"] == pytest.approx(6.2)
    assert result.extracted_values["HGB"] == pytest.approx(142)
    assert result.extracted_values["HCT"] == pytest.approx(0.42)
    assert result.extracted_values["PLT"] == pytest.approx(240)
    assert result.extracted_values["NEU"] == pytest.approx(4.1)
    assert "HGB converted from g/dL to g/L. Please review." in result.warnings
    assert "HCT converted from percent to fraction. Please review." in result.warnings


def test_parse_cbc_text_extracts_patient_age_and_sex() -> None:
    result = parse_cbc_text("Patient Name : John Doe\nAge / Sex : 45 / Male\nWBC 6.2")

    assert result.patient.age == 45
    assert result.patient.sex == "male"


def test_parse_cbc_text_rejects_out_of_range_patient_age() -> None:
    result = parse_cbc_text("Age / Sex : 9 / Female\nWBC 6.2")

    assert result.patient.age is None
    assert result.patient.sex == "female"
    assert any("Extracted age 9 is outside supported range" in warning for warning in result.warnings)


def test_parse_cbc_text_extracts_mchc_before_parenthetical_hgb_alias() -> None:
    result = parse_cbc_text("MCHC (Mean Corpuscular HGB Conc.) 33.3 g/dL 32.0 - 36.0 Normal")

    assert result.extracted_values["MCHC"] == pytest.approx(333)
    assert result.extracted_values["HGB"] is None
    assert "MCHC converted from g/dL to g/L. Please review." in result.warnings


def test_parse_cbc_text_handles_fractional_hct_without_conversion() -> None:
    result = parse_cbc_text("HCT 0.42")

    assert result.extracted_values["HCT"] == pytest.approx(0.42)


def test_unit_normalization_converts_hgb_and_hct_units() -> None:
    hgb = normalize_indicator_value("HGB", Decimal("14.2"), "g/dL")
    hct = normalize_indicator_value("HCT", Decimal("42"), "%")

    assert hgb.value == Decimal("142.0")
    assert hct.value == Decimal("0.42")
    assert hgb.warnings == ["HGB converted from g/dL to g/L. Please review."]
    assert hct.warnings == ["HCT converted from percent to fraction. Please review."]


def test_invalid_extracted_values_are_left_empty_with_warning() -> None:
    result = parse_cbc_text("WBC 999 10^9/L")

    assert result.extracted_values["WBC"] is None
    assert any("WBC value 999 is outside allowed input range" in warning for warning in result.warnings)
