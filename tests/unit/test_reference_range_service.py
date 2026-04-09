import pytest
from sqlalchemy import func, select

from app.core.exceptions import DomainValidationError
from app.models.indicator import Indicator
from app.models.reference_range import ReferenceRange
from app.services.reference_range_service import ReferenceRangeService


def test_seeded_reference_ranges_cover_all_indicators_and_buckets(db_session) -> None:
    total_rows = db_session.scalar(select(func.count()).select_from(ReferenceRange))
    assert total_rows == 84

    grouped_counts = db_session.execute(
        select(Indicator.code, ReferenceRange.sex, func.count(ReferenceRange.id))
        .join(ReferenceRange.indicator)
        .group_by(Indicator.code, ReferenceRange.sex)
    ).all()

    assert len(grouped_counts) == 28
    assert all(count == 3 for _, _, count in grouped_counts)


def test_reference_range_service_resolves_exact_age_buckets(db_session) -> None:
    indicator = db_session.scalar(select(Indicator).where(Indicator.code == "HGB"))
    assert indicator is not None
    service = ReferenceRangeService(db_session)

    young_adult = service.get_for_indicator(indicator.id, "female", 18)
    young_boundary = service.get_for_indicator(indicator.id, "female", 40)
    middle_adult = service.get_for_indicator(indicator.id, "female", 41)
    middle_boundary = service.get_for_indicator(indicator.id, "female", 65)
    elderly = service.get_for_indicator(indicator.id, "female", 66)
    elderly_boundary = service.get_for_indicator(indicator.id, "female", 120)

    assert (young_adult.age_min, young_adult.age_max) == (18, 40)
    assert (young_boundary.age_min, young_boundary.age_max) == (18, 40)
    assert (middle_adult.age_min, middle_adult.age_max) == (41, 65)
    assert (middle_boundary.age_min, middle_boundary.age_max) == (41, 65)
    assert (elderly.age_min, elderly.age_max) == (66, 120)
    assert (elderly_boundary.age_min, elderly_boundary.age_max) == (66, 120)


def test_reference_range_service_raises_for_missing_match(db_session) -> None:
    indicator = db_session.scalar(select(Indicator).where(Indicator.code == "WBC"))
    assert indicator is not None
    service = ReferenceRangeService(db_session)

    with pytest.raises(DomainValidationError, match="No reference range configured"):
        service.get_for_indicator(indicator.id, "male", 17)


def test_reference_range_service_raises_for_ambiguous_matches(db_session) -> None:
    service = ReferenceRangeService(db_session)
    nested = db_session.begin_nested()
    try:
        indicator = Indicator(code="TEST_HGB", name="Test Hemoglobin", unit="g/L", description="test indicator")
        db_session.add(indicator)
        db_session.flush()
        db_session.add_all(
            [
                ReferenceRange(
                    indicator_id=indicator.id,
                    sex="male",
                    age_min=18,
                    age_max=40,
                    normal_min=100,
                    normal_max=150,
                    mild_low_threshold=100,
                    moderate_low_threshold=90,
                    severe_low_threshold=80,
                    mild_high_threshold=150,
                    moderate_high_threshold=160,
                    severe_high_threshold=170,
                ),
                ReferenceRange(
                    indicator_id=indicator.id,
                    sex="male",
                    age_min=30,
                    age_max=50,
                    normal_min=100,
                    normal_max=150,
                    mild_low_threshold=100,
                    moderate_low_threshold=90,
                    severe_low_threshold=80,
                    mild_high_threshold=150,
                    moderate_high_threshold=160,
                    severe_high_threshold=170,
                ),
            ]
        )
        db_session.flush()

        with pytest.raises(DomainValidationError, match="Ambiguous reference range configuration"):
            service.get_for_indicator(indicator.id, "male", 35)
    finally:
        nested.rollback()
