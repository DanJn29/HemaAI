from collections import defaultdict

from app.models.enums import Sex
from app.seed.seed_data import AGE_BUCKETS, INDICATORS, build_reference_range_rows


def test_build_reference_range_rows_has_complete_non_overlapping_coverage() -> None:
    rows = build_reference_range_rows()

    assert len(rows) == len(INDICATORS) * len(Sex) * len(AGE_BUCKETS) == 84

    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["indicator_code"]), str(row["sex"]))].append((int(row["age_min"]), int(row["age_max"])))
        assert row["mild_low_threshold"] == row["normal_min"]
        assert row["mild_high_threshold"] == row["normal_max"]

    assert len(grouped) == len(INDICATORS) * len(Sex)
    expected_buckets = list(AGE_BUCKETS)
    for buckets in grouped.values():
        assert sorted(buckets) == expected_buckets


def test_build_reference_range_rows_preserves_sex_specific_baselines_and_platelet_placeholders() -> None:
    rows = build_reference_range_rows()
    indexed = {
        (str(row["indicator_code"]), str(row["sex"]), int(row["age_min"]), int(row["age_max"])): row
        for row in rows
    }

    male_hgb_young = indexed[("HGB", "male", 18, 40)]
    female_hgb_young = indexed[("HGB", "female", 18, 40)]
    elderly_platelets = indexed[("PLT", "female", 66, 120)]

    assert male_hgb_young["normal_min"] == 135
    assert male_hgb_young["normal_max"] == 175
    assert female_hgb_young["normal_min"] == 120
    assert female_hgb_young["normal_max"] == 155
    assert elderly_platelets["normal_max"] == 380
    assert elderly_platelets["mild_high_threshold"] == elderly_platelets["normal_max"]
    assert elderly_platelets["moderate_high_threshold"] == 420
    assert elderly_platelets["severe_high_threshold"] == 550
