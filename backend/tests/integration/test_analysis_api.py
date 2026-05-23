from decimal import Decimal

from sqlalchemy import select

from app.models.analysis_case import AnalysisCase
from app.models.analysis_value import AnalysisValue


def test_analysis_endpoint_ranks_iron_deficiency_first(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "female",
            "age": 28,
            "values": {
                "HGB": 109,
                "MCV": 72,
                "MCH": 23,
                "RDW": 16.8,
                "RBC": 3.9,
                "PLT": 250,
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["top_hypotheses"][0]["disease_code"] == "iron_deficiency_anemia"
    assert any(item["type"] == "pattern_rule" for item in payload["top_hypotheses"][0]["explanations"])


def test_analysis_endpoint_ranks_bacterial_infection_first(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "male",
            "age": 52,
            "values": {
                "WBC": 15.4,
                "NEU": 12.8,
                "LYM": 1.4,
                "HGB": 150,
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["top_hypotheses"][0]["disease_code"] == "bacterial_infection"


def test_analysis_endpoint_returns_thrombocytopenia_pattern_for_elderly_adult(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "female",
            "age": 72,
            "values": {
                "PLT": 78,
                "HGB": 126,
                "RBC": 4.0,
                "HCT": 0.37,
                "MCV": 90,
                "WBC": 6.8,
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["top_hypotheses"][0]["disease_code"] == "thrombocytopenia_pattern"


def test_analysis_endpoint_returns_normal_fallback_for_low_signal_profile(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "male",
            "age": 72,
            "values": {
                "WBC": 10.1,
                "NEU": 7.4,
                "HGB": 145,
                "PLT": 220,
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["top_hypotheses"][0]["disease_code"] == "normal"


def test_analysis_endpoint_rejects_out_of_range_indicator_value(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "female",
            "age": 28,
            "values": {
                "WBC": 999,
                "HGB": 130,
            },
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["message"] == "One or more CBC indicator values are outside allowed input limits."
    assert detail["errors"] == [
        {
            "indicator_code": "WBC",
            "provided_value": 999,
            "min_allowed": 0.1,
            "max_allowed": 300,
            "message": "WBC must be between 0.1 and 300 10^9/L.",
        }
    ]


def test_analysis_endpoint_accepts_severe_but_plausible_indicator_value(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "male",
            "age": 52,
            "values": {
                "WBC": 20,
                "NEU": 15,
                "HGB": 150,
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    wbc_interpretation = next(
        item for item in payload["indicator_interpretation"] if item["indicator_code"] == "WBC"
    )
    assert wbc_interpretation["deviation_state"] == "severe_high"


def test_indicator_metadata_endpoint_returns_input_limits_and_normal_range(client) -> None:
    response = client.get("/api/v1/indicators/metadata?sex=male&age=35")

    assert response.status_code == 200, response.text
    metadata = response.json()
    wbc_metadata = next(item for item in metadata if item["code"] == "WBC")
    assert wbc_metadata["unit"] == "10^9/L"
    assert wbc_metadata["normal_min"] == 4.0
    assert wbc_metadata["normal_max"] == 10.0
    assert wbc_metadata["min_allowed"] == 0.1
    assert wbc_metadata["max_allowed"] == 300


def test_recompute_refreshes_deviation_states_and_results(client, db_session) -> None:
    create_response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "female",
            "age": 32,
            "values": {
                "PLT": 210,
                "HGB": 130,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    analysis_id = create_response.json()["analysis_id"]

    analysis_case = db_session.scalar(select(AnalysisCase).where(AnalysisCase.id == analysis_id))
    assert analysis_case is not None
    platelet_value = db_session.scalar(
        select(AnalysisValue)
        .where(AnalysisValue.analysis_case_id == analysis_id)
        .where(AnalysisValue.indicator.has(code="PLT"))
    )
    assert platelet_value is not None
    platelet_value.raw_value = Decimal("70")
    db_session.commit()

    recompute_response = client.post(f"/api/v1/recompute/{analysis_id}")
    assert recompute_response.status_code == 200, recompute_response.text
    payload = recompute_response.json()

    assert payload["top_hypotheses"][0]["disease_code"] == "thrombocytopenia_pattern"
    platelet_interpretation = next(
        item for item in payload["indicator_interpretation"] if item["indicator_code"] == "PLT"
    )
    assert platelet_interpretation["deviation_state"] in {"moderate_low", "severe_low"}
    assert platelet_interpretation["normalized_score"] is not None
