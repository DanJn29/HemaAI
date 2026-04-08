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
            "age": 40,
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


def test_analysis_endpoint_returns_normal_fallback_for_low_signal_profile(client) -> None:
    response = client.post(
        "/api/v1/analyses",
        json={
            "sex": "male",
            "age": 36,
            "values": {
                "WBC": 10.1,
                "NEU": 7.4,
                "HGB": 150,
                "PLT": 220,
            },
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["top_hypotheses"][0]["disease_code"] == "normal"


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
