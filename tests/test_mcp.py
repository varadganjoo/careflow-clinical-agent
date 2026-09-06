import json
import pytest
from mcp_server.server import (
    check_drug_interactions,
    query_clinical_guidelines,
    evaluate_patient_safety,
    get_patient_record,
    get_hypertension_guidelines,
)


def test_mcp_check_drug_interactions():
    res = check_drug_interactions(["lisinopril", "losartan"])
    assert len(res) >= 1
    assert "Dual Renin-Angiotensin" in res[0]["title"]

    res_safe = check_drug_interactions(["acetaminophen", "amoxicillin"])
    assert len(res_safe) == 0


def test_mcp_query_guidelines():
    # Hypertension
    res_htn = query_clinical_guidelines("hypertension", sbp=152.0, dbp=94.0)
    assert res_htn["guideline"] == "ACC/AHA 2024"
    assert res_htn["category"] == "Stage 2 Hypertension"

    # Diabetes
    res_dm = query_clinical_guidelines("diabetes", hba1c=8.2, egfr=25.0)
    assert res_dm["guideline"] == "ADA 2025/2026"
    assert "Uncontrolled" in res_dm["status"]


def test_mcp_evaluate_patient_safety():
    res = evaluate_patient_safety("pat-003")
    assert res["alert_count"] >= 1
    titles = [a["title"] for a in res["alerts"]]
    assert any("Dual Renin-Angiotensin" in t for t in titles)


def test_mcp_resources():
    patient_json = get_patient_record("pat-001")
    data = json.loads(patient_json)
    assert data["patient"]["name"] == "Eleanor Vance"

    guidelines = get_hypertension_guidelines()
    assert "ACC/AHA" in guidelines
