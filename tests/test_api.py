"""API tests for the physician UI: patient data, safety checks, consult -> interrupt -> sign-off, and failure paths."""

import json

import pytest
from fastapi.testclient import TestClient

from app.llm import SOAPNote
from app.main import app

client = TestClient(app)

MOCK_NOTE = SOAPNote(
    subjective="Morning headaches.",
    objective="BP 154/96 mmHg.",
    assessment="Stage 2 hypertension, uncontrolled on monotherapy.",
    plan=["Add a second first-line agent.", "Recheck BMP in 2 weeks."],
    icd10_codes=["I10"],
    cpt_codes=["99214"],
)


@pytest.fixture(autouse=True)
def captured_prompts(monkeypatch):
    prompts: list[str] = []

    def fake_generate(prompt, system_instruction, schema, **kwargs):
        prompts.append(prompt)
        return MOCK_NOTE, 100, 100

    monkeypatch.setattr("app.graph.generate_structured", fake_generate)
    return prompts


def start_consult(patient_id="pat-001"):
    response = client.post("/api/consult", json={"patient_id": patient_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_patient_list_ids_match_names_and_flag_alerts():
    patients = {p["id"]: p for p in client.get("/api/patients").json()}
    assert patients["pat-002"]["name"] == "Marcus Holloway"
    assert patients["pat-003"]["name"] == "Robert Chen"
    assert patients["pat-001"]["alert_count"] == 0
    assert patients["pat-003"]["max_severity"] == "CRITICAL"


def test_patient_lookup_is_exact():
    assert client.get("/api/patients/pat-003").json()["patient"]["name"] == "Robert Chen"
    assert client.get("/api/patients/pat-0").status_code == 404
    assert client.get("/api/patients/pat-999").status_code == 404


def test_safety_endpoint_flags_dual_ras_and_hyperkalemia_for_robert():
    titles = [a["title"] for a in client.get("/api/patients/pat-003/safety").json()["alerts"]]
    assert any("Hyperkalemia" in t for t in titles)
    assert len(titles) >= 2


def test_consult_pauses_for_physician_with_note_and_alerts():
    body = start_consult("pat-002")
    assert body["status"] == "AWAITING_PHYSICIAN_REVIEW"
    assert body["state"]["soap_note"]["icd10_codes"] == ["I10"]
    assert any("Metformin" in a["title"] for a in body["state"]["safety_alerts"])
    assert body["state"]["redaction_count"] > 0


def test_transcript_reaches_scribe_without_patient_names(captured_prompts):
    start_consult("pat-001")
    prompt = captured_prompts[-1]
    assert "home readings are still running in the 150s" in prompt
    assert "Eleanor" not in prompt
    assert "Vance" not in prompt
    assert "MRN-92841" not in prompt


def test_approve_commits_to_ehr():
    session = start_consult()["session_id"]
    body = client.post(f"/api/consult/{session}/resume", json={"action": "approve"}).json()
    assert body["status"] == "APPROVED"
    assert body["ehr_committed"] is True


def test_edit_applies_physician_changes():
    session = start_consult()["session_id"]
    edits = {"assessment": "Stage 2 HTN. Physician revised.", "plan": ["Start amlodipine 5 mg daily."]}
    body = client.post(f"/api/consult/{session}/resume", json={"action": "edit", "edits": edits}).json()
    assert body["status"] == "EDITED"
    assert body["final_soap"]["assessment"] == "Stage 2 HTN. Physician revised."
    assert body["final_soap"]["plan"] == ["Start amlodipine 5 mg daily."]


def test_reject_does_not_commit():
    session = start_consult()["session_id"]
    body = client.post(f"/api/consult/{session}/resume", json={"action": "reject", "notes": "Wrong plan."}).json()
    assert body["status"] == "REJECTED"
    assert body["ehr_committed"] is False


def test_sign_off_twice_is_409():
    session = start_consult()["session_id"]
    assert client.post(f"/api/consult/{session}/resume", json={"action": "approve"}).status_code == 200
    assert client.post(f"/api/consult/{session}/resume", json={"action": "approve"}).status_code == 409


def test_edits_cannot_inject_non_soap_fields():
    session = start_consult()["session_id"]
    response = client.post(f"/api/consult/{session}/resume", json={"action": "edit", "edits": {"ehr_committed": True}})
    assert response.status_code == 422


def test_unknown_action_and_session_are_rejected():
    session = start_consult()["session_id"]
    assert client.post(f"/api/consult/{session}/resume", json={"action": "delete"}).status_code == 422
    assert client.post("/api/consult/enc-nope/resume", json={"action": "approve"}).status_code == 404


@pytest.mark.parametrize(
    "failure, status, phrase",
    [
        ("Gemini API generation failed: 429 RESOURCE_EXHAUSTED", 429, "rate limit"),
        ("Gemini API generation failed: 503 UNAVAILABLE", 503, "overloaded"),
        ("GEMINI_API_KEY environment variable is not configured.", 500, "not configured"),
        ("Gemini API generation failed: GenerateRequestsPerDayPerProjectPerModel-FreeTier 429 RESOURCE_EXHAUSTED", 429, "daily"),
    ],
)
def test_consult_llm_failures_return_actionable_errors(monkeypatch, failure, status, phrase):
    def boom(*args, **kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr("app.graph.generate_structured", boom)
    response = client.post("/api/consult", json={"patient_id": "pat-001"})
    assert response.status_code == status
    assert phrase in response.json()["detail"]


def test_soap_stream_failure_becomes_error_event(monkeypatch):
    def boom(*args, **kwargs):
        yield "Partial note"
        raise RuntimeError("Gemini streaming SOAP generation failed: 503 UNAVAILABLE")

    monkeypatch.setattr("app.main.stream_soap_synthesis", boom)
    blocks = client.get("/api/encounters/pat-001/stream").text.strip().split("\n\n")
    last_event = blocks[-1].splitlines()[0]
    assert last_event == "event: error"
    assert json.loads(blocks[-1].splitlines()[1][6:])["retryable"] is True


def test_soap_stream_unknown_patient_is_404():
    assert client.get("/api/encounters/pat-999/stream").status_code == 404
