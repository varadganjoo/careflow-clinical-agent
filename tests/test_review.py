"""The physician gate must enforce review rules even without the HTTP API."""

import json
from pathlib import Path

import pytest
from langgraph.types import Command

from app.graph import build_clinical_graph
from app.llm import SOAPNote


@pytest.fixture
def encounter(monkeypatch):
    note = SOAPNote(subjective="Follow-up.", objective="See chart.", assessment="Review needed.", plan=["Review medications."])
    monkeypatch.setattr("app.graph.generate_structured", lambda *args: (note, 1, 1))
    data = json.loads((Path(__file__).parents[1] / "data/sample_patients/patient_003_drug_interaction.json").read_text())
    graph = build_clinical_graph()
    config = {"configurable": {"thread_id": "review-test"}}
    graph.invoke({"session_id": "review-test", "patient_id": "pat-003", "raw_bundle": data,
                  "physician_notes": "", "ehr_committed": False}, config)
    return graph, config


@pytest.mark.parametrize("payload", [
    {"action": "unknown"}, {"action": "approve"},
    {"action": "approve", "acknowledge_critical_alerts": True, "notes": "  "},
    {"action": "edit", "edits": {"ehr_committed": True}},
    {"action": "edit", "edits": {}},
])
def test_direct_graph_resume_cannot_bypass_review(encounter, payload):
    graph, config = encounter
    graph.invoke(Command(resume=payload), config)
    snapshot = graph.get_state(config)
    assert snapshot.values["ehr_committed"] is False
    assert snapshot.tasks[0].interrupts[0].value["review_error"]
    graph.invoke(Command(resume={"action": "reject"}), config)
    assert graph.get_state(config).values["physician_status"] == "REJECTED"


def test_invalid_direct_resume_can_be_corrected(encounter):
    graph, config = encounter
    graph.invoke(Command(resume={"action": "approve"}), config)
    graph.invoke(Command(resume={"action": "approve", "acknowledge_critical_alerts": True,
                                "notes": "Critical alerts reviewed and addressed."}), config)
    assert graph.get_state(config).values["ehr_committed"] is True


def test_empty_resume_leaves_review_paused(encounter):
    graph, config = encounter
    graph.invoke(Command(resume={}), config)
    snapshot = graph.get_state(config)
    assert snapshot.values["ehr_committed"] is False
    assert snapshot.tasks[0].interrupts


def test_review_receipt_records_actual_changes_and_critical_alerts(encounter):
    graph, config = encounter
    graph.invoke(Command(resume={"action": "edit", "edits": {"assessment": "Physician revised."},
                                "acknowledge_critical_alerts": True,
                                "notes": "Critical alerts reviewed; medication plan revised."}), config)
    state = graph.get_state(config).values
    receipt = state["review_receipt"]
    assert state["ehr_committed"] is True
    assert receipt["action"] == "edit"
    assert receipt["changes"] == {"assessment": {"before": "Review needed.", "after": "Physician revised."}}
    assert receipt["critical_alerts_acknowledged"]
    assert receipt["reviewed_at"].endswith("+00:00")


def test_rejection_never_requires_alert_acknowledgement(encounter):
    graph, config = encounter
    graph.invoke(Command(resume={"action": "reject"}), config)
    state = graph.get_state(config).values
    assert state["ehr_committed"] is False
    assert state["review_receipt"]["action"] == "reject"
