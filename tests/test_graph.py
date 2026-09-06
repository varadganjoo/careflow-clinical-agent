import json
from pathlib import Path
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.fhir import FHIRBundle
from app.graph import build_clinical_graph, EncounterState
from app.llm import SOAPNote

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_patients"


@pytest.fixture(autouse=True)
def mock_gemini_soap(monkeypatch):
    """Mocks LLM SOAP generation for deterministic graph testing without live API keys."""
    mock_note = SOAPNote(
        subjective="Patient presents for routine follow-up regarding chronic condition management.",
        objective="Blood pressure 152/94 mmHg. Laboratory evaluation reviewed.",
        assessment="Stage 2 Essential Hypertension and glycemic evaluation per ACC/AHA guidelines.",
        plan=[
            "Initiate dual antihypertensive therapy per ACC/AHA guidelines.",
            "Dietary sodium restriction and lifestyle modification.",
            "Follow-up metabolic panel in 3 months.",
        ],
        icd10_codes=["I10", "E11.9"],
        cpt_codes=["99214"],
    )
    monkeypatch.setattr(
        "app.graph.generate_structured",
        lambda prompt, sys_inst, schema, **kw: (mock_note, 120, 180),
    )


def test_graph_pauses_at_physician_review_gate():
    file = DATA_DIR / "patient_001_hypertension.json"
    bundle_data = json.loads(file.read_text(encoding="utf-8"))

    checkpointer = MemorySaver()
    graph = build_clinical_graph(checkpointer=checkpointer)

    session_id = "test-session-001"
    config = {"configurable": {"thread_id": session_id}}

    initial_state: EncounterState = {
        "session_id": session_id,
        "patient_id": "pat-001",
        "raw_bundle": bundle_data,
        "deid_text": "",
        "phi_vault": {},
        "safety_alerts": [],
        "clinical_findings": [],
        "soap_note": {},
        "physician_status": "PENDING_REVIEW",
        "physician_edits": {},
        "physician_notes": "",
        "ehr_committed": False,
    }

    # Run graph - should pause at physician_gate
    graph.invoke(initial_state, config=config)

    state = graph.get_state(config)
    assert len(state.tasks) > 0
    assert len(state.tasks[0].interrupts) > 0

    interrupt_data = state.tasks[0].interrupts[0].value
    assert interrupt_data["status"] == "AWAITING_PHYSICIAN_REVIEW"
    assert "soap_note" in interrupt_data
    assert "safety_alerts" in interrupt_data


def test_graph_resumes_with_physician_approval():
    file = DATA_DIR / "patient_001_hypertension.json"
    bundle_data = json.loads(file.read_text(encoding="utf-8"))

    checkpointer = MemorySaver()
    graph = build_clinical_graph(checkpointer=checkpointer)

    session_id = "test-session-002"
    config = {"configurable": {"thread_id": session_id}}

    initial_state: EncounterState = {
        "session_id": session_id,
        "patient_id": "pat-001",
        "raw_bundle": bundle_data,
        "deid_text": "",
        "phi_vault": {},
        "safety_alerts": [],
        "clinical_findings": [],
        "soap_note": {},
        "physician_status": "PENDING_REVIEW",
        "physician_edits": {},
        "physician_notes": "",
        "ehr_committed": False,
    }

    graph.invoke(initial_state, config=config)

    # Resume with approval
    graph.invoke(Command(resume={"action": "approve", "notes": "Approved by Dr. Test"}), config=config)

    final_state = graph.get_state(config)
    assert final_state.values["physician_status"] == "APPROVED"
    assert final_state.values["ehr_committed"] is True


def test_graph_resumes_with_physician_edit():
    file = DATA_DIR / "patient_001_hypertension.json"
    bundle_data = json.loads(file.read_text(encoding="utf-8"))

    checkpointer = MemorySaver()
    graph = build_clinical_graph(checkpointer=checkpointer)

    session_id = "test-session-003"
    config = {"configurable": {"thread_id": session_id}}

    initial_state: EncounterState = {
        "session_id": session_id,
        "patient_id": "pat-001",
        "raw_bundle": bundle_data,
        "deid_text": "",
        "phi_vault": {},
        "safety_alerts": [],
        "clinical_findings": [],
        "soap_note": {},
        "physician_status": "PENDING_REVIEW",
        "physician_edits": {},
        "physician_notes": "",
        "ehr_committed": False,
    }

    graph.invoke(initial_state, config=config)

    # Resume with edits
    edited_plan = ["Add Amlodipine 5mg daily", "Follow up in 4 weeks"]
    graph.invoke(
        Command(resume={"action": "edit", "edits": {"plan": edited_plan}, "notes": "Adjusted medication plan"}),
        config=config,
    )

    final_state = graph.get_state(config)
    assert final_state.values["physician_status"] == "EDITED"
    assert final_state.values["soap_note"]["plan"] == edited_plan
    assert final_state.values["ehr_committed"] is True
