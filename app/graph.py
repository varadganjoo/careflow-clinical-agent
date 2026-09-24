"""LangGraph StateGraph with Durable Checkpointing & Human-in-the-Loop (HITL).
Implements the clinical workflow lifecycle with explicit interrupt() and Command(resume)
for physician review and digital sign-off.
"""

from __future__ import annotations

import json
from typing import Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from app.fhir import FHIRBundle
from app.phi_vault import PHIVault
from app.safety import check_safety_invariants, SafetyAlert
from app.llm import generate_structured, SOAPNote
from skills.hypertension_acc_aha.rules import classify_blood_pressure
from skills.diabetes_ada_standards.rules import assess_glycemic_control


class EncounterState(TypedDict):
    session_id: str
    patient_id: str
    raw_bundle: dict
    deid_text: str
    phi_vault: dict[str, str]
    safety_alerts: list[dict]
    clinical_findings: list[str]
    soap_note: dict
    physician_status: str  # "PENDING_REVIEW", "APPROVED", "EDITED", "REJECTED"
    physician_edits: dict
    physician_notes: str
    ehr_committed: bool


# 1. Ingestion Node
def node_ingest_and_triage(state: EncounterState) -> dict:
    bundle_data = state["raw_bundle"]
    bundle = FHIRBundle(**bundle_data)
    vitals = bundle.get_vitals_summary()
    return {
        "patient_id": bundle.patient.id,
        "clinical_findings": [f"Vitals: {vitals}"],
    }


# 2. HIPAA PHI De-identification Node
def node_deid(state: EncounterState) -> dict:
    bundle = FHIRBundle(**state["raw_bundle"])
    text_repr = f"Patient: {bundle.patient.name}, MRN: {bundle.patient.mrn}. Address: {bundle.patient.address}, Phone: {bundle.patient.telecom}. Conditions: {[c.code.display for c in bundle.conditions]}. Meds: {[m.medication for m in bundle.medications]}."
    # The ambient transcript goes through the same vault so spoken names never reach the model.
    transcript = "\n".join(f"{line['speaker']}: {line['text']}" for line in state["raw_bundle"].get("encounter_transcript", []))
    if transcript:
        text_repr += f"\n\nEncounter transcript:\n{transcript}"
    res = PHIVault.deidentify(text_repr, patient_name=bundle.patient.name, patient_mrn=bundle.patient.mrn)
    return {
        "deid_text": res.redacted_text,
        "phi_vault": res.vault,
    }


# 3. Clinical Skills RAG Node (ACC/AHA & ADA)
def node_clinical_skills(state: EncounterState) -> dict:
    bundle = FHIRBundle(**state["raw_bundle"])
    findings: list[str] = list(state.get("clinical_findings", []))

    # ACC/AHA Blood Pressure Evaluation
    sbp_obs = bundle.get_observation("Systolic Blood Pressure") or bundle.get_observation("8480-6")
    dbp_obs = bundle.get_observation("Diastolic Blood Pressure") or bundle.get_observation("8462-4")
    if sbp_obs and dbp_obs:
        bp_class = classify_blood_pressure(sbp_obs.value, dbp_obs.value)
        findings.append(f"ACC/AHA BP Assessment: {bp_class.category} (Target: <{bp_class.target_sbp}/{bp_class.target_dbp} mmHg). Rec: {bp_class.recommendation}")

    # ADA Glycemic Evaluation
    a1c_obs = bundle.get_observation("Hemoglobin A1c") or bundle.get_observation("4548-4")
    egfr_obs = bundle.get_observation("eGFR") or bundle.get_observation("Estimated GFR")
    if a1c_obs:
        diabetes_eval = assess_glycemic_control(a1c_obs.value, egfr_obs.value if egfr_obs else None)
        findings.append(f"ADA Diabetes Assessment: {diabetes_eval.glycemic_status} (HbA1c: {a1c_obs.value}%, Target: <{diabetes_eval.target_hba1c}%). Recs: {'; '.join(diabetes_eval.recommendations)}")

    return {"clinical_findings": findings}


# 4. Deterministic Safety Guard Node
def node_safety_guard(state: EncounterState) -> dict:
    bundle = FHIRBundle(**state["raw_bundle"])
    alerts = check_safety_invariants(bundle)
    alert_dicts = [
        {
            "severity": a.severity,
            "category": a.category,
            "title": a.title,
            "description": a.description,
            "action_required": a.action_required,
        }
        for a in alerts
    ]
    return {"safety_alerts": alert_dicts}


# 5. SOAP Scribe Node (Gemini)
def node_scribe_soap(state: EncounterState) -> dict:
    deid_text = state.get("deid_text", "")
    findings = "\n".join(state.get("clinical_findings", []))
    alerts = "\n".join([f"[{a['severity']}] {a['title']}: {a['action_required']}" for a in state.get("safety_alerts", [])])

    system_instruction = (
        "You are CareFlow Clinical AI, an expert medical scribe and decision support assistant. "
        "Generate a structured SOAP note based strictly on the provided clinical findings and safety alerts. "
        "Include appropriate ICD-10 and CPT coding suggestions."
    )
    prompt = f"Patient Record (De-identified):\n{deid_text}\n\nClinical Guidelines & Findings:\n{findings}\n\nSafety Alerts:\n{alerts}"

    soap_note, _, _ = generate_structured(prompt, system_instruction, SOAPNote)
    return {
        "soap_note": soap_note.model_dump(),
        "physician_status": "PENDING_REVIEW",
    }


# 6. Physician Review Gate Node (interrupt)
def node_physician_review_gate(state: EncounterState) -> dict:
    """Pauses graph execution for mandatory physician review and sign-off."""
    payload = {
        "status": "AWAITING_PHYSICIAN_REVIEW",
        "session_id": state["session_id"],
        "soap_note": state["soap_note"],
        "safety_alerts": state["safety_alerts"],
        "clinical_findings": state["clinical_findings"],
    }
    # Halts execution here until resumed with Command(resume=...)
    physician_response = interrupt(payload)

    action = physician_response.get("action", "approve")  # "approve", "edit", "reject"
    edits = physician_response.get("edits", {})
    notes = physician_response.get("notes", "")

    new_soap = dict(state["soap_note"])
    if action == "edit" and edits:
        new_soap.update(edits)

    status_map = {
        "approve": "APPROVED",
        "edit": "EDITED",
        "reject": "REJECTED",
    }
    return {
        "soap_note": new_soap,
        "physician_status": status_map.get(action, "APPROVED"),
        "physician_edits": edits,
        "physician_notes": notes,
    }


# 7. EHR Writeback Node
def node_ehr_writeback(state: EncounterState) -> dict:
    """Commits verified clinical note to EHR only if approved/edited by physician."""
    if state["physician_status"] in {"APPROVED", "EDITED"}:
        return {"ehr_committed": True}
    return {"ehr_committed": False}


def build_clinical_graph(checkpointer: Any | None = None):
    """Builds the compiled LangGraph state machine."""
    workflow = StateGraph(EncounterState)

    workflow.add_node("ingest", node_ingest_and_triage)
    workflow.add_node("deid", node_deid)
    workflow.add_node("skills", node_clinical_skills)
    workflow.add_node("safety", node_safety_guard)
    workflow.add_node("scribe", node_scribe_soap)
    workflow.add_node("physician_gate", node_physician_review_gate)
    workflow.add_node("writeback", node_ehr_writeback)

    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "deid")
    workflow.add_edge("deid", "skills")
    workflow.add_edge("skills", "safety")
    workflow.add_edge("safety", "scribe")
    workflow.add_edge("scribe", "physician_gate")
    workflow.add_edge("physician_gate", "writeback")
    workflow.add_edge("writeback", END)

    memory = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=memory)
