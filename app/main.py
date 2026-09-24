"""CareFlow Clinical Agent - FastAPI Server & Physician Portal Backend.
Connects the Physician UI to the LangGraph durable state machine with HITL interrupt/resume.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.fhir import FHIRBundle
from app.graph import build_clinical_graph, EncounterState
from app.llm import DEFAULT_MODEL, GROQ_MODEL, MODEL_CHAIN, _groq_configured, describe_llm_error, stream_soap_synthesis
from app.phi_vault import PHIVault
from app.safety import check_safety_invariants
from skills.hypertension_acc_aha.rules import classify_blood_pressure
from skills.diabetes_ada_standards.rules import assess_glycemic_control

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample_patients"
STATIC_DIR = ROOT / "static"

logger = logging.getLogger("careflow.api")

app = FastAPI(
    title="CareFlow Clinical Agent",
    description="Enterprise Clinical Decision Support & EHR Documentation Platform",
    version="1.0.0",
)

# ponytail: in-memory checkpointer is per-instance; a recycled serverless instance loses paused encounters.
# Swap in a Postgres checkpointer if sign-offs must survive restarts.
checkpointer = MemorySaver()
graph = build_clinical_graph(checkpointer=checkpointer)

# In-memory store for active session thread_ids
active_sessions: dict[str, dict] = {}

SEVERITY_RANK = {"CRITICAL": 2, "WARNING": 1}


class ConsultRequest(BaseModel):
    patient_id: str = Field(..., max_length=32, description="Patient ID e.g. pat-001, pat-002, pat-003")
    notes: str = Field("", max_length=2000, description="Optional encounter notes or clinician observations")


class SOAPEdits(BaseModel):
    """Physician edits may only touch SOAP note fields; anything else is rejected."""

    model_config = ConfigDict(extra="forbid")

    subjective: str | None = Field(None, max_length=5000)
    objective: str | None = Field(None, max_length=5000)
    assessment: str | None = Field(None, max_length=5000)
    plan: list[str] | None = Field(None, max_length=30)
    icd10_codes: list[str] | None = Field(None, max_length=30)
    cpt_codes: list[str] | None = Field(None, max_length=30)


class ResumeRequest(BaseModel):
    action: Literal["approve", "edit", "reject"] = "approve"
    edits: SOAPEdits = Field(default_factory=SOAPEdits, description="Updated SOAP fields if action is 'edit'")
    notes: str = Field("", max_length=2000, description="Physician clinical notes or rationale")


def _get_patient_bundle(patient_id: str) -> dict:
    clean_id = patient_id.replace("pat-", "").replace("patient_", "").replace(".json", "")
    if clean_id.isdigit():
        for file in DATA_DIR.glob("*.json"):
            if file.name.startswith(f"patient_{clean_id}_"):
                return json.loads(file.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")


def _alert_dicts(bundle: FHIRBundle) -> list[dict]:
    return [
        {
            "severity": a.severity,
            "category": a.category,
            "title": a.title,
            "description": a.description,
            "action_required": a.action_required,
        }
        for a in check_safety_invariants(bundle)
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model": DEFAULT_MODEL,
        "models": MODEL_CHAIN,
        "backup": f"groq/{GROQ_MODEL}" if _groq_configured() else None,
        "active_sessions": len(active_sessions),
        "checkpointer": "MemorySaver",
    }


@app.get("/api/patients")
def list_patients() -> list[dict]:
    patients = []
    for file in sorted(DATA_DIR.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        patient = data["patient"]
        alerts = _alert_dicts(FHIRBundle(**data))
        vitals = [f"{o['display']}: {o['value']} {o['unit']}" for o in data.get("observations", []) if o.get("category") == "vital-signs"]
        patients.append({
            "id": patient["id"],
            "name": patient["name"],
            "mrn": patient["mrn"],
            "gender": patient["gender"],
            "birthDate": patient["birthDate"],
            "primary_condition": data["conditions"][0]["code"]["display"] if data.get("conditions") else "None",
            "vitals_summary": ", ".join(vitals),
            "alert_count": len(alerts),
            "max_severity": max((a["severity"] for a in alerts), key=lambda s: SEVERITY_RANK.get(s, 0), default=None),
            "file": file.name,
        })
    return patients


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str) -> dict:
    return _get_patient_bundle(patient_id)


@app.get("/api/patients/{patient_id}/safety")
def get_patient_safety(patient_id: str) -> dict:
    """Deterministic safety invariants for the patient's current meds and labs (no LLM call)."""
    bundle = FHIRBundle(**_get_patient_bundle(patient_id))
    return {"patient_id": bundle.patient.id, "alerts": _alert_dicts(bundle)}


def stream_encounter_soap(bundle_data: dict, patient_id: str) -> Iterator[str]:
    bundle = FHIRBundle(**bundle_data)

    text_repr = (
        f"Patient: {bundle.patient.name}, MRN: {bundle.patient.mrn}. "
        f"Conditions: {[c.code.display for c in bundle.conditions]}. "
        f"Meds: {[m.medication for m in bundle.medications]}."
    )
    res = PHIVault.deidentify(text_repr, patient_name=bundle.patient.name, patient_mrn=bundle.patient.mrn)

    findings = [f"Vitals: {bundle.get_vitals_summary()}"]
    sbp_obs = bundle.get_observation("Systolic Blood Pressure") or bundle.get_observation("8480-6")
    dbp_obs = bundle.get_observation("Diastolic Blood Pressure") or bundle.get_observation("8462-4")
    if sbp_obs and dbp_obs:
        bp_class = classify_blood_pressure(sbp_obs.value, dbp_obs.value)
        findings.append(f"ACC/AHA BP: {bp_class.category}. Rec: {bp_class.recommendation}")

    a1c_obs = bundle.get_observation("Hemoglobin A1c") or bundle.get_observation("4548-4")
    egfr_obs = bundle.get_observation("eGFR") or bundle.get_observation("Estimated GFR")
    if a1c_obs:
        diabetes_eval = assess_glycemic_control(a1c_obs.value, egfr_obs.value if egfr_obs else None)
        findings.append(f"ADA Diabetes: {diabetes_eval.glycemic_status}. Recs: {'; '.join(diabetes_eval.recommendations)}")

    alerts = _alert_dicts(bundle)
    yield _sse("alerts", alerts)
    yield _sse("start", {"patient_id": patient_id, "model": DEFAULT_MODEL})

    system_instruction = (
        "You are CareFlow Clinical AI, an expert medical scribe and clinical decision support system. "
        "Generate an authoritative clinical SOAP note in clean Markdown format with sections: "
        "### Subjective\n### Objective\n### Assessment\n### Plan\n### Recommended Billing Codes (ICD-10 & CPT).\n"
        "Ground your assessment strictly in the provided patient findings and clinical safety alerts."
    )
    prompt = (
        f"Patient Record (De-identified):\n{res.redacted_text}\n\n"
        f"Clinical Guidelines & Findings:\n" + "\n".join(findings) + "\n\n"
        f"Safety Alerts:\n" + "\n".join([f"[{a['severity']}] {a['title']}: {a['action_required']}" for a in alerts])
    )

    try:
        for chunk in stream_soap_synthesis(prompt=prompt, system_instruction=system_instruction):
            yield _sse("chunk", {"text": chunk})
    except Exception as exc:
        # Surface mid-stream model failures instead of silently truncating the note.
        logger.exception("SOAP stream failed")
        status, message, retryable = describe_llm_error(exc)
        yield _sse("error", {"message": message, "retryable": retryable})
        return

    yield _sse("complete", {"status": "complete", "patient_id": patient_id})


@app.get("/api/encounters/{patient_id}/stream")
def stream_encounter_endpoint(patient_id: str):
    """Streams clinical SOAP synthesis via Server-Sent Events (SSE)."""
    bundle_data = _get_patient_bundle(patient_id)  # 404 before the stream starts, not mid-stream
    return StreamingResponse(
        stream_encounter_soap(bundle_data, patient_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/consult")
def start_consultation(body: ConsultRequest) -> dict:
    bundle_data = _get_patient_bundle(body.patient_id)
    session_id = f"enc-{uuid.uuid4().hex[:8]}"

    initial_state: EncounterState = {
        "session_id": session_id,
        "patient_id": body.patient_id,
        "raw_bundle": bundle_data,
        "deid_text": "",
        "phi_vault": {},
        "safety_alerts": [],
        "clinical_findings": [],
        "soap_note": {},
        "physician_status": "PENDING_REVIEW",
        "physician_edits": {},
        "physician_notes": body.notes,
        "ehr_committed": False,
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        # Run graph until it pauses at the interrupt (physician_gate)
        graph.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Consultation run failed")
        status, message, _ = describe_llm_error(exc)
        raise HTTPException(status_code=status, detail=message) from exc

    state_snapshot = graph.get_state(config)
    is_interrupted = len(state_snapshot.tasks) > 0 and len(state_snapshot.tasks[0].interrupts) > 0
    interrupt_value = state_snapshot.tasks[0].interrupts[0].value if is_interrupted else None
    status = "AWAITING_PHYSICIAN_REVIEW" if is_interrupted else "COMPLETED"

    active_sessions[session_id] = {"patient_id": body.patient_id, "thread_id": session_id, "status": status}
    values = state_snapshot.values
    return {
        "session_id": session_id,
        "status": status,
        "is_interrupted": is_interrupted,
        "interrupt_payload": interrupt_value,
        "state": {
            "soap_note": values.get("soap_note"),
            "safety_alerts": values.get("safety_alerts"),
            "clinical_findings": values.get("clinical_findings"),
            "deid_text": values.get("deid_text"),
            "redaction_count": len(values.get("phi_vault") or {}),
        },
    }


@app.post("/api/consult/{session_id}/resume")
def resume_consultation(session_id: str, body: ResumeRequest) -> dict:
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=404,
            detail="This encounter is no longer open (it was already signed, or the server restarted). Generate a new draft.",
        )

    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph.get_state(config)

    if not state_snapshot.tasks or not state_snapshot.tasks[0].interrupts:
        raise HTTPException(status_code=409, detail="This encounter has already been signed or rejected.")

    edits = body.edits.model_dump(exclude_none=True)
    if body.action == "edit" and not edits:
        raise HTTPException(status_code=422, detail="Action 'edit' requires at least one edited SOAP field.")

    resume_payload = {
        "action": body.action,
        "edits": edits,
        "notes": body.notes,
    }

    graph.invoke(Command(resume=resume_payload), config=config)
    final_state = graph.get_state(config)
    active_sessions[session_id]["status"] = final_state.values.get("physician_status", "COMPLETED")

    return {
        "session_id": session_id,
        "status": final_state.values.get("physician_status"),
        "ehr_committed": final_state.values.get("ehr_committed", False),
        "final_soap": final_state.values.get("soap_note"),
        "physician_notes": final_state.values.get("physician_notes"),
    }


@app.get("/api/consult/{session_id}/status")
def get_consult_status(session_id: str) -> dict:
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph.get_state(config)
    return {
        "session_id": session_id,
        "next_nodes": list(state_snapshot.next),
        "values": state_snapshot.values,
    }
