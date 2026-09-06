"""CareFlow Model Context Protocol (MCP) Server.
Exposes FHIR resources, clinical tools, and consultation prompts
over STDIO and HTTP/SSE transports for external AI hosts (Claude Desktop, Cursor, EHRs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from mcp.server.mcpserver import MCPServer

from app.fhir import FHIRBundle
from app.safety import check_safety_invariants, KNOWN_INTERACTIONS
from app.llm import generate_structured, SOAPNote
from skills.hypertension_acc_aha.rules import classify_blood_pressure
from skills.diabetes_ada_standards.rules import assess_glycemic_control

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample_patients"
SKILLS_DIR = ROOT / "skills"

# Create MCP Server instance
mcp = MCPServer("careflow-clinical-agent")


def _load_patient_bundle(patient_id: str) -> FHIRBundle | None:
    """Helper to load synthetic patient bundle by ID."""
    clean_id = patient_id.replace("pat-", "").replace("patient_", "").replace(".json", "")
    for file in DATA_DIR.glob("*.json"):
        if clean_id in file.name:
            data = json.loads(file.read_text(encoding="utf-8"))
            return FHIRBundle(**data)
    # Default to first patient if not found
    for file in DATA_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        return FHIRBundle(**data)
    return None


# ==============================================================================
# MCP RESOURCES (Passive Read-Only Context)
# ==============================================================================

@mcp.resource("fhir://patients/{patient_id}")
def get_patient_record(patient_id: str) -> str:
    """Retrieves patient FHIR bundle (demographics, vitals, labs, medications, allergies)."""
    bundle = _load_patient_bundle(patient_id)
    if bundle is None:
        return json.dumps({"error": f"Patient {patient_id} not found."})
    return bundle.model_dump_json(indent=2)


@mcp.resource("fhir://guidelines/hypertension")
def get_hypertension_guidelines() -> str:
    """Retrieves ACC/AHA 2024 Clinical Practice Guidelines for Hypertension."""
    skill_file = SKILLS_DIR / "hypertension_acc_aha" / "SKILL.md"
    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    return "ACC/AHA Guidelines: Target BP < 130/80 mmHg. Stage 1: 130-139/80-89. Stage 2: >= 140/90."


@mcp.resource("fhir://guidelines/diabetes")
def get_diabetes_guidelines() -> str:
    """Retrieves ADA 2025/2026 Standards of Care in Diabetes."""
    skill_file = SKILLS_DIR / "diabetes_ada_standards" / "SKILL.md"
    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    return "ADA Guidelines: Target HbA1c < 7.0%. Metformin contraindicated if eGFR < 30 mL/min."


# ==============================================================================
# MCP TOOLS (Executable Clinical Functions)
# ==============================================================================

@mcp.tool()
def check_drug_interactions(drugs: list[str]) -> list[dict[str, Any]]:
    """Checks a list of medications against the deterministic clinical interaction database.
    
    Args:
        drugs: List of medication names (e.g. ["lisinopril", "losartan", "metformin"])
    """
    normalized = [d.lower() for d in drugs]
    detected = []
    for item in KNOWN_INTERACTIONS:
        d1, d2 = item["drugs"]
        if d2 == "ckd_severe":
            continue
        if any(d1 in med for med in normalized) and any(d2 in med for med in normalized):
            detected.append(item)
    return detected


@mcp.tool()
def query_clinical_guidelines(
    condition: str,
    sbp: float = 0.0,
    dbp: float = 0.0,
    hba1c: float = 0.0,
    egfr: float = 0.0,
) -> dict[str, Any]:
    """Queries clinical practice guidelines (ACC/AHA or ADA) with patient parameters.
    
    Args:
        condition: 'hypertension' or 'diabetes'
        sbp: Systolic blood pressure (mmHg)
        dbp: Diastolic blood pressure (mmHg)
        hba1c: Hemoglobin A1c percentage (%)
        egfr: Estimated GFR (mL/min/1.73m2)
    """
    cond = condition.lower()
    if "hyper" in cond or "bp" in cond or "blood pressure" in cond:
        res = classify_blood_pressure(sbp or 120.0, dbp or 80.0)
        return {
            "guideline": "ACC/AHA 2024",
            "category": res.category,
            "target": f"<{res.target_sbp}/{res.target_dbp} mmHg",
            "recommendation": res.recommendation,
            "first_line_classes": list(res.first_line_classes),
        }
    elif "diab" in cond or "a1c" in cond:
        res = assess_glycemic_control(hba1c or 6.0, egfr=egfr or None)
        return {
            "guideline": "ADA 2025/2026",
            "status": res.glycemic_status,
            "target": f"<{res.target_hba1c}%",
            "recommendations": res.recommendations,
        }
    return {"error": f"Unknown guideline condition: {condition}. Use 'hypertension' or 'diabetes'."}


@mcp.tool()
def evaluate_patient_safety(patient_id: str) -> dict[str, Any]:
    """Runs deterministic safety check across patient vitals, labs, and active medications.
    
    Args:
        patient_id: Identifier of the patient (e.g. 'pat-001', 'pat-002', 'pat-003')
    """
    bundle = _load_patient_bundle(patient_id)
    if bundle is None:
        return {"error": f"Patient {patient_id} not found."}
    alerts = check_safety_invariants(bundle)
    return {
        "patient_id": bundle.patient.id,
        "patient_name": bundle.patient.name,
        "alert_count": len(alerts),
        "alerts": [
            {
                "severity": a.severity,
                "category": a.category,
                "title": a.title,
                "description": a.description,
                "action_required": a.action_required,
            }
            for a in alerts
        ],
    }


@mcp.tool()
def generate_clinical_soap(
    patient_id: str,
    encounter_notes: str,
    proposed_plan: str = "",
) -> dict[str, Any]:
    """Generates a structured clinical SOAP note using Gemini 3.8 Flash with Pydantic adherence.
    
    Args:
        patient_id: Identifier of the patient
        encounter_notes: Subjective complaints, history of present illness, or transcript
        proposed_plan: Optional proposed treatment plan
    """
    bundle = _load_patient_bundle(patient_id)
    context = ""
    if bundle:
        context = f"Vitals: {bundle.get_vitals_summary()}. Conditions: {[c.code.display for c in bundle.conditions]}."
    
    prompt = f"Patient Context:\n{context}\n\nEncounter Notes:\n{encounter_notes}\n\nProposed Plan:\n{proposed_plan}"
    system = "Generate a structured SOAP note with ICD-10 and CPT coding suggestions."
    soap, prompt_tokens, output_tokens = generate_structured(prompt, system, SOAPNote)
    return {
        "soap_note": soap.model_dump(),
        "model": "gemini-3.8-flash",
        "tokens": {"prompt": prompt_tokens, "output": output_tokens},
    }


# ==============================================================================
# MCP PROMPTS (Reusable Consultation Templates)
# ==============================================================================

@mcp.prompt()
def clinical_consultation(patient_id: str, chief_complaint: str) -> str:
    """Generates standard clinical consultation prompt template."""
    return (
        f"You are a clinical decision support assistant reviewing Patient {patient_id}.\n"
        f"Chief Complaint: {chief_complaint}\n\n"
        f"Instructions:\n"
        f"1. Query the patient's record using resource fhir://patients/{patient_id}.\n"
        f"2. Evaluate safety invariants with tool evaluate_patient_safety.\n"
        f"3. Apply relevant clinical guidelines (hypertension or diabetes).\n"
        f"4. Propose an evidence-based clinical plan."
    )


if __name__ == "__main__":
    # Run via STDIO by default
    mcp.run()
