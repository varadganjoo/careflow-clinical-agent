"""FHIR v4 (Fast Healthcare Interoperability Resources) Pydantic Models.
Standard data exchange models matching Epic/Cerner EHR specifications.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class CodeableConcept(BaseModel):
    code: str = ""
    system: str = "http://hl7.org/fhir/sid/icd-10-cm"
    display: str = ""


class Patient(BaseModel):
    resourceType: Literal["Patient"] = "Patient"
    id: str
    mrn: str = ""
    name: str
    birthDate: str
    gender: str
    telecom: str = ""
    address: str = ""


class Observation(BaseModel):
    resourceType: Literal["Observation"] = "Observation"
    id: str
    status: str = "final"
    category: str = "vital-signs" # or laboratory
    code: str # LOINC or display e.g. "8480-6" (Systolic BP)
    display: str
    value: float
    unit: str
    reference_range: str = ""
    effectiveDateTime: str = ""


class Condition(BaseModel):
    resourceType: Literal["Condition"] = "Condition"
    id: str
    clinicalStatus: str = "active"
    code: CodeableConcept
    onsetDateTime: str = ""


class MedicationRequest(BaseModel):
    resourceType: Literal["MedicationRequest"] = "MedicationRequest"
    id: str
    status: str = "active"
    medication: str # e.g. "Lisinopril 20mg daily"
    dosage: str = ""
    authoredOn: str = ""


class AllergyIntolerance(BaseModel):
    resourceType: Literal["AllergyIntolerance"] = "AllergyIntolerance"
    id: str
    substance: str # e.g. "Penicillin"
    reaction: str = ""
    severity: str = "moderate" # mild, moderate, severe


class FHIRBundle(BaseModel):
    resourceType: Literal["Bundle"] = "Bundle"
    id: str
    patient: Patient
    observations: list[Observation] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[MedicationRequest] = Field(default_factory=list)
    allergies: list[AllergyIntolerance] = Field(default_factory=list)

    def get_observation(self, display_or_code: str) -> Observation | None:
        """Find the latest observation by display name or LOINC code."""
        for obs in reversed(self.observations):
            if display_or_code.lower() in obs.display.lower() or obs.code == display_or_code:
                return obs
        return None

    def get_vitals_summary(self) -> dict[str, str]:
        """Extract key vitals formatted for clinical view."""
        summary = {}
        for obs in self.observations:
            if obs.category == "vital-signs" or "blood pressure" in obs.display.lower():
                summary[obs.display] = f"{obs.value} {obs.unit}"
        return summary
