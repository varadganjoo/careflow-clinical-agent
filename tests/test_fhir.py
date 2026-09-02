import json
from pathlib import Path
import pytest
from app.fhir import FHIRBundle, Patient, Observation, Condition

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_patients"


def test_fhir_bundle_parsing():
    file = DATA_DIR / "patient_001_hypertension.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    bundle = FHIRBundle(**data)

    assert bundle.patient.name == "Eleanor Vance"
    assert bundle.patient.mrn == "MRN-92841"
    assert len(bundle.observations) == 4
    assert len(bundle.conditions) == 1
    assert bundle.conditions[0].code.code == "I10"


def test_observation_lookup():
    file = DATA_DIR / "patient_002_diabetes_ckd.json"
    bundle = FHIRBundle(**json.loads(file.read_text(encoding="utf-8")))

    a1c = bundle.get_observation("Hemoglobin A1c")
    assert a1c is not None
    assert a1c.value == 8.4
    assert a1c.unit == "%"

    egfr = bundle.get_observation("eGFR")
    assert egfr is not None
    assert egfr.value == 26.0


def test_vitals_summary():
    file = DATA_DIR / "patient_001_hypertension.json"
    bundle = FHIRBundle(**json.loads(file.read_text(encoding="utf-8")))

    vitals = bundle.get_vitals_summary()
    assert "Systolic Blood Pressure" in vitals
    assert "154.0 mmHg" in vitals["Systolic Blood Pressure"]
