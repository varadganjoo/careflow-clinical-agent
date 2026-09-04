import json
from pathlib import Path
import pytest
from app.fhir import FHIRBundle
from app.safety import check_safety_invariants

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_patients"


def test_blocks_dual_ras_blockade():
    """Patient 003 has Lisinopril + Losartan (Dual ACEi + ARB)."""
    file = DATA_DIR / "patient_003_drug_interaction.json"
    bundle = FHIRBundle(**json.loads(file.read_text(encoding="utf-8")))

    alerts = check_safety_invariants(bundle)
    assert len(alerts) >= 1
    dual_ras = next((a for a in alerts if "Dual Renin-Angiotensin" in a.title), None)
    assert dual_ras is not None
    assert dual_ras.severity == "CRITICAL"
    assert "Discontinue either" in dual_ras.action_required


def test_blocks_metformin_in_severe_ckd():
    """Patient 002 has eGFR 26 (< 30) and is on Metformin."""
    file = DATA_DIR / "patient_002_diabetes_ckd.json"
    bundle = FHIRBundle(**json.loads(file.read_text(encoding="utf-8")))

    alerts = check_safety_invariants(bundle)
    metformin_alert = next((a for a in alerts if "Metformin Contraindicated" in a.title), None)
    assert metformin_alert is not None
    assert metformin_alert.severity == "CRITICAL"
    assert "lactic acidosis" in metformin_alert.description.lower()


def test_detects_hyperkalemia():
    """Patient 003 has potassium 5.6 mEq/L (>= 5.5)."""
    file = DATA_DIR / "patient_003_drug_interaction.json"
    bundle = FHIRBundle(**json.loads(file.read_text(encoding="utf-8")))

    alerts = check_safety_invariants(bundle)
    k_alert = next((a for a in alerts if "Hyperkalemia" in a.title), None)
    assert k_alert is not None
    assert k_alert.severity == "CRITICAL"
