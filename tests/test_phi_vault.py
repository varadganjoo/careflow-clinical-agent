import pytest
from app.phi_vault import PHIVault


def test_deidentify_masks_hipaa_identifiers():
    sample_text = (
        "Patient: Eleanor Vance, MRN: MRN-92841. "
        "Contact: 555-234-5678, email: eleanor.vance@example.com. "
        "Address: 742 Evergreen Terrace, Springfield, IL. "
        "SSN: 987-65-4321."
    )

    res = PHIVault.deidentify(sample_text, patient_name="Eleanor Vance", patient_mrn="MRN-92841")

    assert "Eleanor Vance" not in res.redacted_text
    assert "MRN-92841" not in res.redacted_text
    assert "555-234-5678" not in res.redacted_text
    assert "eleanor.vance@example.com" not in res.redacted_text
    assert "987-65-4321" not in res.redacted_text
    assert "[PATIENT_NAME_1]" in res.redacted_text
    assert "[MRN_1]" in res.redacted_text
    assert "[PHONE_1]" in res.redacted_text
    assert "[EMAIL_1]" in res.redacted_text
    assert "[SSN_1]" in res.redacted_text


def test_rehydrate_restores_original_values():
    original = "Dr. Smith examined Eleanor Vance with MRN-92841."
    deid = PHIVault.deidentify(original, patient_name="Eleanor Vance", patient_mrn="MRN-92841")

    assert "[PATIENT_NAME_1]" in deid.redacted_text
    restored = PHIVault.rehydrate(deid.redacted_text, deid.vault)
    assert restored == original
