"""Deterministic Clinical Safety Guardrails.
Enforces hard medical invariants (drug-drug interactions, renal dosage cutoffs,
lab anomaly alerts) in code to guarantee patient safety before model generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from app.fhir import FHIRBundle


@dataclass(frozen=True)
class SafetyAlert:
    severity: str  # "CRITICAL", "WARNING", "INFO"
    category: str  # "DRUG_DRUG", "CONTRAINDICATION", "LAB_ANOMALY"
    title: str
    description: str
    action_required: str


# Common Drug-Drug Interaction Database (Standard Clinical Oncology & Internal Med Pairs)
KNOWN_INTERACTIONS: list[dict] = [
    {
        "drugs": ("lisinopril", "losartan"),
        "severity": "CRITICAL",
        "category": "DRUG_DRUG",
        "title": "Dual Renin-Angiotensin System (RAS) Blockade",
        "description": "Concurrent use of an ACE inhibitor (Lisinopril) and an ARB (Losartan) increases risks of severe hypotension, hyperkalemia, and acute renal failure.",
        "action_required": "Discontinue either the ACE inhibitor or the ARB.",
    },
    {
        "drugs": ("metformin", "ckd_severe"),
        "severity": "CRITICAL",
        "category": "CONTRAINDICATION",
        "title": "Metformin with Severe Renal Impairment",
        "description": "Metformin is strictly contraindicated when eGFR < 30 mL/min/1.73m2 due to high risk of fatal lactic acidosis.",
        "action_required": "Discontinue Metformin immediately and switch to an alternative antihyperglycemic.",
    },
    {
        "drugs": ("metoprolol", "diltiazem"),
        "severity": "CRITICAL",
        "category": "DRUG_DRUG",
        "title": "Severe Bradycardia Risk",
        "description": "Combined use of a beta blocker and non-dihydropyridine calcium channel blocker can precipitate heart block or severe bradycardia.",
        "action_required": "Monitor heart rate; avoid concurrent titration without cardiology consult.",
    },
    {
        "drugs": ("ibuprofen", "lisinopril"),
        "severity": "WARNING",
        "category": "DRUG_DRUG",
        "title": "NSAID + ACE Inhibitor Renal Toxicity",
        "description": "NSAIDs blunt the antihypertensive effect of ACE inhibitors and precipitate acute kidney injury.",
        "action_required": "Use acetaminophen for analgesia instead of NSAIDs.",
    },
]


def check_safety_invariants(bundle: FHIRBundle, proposed_medications: list[str] | None = None) -> list[SafetyAlert]:
    """Evaluates bundle vitals, labs, active medications, and proposed medications."""
    alerts: list[SafetyAlert] = []

    # 1. Collect all active and proposed medications (normalized lowercase)
    active_meds = [m.medication.lower() for m in bundle.medications]
    if proposed_medications:
        active_meds.extend([m.lower() for m in proposed_medications])

    # 2. Check Drug-Drug Interactions
    for interaction in KNOWN_INTERACTIONS:
        d1, d2 = interaction["drugs"]
        if d2 == "ckd_severe":
            continue  # Handled in lab checks below
        has_d1 = any(d1 in med for med in active_meds)
        has_d2 = any(d2 in med for med in active_meds)
        if has_d1 and has_d2:
            alerts.append(
                SafetyAlert(
                    severity=interaction["severity"],
                    category=interaction["category"],
                    title=interaction["title"],
                    description=interaction["description"],
                    action_required=interaction["action_required"],
                )
            )

    # 3. Check Lab Values & Renal Function
    egfr_obs = bundle.get_observation("eGFR") or bundle.get_observation("Estimated GFR")
    if egfr_obs:
        if egfr_obs.value < 30.0 and any("metformin" in med for med in active_meds):
            alerts.append(
                SafetyAlert(
                    severity="CRITICAL",
                    category="CONTRAINDICATION",
                    title="Metformin Contraindicated (eGFR < 30)",
                    description=f"Patient eGFR is {egfr_obs.value} mL/min. Metformin is contraindicated due to lactic acidosis risk.",
                    action_required="Discontinue Metformin.",
                )
            )
        elif egfr_obs.value < 45.0 and any("metformin" in med for med in active_meds):
            alerts.append(
                SafetyAlert(
                    severity="WARNING",
                    category="CONTRAINDICATION",
                    title="Metformin Renal Dose Reduction Advised",
                    description=f"Patient eGFR is {egfr_obs.value} mL/min. Maximum recommended dose is 1000 mg/day.",
                    action_required="Reduce Metformin dose and monitor renal function every 3 months.",
                )
            )

    potassium_obs = bundle.get_observation("Potassium")
    if potassium_obs and potassium_obs.value >= 5.5:
        alerts.append(
            SafetyAlert(
                severity="CRITICAL",
                category="LAB_ANOMALY",
                title=f"Hyperkalemia ({potassium_obs.value} mEq/L)",
                description="Potassium is critically elevated. ACE inhibitors, ARBs, and spironolactone increase hyperkalemia risk.",
                action_required="Recheck serum potassium, obtain ECG, and hold potassium-sparing medications.",
            )
        )

    # 4. Check Vital Signs: Hypertensive Crisis
    sbp_obs = bundle.get_observation("Systolic Blood Pressure") or bundle.get_observation("8480-6")
    dbp_obs = bundle.get_observation("Diastolic Blood Pressure") or bundle.get_observation("8462-4")
    if (sbp_obs and sbp_obs.value >= 180.0) or (dbp_obs and dbp_obs.value >= 120.0):
        alerts.append(
            SafetyAlert(
                severity="CRITICAL",
                category="LAB_ANOMALY",
                title="Hypertensive Urgency / Crisis",
                description=f"Blood pressure is {sbp_obs.value if sbp_obs else '?'}/{dbp_obs.value if dbp_obs else '?'} mmHg.",
                action_required="Evaluate immediately for target organ damage (headache, chest pain, shortness of breath).",
            )
        )

    return alerts
