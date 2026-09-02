"""Clinical Skills: ADA Standards of Care in Diabetes (2025/2026).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DiabetesAssessment:
    glycemic_status: str
    target_hba1c: float
    recommendations: list[str]


def assess_glycemic_control(hba1c: float, egfr: float | None = None, has_ascvd: bool = False, has_ckd: bool = False) -> DiabetesAssessment:
    """Evaluates diabetes management based on ADA Standards of Care."""
    target_hba1c = 7.0
    recommendations = []

    if hba1c < 5.7:
        status = "Normal"
    elif 5.7 <= hba1c <= 6.4:
        status = "Prediabetes"
        recommendations.append("Intensive lifestyle intervention (weight management, physical activity). Consider Metformin if BMI >= 35.")
    elif 6.5 <= hba1c <= 7.0:
        status = "Controlled Type 2 Diabetes"
        recommendations.append("Continue current regimen; maintain lifestyle modifications.")
    else:
        status = "Uncontrolled Type 2 Diabetes"
        recommendations.append("Step up antihyperglycemic therapy to achieve HbA1c < 7.0%.")

    # Add organ-protective guideline recommendations
    if has_ckd or (egfr is not None and egfr < 60):
        recommendations.append("Initiate SGLT2 inhibitor (Empagliflozin/Dapagliflozin) for renal and cardiovascular protection (if eGFR >= 20).")
    if has_ascvd:
        recommendations.append("Prioritize GLP-1 receptor agonist with proven CVD benefit (Semaglutide/Dulaglutide).")

    return DiabetesAssessment(
        glycemic_status=status,
        target_hba1c=target_hba1c,
        recommendations=recommendations,
    )
