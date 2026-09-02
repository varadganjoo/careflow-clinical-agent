"""Clinical Skills: ACC/AHA Hypertension Practice Guidelines (2024/2025 Updates).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BPClassification:
    category: str
    target_sbp: int
    target_dbp: int
    recommendation: str
    first_line_classes: tuple[str, ...]


def classify_blood_pressure(sbp: float, dbp: float, has_diabetes: bool = False, has_ckd: bool = False) -> BPClassification:
    """Classifies blood pressure based on 2024 ACC/AHA guidelines."""
    target_sbp = 130
    target_dbp = 80

    if sbp < 120 and dbp < 80:
        return BPClassification(
            category="Normal",
            target_sbp=target_sbp,
            target_dbp=target_dbp,
            recommendation="Promote optimal lifestyle habits. Recheck in 1 year.",
            first_line_classes=(),
        )
    elif 120 <= sbp <= 129 and dbp < 80:
        return BPClassification(
            category="Elevated",
            target_sbp=target_sbp,
            target_dbp=target_dbp,
            recommendation="Nonpharmacologic therapy (DASH diet, sodium reduction, exercise). Reassess in 3-6 months.",
            first_line_classes=(),
        )
    elif (130 <= sbp <= 139) or (80 <= dbp <= 89):
        return BPClassification(
            category="Stage 1 Hypertension",
            target_sbp=target_sbp,
            target_dbp=target_dbp,
            recommendation="Assess 10-year ASCVD risk. If >= 10% or comorbid diabetes/CKD, initiate 1 first-line antihypertensive.",
            first_line_classes=("ACEi/ARB", "Thiazide diuretic", "Dihydropyridine CCB"),
        )
    else:  # sbp >= 140 or dbp >= 90
        return BPClassification(
            category="Stage 2 Hypertension",
            target_sbp=target_sbp,
            target_dbp=target_dbp,
            recommendation="Initiate therapy with 2 first-line agents of different classes (e.g. ACEi/ARB + CCB or Thiazide).",
            first_line_classes=("ACEi/ARB", "Dihydropyridine CCB", "Thiazide diuretic"),
        )
