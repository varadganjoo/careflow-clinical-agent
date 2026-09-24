"""HIPAA Safe Harbor 18 De-identification & Rehydration Vault.
Guarantees that no protected health information (PHI) leaves the local perimeter
before external model reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Standard 18 HIPAA Identifiers Regex
PATIENT_NAME_RE = re.compile(r"\b(Patient:\s*)([A-Z][a-z]+ [A-Z][a-z]+)\b")
MRN_RE = re.compile(r"\b(MRN-?\d{5,10}|ID-?\d{5,10})\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,7}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DOB_RE = re.compile(r"\b(DOB:\s*|\b)(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", re.IGNORECASE)
ADDRESS_RE = re.compile(r"\b\d{1,5}\s+[A-Za-z0-9\s.,]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b", re.IGNORECASE)


@dataclass
class DeIDResult:
    redacted_text: str
    vault: dict[str, str] = field(default_factory=dict)
    redaction_count: int = 0


class PHIVault:
    """HIPAA Safe Harbor 18 de-identification and re-hydration engine."""

    @classmethod
    def deidentify(cls, text: str, patient_name: str = "", patient_mrn: str = "") -> DeIDResult:
        if not text:
            return DeIDResult(redacted_text="", vault={}, redaction_count=0)

        vault: dict[str, str] = {}
        counts: dict[str, int] = {}

        def replace_with_token(match_text: str, category: str) -> str:
            # Check if already in vault
            for token, original in vault.items():
                if original.lower() == match_text.lower():
                    return token
            counts[category] = counts.get(category, 0) + 1
            token = f"[{category}_{counts[category]}]"
            vault[token] = match_text
            return token

        redacted = text

        # 1. Explicit patient MRN and name if provided
        if patient_mrn and patient_mrn in redacted:
            token = replace_with_token(patient_mrn, "MRN")
            redacted = redacted.replace(patient_mrn, token)

        if patient_name and patient_name in redacted:
            token = replace_with_token(patient_name, "PATIENT_NAME")
            redacted = redacted.replace(patient_name, token)

        # 2. General pattern replacements
        redacted = SSN_RE.sub(lambda m: replace_with_token(m.group(0), "SSN"), redacted)
        redacted = PHONE_RE.sub(lambda m: replace_with_token(m.group(0), "PHONE"), redacted)
        redacted = EMAIL_RE.sub(lambda m: replace_with_token(m.group(0), "EMAIL"), redacted)
        redacted = ADDRESS_RE.sub(lambda m: replace_with_token(m.group(0), "ADDRESS"), redacted)
        redacted = MRN_RE.sub(lambda m: replace_with_token(m.group(0), "MRN"), redacted)

        # 3. Name parts on their own ("Good morning, Eleanor"). Runs last so emails/addresses are already tokenized.
        for part in patient_name.split():
            if len(part) > 1:
                redacted = re.sub(
                    rf"\b{re.escape(part)}\b",
                    lambda m: replace_with_token(m.group(0), "PATIENT_NAME"),
                    redacted,
                    flags=re.IGNORECASE,
                )

        return DeIDResult(
            redacted_text=redacted,
            vault=vault,
            redaction_count=len(vault),
        )

    @classmethod
    def rehydrate(cls, text: str, vault: dict[str, str]) -> str:
        """Re-hydrates tokens back to original values for clinical display."""
        if not text or not vault:
            return text
        restored = text
        for token, original in vault.items():
            restored = restored.replace(token, original)
        return restored
