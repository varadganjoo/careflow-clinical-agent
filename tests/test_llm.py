"""Unit tests for Gemini 3.8 Flash Client error handling and invariants.
Ensures that missing API keys or upstream model failures fail loudly with explicit errors
rather than silently returning canned/mocked SOAP notes.
"""

from unittest.mock import MagicMock
import pytest
from app.llm import SOAPNote, generate_structured
from app.graph import node_scribe_soap, EncounterState


def test_generate_structured_raises_when_api_key_missing(monkeypatch):
    """Invariant: When GEMINI_API_KEY is not configured, generate_structured must raise RuntimeError."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY environment variable is not configured"):
        generate_structured(
            prompt="Patient presents for evaluation",
            system_instruction="Generate SOAP note",
            schema=SOAPNote,
        )


def test_generate_structured_raises_when_gemini_api_fails(monkeypatch):
    """Invariant: When Gemini API raises an exception, generate_structured must fail loudly."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-valid-looking-key-12345")
    
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable: Model overloaded")
    monkeypatch.setattr("app.llm.get_gemini_client", lambda: mock_client)

    with pytest.raises(RuntimeError, match="Gemini API generation failed: 503 Service Unavailable"):
        generate_structured(
            prompt="Patient presents for evaluation",
            system_instruction="Generate SOAP note",
            schema=SOAPNote,
        )


def test_node_scribe_soap_propagates_error_on_llm_failure(monkeypatch):
    """Invariant: Scribe node does not swallow LLM errors or fall back to canned notes."""
    monkeypatch.setattr(
        "app.graph.generate_structured",
        MagicMock(side_effect=RuntimeError("Upstream LLM failure")),
    )

    state: EncounterState = {
        "session_id": "test-session",
        "patient_id": "pat-001",
        "raw_bundle": {},
        "deid_text": "De-identified patient summary",
        "phi_vault": {},
        "safety_alerts": [],
        "clinical_findings": ["BP: 152/94 mmHg"],
        "soap_note": {},
        "physician_status": "PENDING_REVIEW",
        "physician_edits": {},
        "physician_notes": "",
        "ehr_committed": False,
    }

    with pytest.raises(RuntimeError, match="Upstream LLM failure"):
        node_scribe_soap(state)
