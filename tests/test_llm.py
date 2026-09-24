"""Unit tests for Gemini 3.6 Flash Client error handling and invariants.
Ensures that missing API keys or upstream model failures fail loudly with explicit errors
rather than silently returning canned/mocked SOAP notes.
"""

from unittest.mock import MagicMock
import pytest
from app.llm import SOAPNote, generate_structured, stream_soap_synthesis
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

    with pytest.raises(RuntimeError, match="Gemini API generation failed: .*503 Service Unavailable"):
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


# --- Model fallback chain ---

from app import llm as llm_module
from app.llm import SOAPNote


def _client_failing(monkeypatch, failures: dict):
    """Mock client whose models fail with the given error text; others succeed."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-valid-looking-key-12345")
    calls: list[str] = []

    def generate(model, **kwargs):
        calls.append(model)
        if model in failures:
            raise Exception(failures[model])
        response = MagicMock()
        response.parsed = f"note from {model}"
        response.usage_metadata = None
        return response

    def stream(model, **kwargs):
        calls.append(model)
        if model in failures:
            raise Exception(failures[model])
        return iter([MagicMock(text=f"chunk from {model}")])

    client = MagicMock()
    client.models.generate_content.side_effect = generate
    client.models.generate_content_stream.side_effect = stream
    monkeypatch.setattr("app.llm.get_gemini_client", lambda: client)
    return calls


def test_default_chain_is_36_37_38():
    assert llm_module.MODEL_CHAIN == ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"]


@pytest.mark.parametrize("error", ["429 RESOURCE_EXHAUSTED", "503 UNAVAILABLE", "404 NOT_FOUND model retired"])
def test_structured_falls_back_to_next_model(monkeypatch, error):
    calls = _client_failing(monkeypatch, {"gemini-3.6-flash": error})
    result = generate_structured(prompt="p", system_instruction="s", schema=SOAPNote)
    assert result[0] == "note from gemini-3.7-flash"
    assert calls == ["gemini-3.6-flash", "gemini-3.7-flash"]


def test_structured_does_not_fall_back_on_bad_request(monkeypatch):
    calls = _client_failing(monkeypatch, {"gemini-3.6-flash": "400 INVALID_ARGUMENT schema rejected"})
    with pytest.raises(RuntimeError, match="400 INVALID_ARGUMENT"):
        generate_structured(prompt="p", system_instruction="s", schema=SOAPNote)
    assert calls == ["gemini-3.6-flash"]


def test_all_models_exhausted_reports_the_daily_quota(monkeypatch):
    _client_failing(monkeypatch, {
        "gemini-3.6-flash": "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        "gemini-3.7-flash": "404 NOT_FOUND",
        "gemini-3.8-flash": "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier",
    })
    with pytest.raises(RuntimeError) as err:
        generate_structured(prompt="p", system_instruction="s", schema=SOAPNote)
    assert all(m in str(err.value) for m in llm_module.MODEL_CHAIN)
    status, message, retryable = llm_module.describe_llm_error(err.value)
    assert (status, retryable) == (429, False)
    assert "daily" in message


def test_stream_falls_back_before_first_chunk(monkeypatch):
    calls = _client_failing(monkeypatch, {"gemini-3.6-flash": "503 UNAVAILABLE"})
    assert list(stream_soap_synthesis(prompt="p", system_instruction="s")) == ["chunk from gemini-3.7-flash"]
    assert calls == ["gemini-3.6-flash", "gemini-3.7-flash"]


def test_stream_does_not_switch_models_after_text_started(monkeypatch):
    calls = _client_failing(monkeypatch, {})

    def half_then_fail(model, **kwargs):
        calls.append(model)
        yield MagicMock(text="partial ")
        raise Exception("503 UNAVAILABLE")

    llm_module.get_gemini_client().models.generate_content_stream.side_effect = half_then_fail
    received = []
    with pytest.raises(RuntimeError):
        for chunk in stream_soap_synthesis(prompt="p", system_instruction="s"):
            received.append(chunk)
    assert received == ["partial "]
    assert calls == ["gemini-3.6-flash"]


def test_daily_quota_plus_overloaded_fallbacks_is_retryable():
    status, message, retryable = llm_module.describe_llm_error(RuntimeError(
        "gemini-3.6-flash: 429 GenerateRequestsPerDayPerProjectPerModel-FreeTier | gemini-3.7-flash: 503 UNAVAILABLE"
    ))
    assert (status, retryable) == (503, True)
    assert "fallback models are overloaded" in message
