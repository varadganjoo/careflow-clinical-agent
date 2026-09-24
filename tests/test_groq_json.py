"""Groq's gpt-oss sometimes double-escapes newlines inside JSON strings; both forms must parse to real newlines."""

from app import llm
from app.llm import SOAPNote, generate_structured


def test_groq_json_newlines_normal_and_double_escaped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    raw = r'{"subjective": "line one\nline two", "objective": "o", "assessment": "1. X.\\n2. Y.", "plan": ["p"]}'
    monkeypatch.setattr(llm, "groq_chat", lambda *a, **k: (raw, 1, 1))

    note, _, _ = generate_structured("p", "s", SOAPNote)

    assert note.subjective == "line one\nline two"
    assert note.assessment == "1. X.\n2. Y."
