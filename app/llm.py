"""Gemini Flash Clinical LLM Client & Structured Output Engine.
Uses the unified google-genai SDK with streaming and structured output validation.
"""

from __future__ import annotations

import os
import logging
from typing import TypeVar, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("careflow.llm")

# Primary model: Gemini 2.5 Flash / Gemini 3.8 Flash
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

T = TypeVar("T", bound=BaseModel)


class SOAPNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: list[str]
    icd10_codes: list[str] = []
    cpt_codes: list[str] = []


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from google import genai
        from google.genai import types

        # Retry 503 load-shedding only; 429 means the free-tier RPM quota is spent and retrying just burns more of it.
        retry = types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=10.0, http_status_codes=[503])
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))
    except Exception as exc:
        logger.warning(f"Failed to initialize google-genai client: {exc}")
        return None


def generate_structured(
    prompt: str,
    system_instruction: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
) -> tuple[T, int, int]:
    """Generates structured Pydantic output using Gemini 3.8 Flash.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not configured or the Gemini API call fails.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please configure GEMINI_API_KEY in .env to enable clinical SOAP synthesis."
        )

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2,
            ),
        )
        parsed: T = response.parsed
        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        return parsed, prompt_tokens, output_tokens
    except Exception as exc:
        logger.error(f"Gemini API generation failed: {exc}")
        raise RuntimeError(f"Gemini API generation failed: {exc}") from exc


def stream_soap_synthesis(
    prompt: str,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
):
    """Streams SOAP note generation chunks from Gemini.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not configured or generation fails.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please configure GEMINI_API_KEY in .env to enable clinical SOAP streaming."
        )

    try:
        from google.genai import types

        config = (
            types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )
            if system_instruction
            else types.GenerateContentConfig(temperature=0.2)
        )

        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        logger.error(f"Gemini streaming SOAP generation failed: {exc}")
        raise RuntimeError(f"Gemini streaming SOAP generation failed: {exc}") from exc

