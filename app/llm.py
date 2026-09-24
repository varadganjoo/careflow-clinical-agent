"""Clinical LLM client: Gemini with model fallback, then Groq as a last-resort backup.
Uses the unified google-genai SDK with streaming and structured output validation.
"""

from __future__ import annotations

import json
import os
import logging
import urllib.error
import urllib.request
from typing import TypeVar, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("careflow.llm")

# Models tried in order. Free-tier quotas are per model, so falling back also extends the demo's daily budget.
MODEL_CHAIN = [
    m.strip()
    for m in os.getenv("GEMINI_MODELS", "gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash").split(",")
    if m.strip()
]
DEFAULT_MODEL = MODEL_CHAIN[0]

# Backup when every Gemini model is out of quota, overloaded, or unavailable.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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

        # Retry 503 load-shedding only; 429 means a quota is spent and retrying just burns more of it.
        retry = types.HttpRetryOptions(attempts=2, initial_delay=2.0, max_delay=4.0, http_status_codes=[503])
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))
    except Exception as exc:
        logger.warning(f"Failed to initialize google-genai client: {exc}")
        return None


def _should_fall_back(exc: Exception) -> bool:
    """Quota (429), overload (503), or unavailable model (404) on one model is worth trying the next."""
    text = str(exc)
    return any(code in text for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "404", "NOT_FOUND"))


def _groq_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def groq_chat(
    prompt: str, system_instruction: str = "", schema: Type[BaseModel] | None = None, temperature: float = 0.2
) -> tuple[str, int, int]:
    """One Groq chat completion (OpenAI-compatible API). Returns (content, prompt_tokens, output_tokens)."""
    messages = ([{"role": "system", "content": system_instruction}] if system_instruction else []) + [
        {"role": "user", "content": prompt}
    ]
    body: dict = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature}
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY'].strip()}",
            "Content-Type": "application/json",
            "User-Agent": "careflow/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{exc.code} {exc.read()[:300].decode('utf-8', 'replace')}") from exc
    usage = payload.get("usage") or {}
    return payload["choices"][0]["message"]["content"], usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def generate_structured(
    prompt: str,
    system_instruction: str,
    schema: Type[T],
    model: str | None = None,
) -> tuple[T, int, int]:
    """Generates structured Pydantic output: Gemini models in order, then Groq.

    Raises:
        RuntimeError: If no provider is configured or every provider fails.
    """
    errors: list[str] = []
    client = get_gemini_client()
    if client is None:
        errors.append(
            "gemini: GEMINI_API_KEY environment variable is not configured. "
            "Please configure GEMINI_API_KEY in .env to enable clinical SOAP synthesis."
        )
    else:
        from google.genai import types

        for model_name in [model] if model else MODEL_CHAIN:
            try:
                response = client.models.generate_content(
                    model=model_name,
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
                errors.append(f"{model_name}: {exc}")
                if not _should_fall_back(exc):
                    raise RuntimeError(f"Gemini API generation failed: {' | '.join(errors)}") from exc
                logger.warning(f"Gemini model {model_name} failed, trying next: {exc}")

    if _groq_configured():
        try:
            content, prompt_tokens, output_tokens = groq_chat(prompt, system_instruction, schema)
            # gpt-oss sometimes double-escapes newlines inside JSON strings (a literal backslash-n).
            parsed = schema.model_validate_json(content.replace("\\\\n", "\\n"))
            logger.info(f"Served by Groq backup ({GROQ_MODEL})")
            return parsed, prompt_tokens, output_tokens
        except Exception as exc:
            errors.append(f"groq {GROQ_MODEL}: {exc}")

    logger.error(f"Generation failed on every provider: {errors}")
    raise RuntimeError(f"Gemini API generation failed: {' | '.join(errors)}")


def stream_soap_synthesis(
    prompt: str,
    system_instruction: str = "",
    model: str | None = None,
):
    """Streams SOAP text: Gemini models in order until text starts flowing, then Groq (sent as one chunk).

    Raises:
        RuntimeError: If no provider is configured or generation fails.
    """
    errors: list[str] = []
    client = get_gemini_client()
    if client is None:
        errors.append(
            "gemini: GEMINI_API_KEY environment variable is not configured. "
            "Please configure GEMINI_API_KEY in .env to enable clinical SOAP streaming."
        )
    else:
        from google.genai import types

        config = (
            types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
            if system_instruction
            else types.GenerateContentConfig(temperature=0.2)
        )
        for model_name in [model] if model else MODEL_CHAIN:
            started = False
            try:
                for chunk in client.models.generate_content_stream(model=model_name, contents=prompt, config=config):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                # Once text has reached the caller, switching models would splice two different notes together.
                if started or not _should_fall_back(exc):
                    raise RuntimeError(f"Gemini streaming SOAP generation failed: {' | '.join(errors)}") from exc
                logger.warning(f"Gemini model {model_name} failed before streaming, trying next: {exc}")

    if _groq_configured():
        try:
            text, _, _ = groq_chat(prompt, system_instruction)
        except Exception as exc:
            errors.append(f"groq {GROQ_MODEL}: {exc}")
        else:
            yield text
            return

    logger.error(f"SOAP streaming failed on every provider: {errors}")
    raise RuntimeError(f"Gemini streaming SOAP generation failed: {' | '.join(errors)}")


def describe_llm_error(exc: Exception) -> tuple[int, str, bool]:
    """Maps a model failure to (HTTP status, message a demo visitor can act on, whether retrying soon can help)."""
    text = str(exc)
    if "PerDay" in text and ("503" in text or "UNAVAILABLE" in text):
        return 503, "The primary model's daily quota is used up and the fallback models are overloaded. Try again in a few minutes.", True
    if "PerDay" in text:
        # Retrying will not help until the free-tier daily quota resets (midnight Pacific).
        return 429, "The demo's daily Gemini free-tier quota is used up. It resets at midnight Pacific time.", False
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return 429, "Model rate limit reached. Wait about a minute, then retry.", True
    if "503" in text or "UNAVAILABLE" in text:
        return 503, "The models are overloaded right now (already retried with backoff). Try again shortly.", True
    if "not configured" in text:
        return 500, "Model API keys are not configured on this deployment.", False
    return 502, f"Model call failed: {text[:200]}", False
