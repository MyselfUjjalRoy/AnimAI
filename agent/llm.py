import json
import os
import sys
import logging
from typing import Any, List

log = logging.getLogger("animai.llm")

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file with your API key.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Prefer highest quality first, then faster/cheaper fallbacks.
PREFERRED_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_PREFERENCE_ORDER = [
    PREFERRED_MODEL,
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
]

def _normalize_model_name(name: str) -> str:
    if not name:
        return ""
    return name.replace("models/", "")


def _discover_generate_content_models() -> List[str]:
    """
    Discover models accessible by this API key and keep only Gemini models
    that support generateContent.
    """
    discovered: List[str] = []
    try:
        for model in client.models.list():
            model_name = _normalize_model_name(getattr(model, "name", ""))
            if not model_name.startswith("gemini"):
                continue

            supported = getattr(model, "supported_actions", None) or []
            if supported and "generateContent" not in supported:
                continue

            discovered.append(model_name)
    except Exception:
        # If listing fails (network/API limitations), fallback to known names.
        return list(dict.fromkeys(MODEL_PREFERENCE_ORDER))

    if not discovered:
        return list(dict.fromkeys(MODEL_PREFERENCE_ORDER))

    return list(dict.fromkeys(discovered))


def _build_model_candidates(preferred_model: str = None) -> List[str]:
    """Order candidates by preference but keep only discovered models when possible."""
    available = _discover_generate_content_models()
    available_set = set(available)

    ordered: List[str] = []

    effective_preferred = _normalize_model_name(preferred_model or PREFERRED_MODEL)
    preference_order = [
        effective_preferred,
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash-lite",
    ]

    # Always try the user-configured preferred model first, even if discovery
    # does not report it (some projects expose preview aliases only).
    preferred_normalized = effective_preferred
    if preferred_normalized:
        ordered.append(preferred_normalized)

    # If user asks for 3.0-flash, also try known aliases before fallbacks.
    if preferred_normalized == "gemini-3-flash-preview":
        for alias in ["gemini-3.0-flash-preview", "gemini-3.0-flash-exp"]:
            if alias not in ordered:
                ordered.append(alias)

    for model_name in preference_order:
        normalized = _normalize_model_name(model_name)
        if normalized in available_set and normalized not in ordered:
            ordered.append(normalized)

    # Add any remaining discovered Gemini models at the end.
    for model_name in available:
        if model_name not in ordered:
            ordered.append(model_name)

    return ordered or list(dict.fromkeys(preference_order))


def _extract_text_from_response(response) -> str:
    """Extract plain text from google-genai responses across SDK variants."""
    if response is None:
        return ""

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        try:
            return json.dumps(parsed, ensure_ascii=True)
        except Exception:
            return str(parsed)

    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    candidates = getattr(response, "candidates", None) or []
    collected_parts: List[str] = []

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue

        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                collected_parts.append(part_text)

    if collected_parts:
        # Parts can be streamed fragments of one JSON string/object; preserve exact sequence.
        return "".join(collected_parts).strip()

    return ""


def _extract_finish_reason(response) -> str:
    """Best-effort extraction of candidate finish reason across SDK variants."""
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "unknown"

        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return "unknown"

        # Enum-like objects may expose a `.name` property.
        return str(getattr(reason, "name", reason))
    except Exception:
        return "unknown"


def call_llm(
    prompt: str,
    max_tokens: int = 3000,
    response_mime_type: str = None,
    response_schema: Any = None,
    preferred_model: str = None,
    disable_thinking: bool = False,
) -> str:
    """
    Backward-compatible helper returning only text.
    """
    result = call_llm_detailed(
        prompt=prompt,
        max_tokens=max_tokens,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        preferred_model=preferred_model,
        disable_thinking=disable_thinking,
    )
    return result["text"]


def call_llm_detailed(
    prompt: str,
    max_tokens: int = 3000,
    response_mime_type: str = None,
    response_schema: Any = None,
    preferred_model: str = None,
    disable_thinking: bool = False,
) -> dict:
    """
    Sends a prompt to Gemini and returns text + diagnostics metadata.
    """
    last_error = None
    candidates = _build_model_candidates(preferred_model=preferred_model)

    for model_name in candidates:
        try:
            config_kwargs = {
                "max_output_tokens": max_tokens,
                "temperature": 0.2,
            }
            if disable_thinking:
                thinking_config_ctor = getattr(types, "ThinkingConfig", None)
                if thinking_config_ctor is not None:
                    config_kwargs["thinking_config"] = thinking_config_ctor(
                        thinking_budget=0
                    )
            if response_mime_type:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            text = _extract_text_from_response(response)
            finish_reason = _extract_finish_reason(response)
            log.info("model=%s finish_reason=%s chars=%d", model_name, finish_reason, len(text))
            if text:
                if response_mime_type == "application/json":
                    try:
                        json.loads(text)
                    except Exception as parse_err:
                        last_error = RuntimeError(
                            f"Model '{model_name}' returned invalid JSON: {parse_err}"
                        )
                        continue
                return {
                    "text": text,
                    "model": model_name,
                    "finish_reason": finish_reason,
                    "candidates_tried": candidates,
                    "error": None,
                }

            # Empty response payloads occasionally happen on quota/safety edges.
            last_error = RuntimeError(f"Model '{model_name}' returned no text content")
            continue
        except Exception as err:
            last_error = err
            continue

    error = RuntimeError(
        "Gemini request failed for all configured/discovered models. "
        "Check your API key, model access, and quota/billing in Google AI Studio. "
        f"Tried: {candidates}. Last error: {last_error}"
    )
    raise error