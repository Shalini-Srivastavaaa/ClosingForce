"""LLM configuration and retry helpers for ClosingForce agents.

Provides a centralized way to get the Gemini LLM and a robust retry wrapper
specifically tuned for Gemini free-tier rate limits (429 RESOURCE_EXHAUSTED).
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Load environment variables once
load_dotenv()


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini Chat model.

    Hard default: gemini-2.5-flash-lite (best current free tier choice).
    Respects GEMINI_MODEL env var if set.
    """
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add it to your .env file "
            "(get one free at https://aistudio.google.com/app/apikey)"
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )


def _is_gemini_rate_limit_error(exception: Exception) -> bool:
    """Return True if this is a Gemini 429 / quota error we should retry."""
    error_str = str(exception).lower()

    # Check for common rate limit signals
    if any(phrase in error_str for phrase in ("429", "resource_exhausted", "rate limit", "quota")):
        return True

    # Try to detect google.genai ClientError with status 429
    try:
        from google.genai import errors as genai_errors
        if isinstance(exception, genai_errors.ClientError):
            # Some versions expose .status_code or the raw response
            status = getattr(exception, "status_code", None)
            if status == 429:
                return True
            # Fallback: check inside the exception args
            if "429" in str(exception.args):
                return True
    except Exception:
        pass

    return False


# Stronger retry policy for Gemini free tier
# 8 attempts with jittered exponential backoff (total wait can exceed 2 minutes)
_gemini_retry = retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential_jitter(multiplier=1.5, min=3, max=45, jitter=5),
    retry=retry_if_exception(_is_gemini_rate_limit_error),
    reraise=True,
)


def invoke_with_retry(runnable: Any, input_data: dict[str, Any], **kwargs) -> Any:
    """Invoke a LangChain runnable with aggressive retry for Gemini rate limits.

    This is the main entry point used by all agents.
    """
    model_name = getattr(runnable, "model", "unknown")

    @_gemini_retry
    def _invoke():
        try:
            return runnable.invoke(input_data, **kwargs)
        except Exception as e:
            if _is_gemini_rate_limit_error(e):
                print(f"[LLM Retry] Rate limit hit on model '{model_name}' — backing off and retrying... ({type(e).__name__})")
            raise

    return _invoke()