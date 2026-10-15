"""Google Gemini provider. Every call has a timeout and fails with a clear message."""
import logging
import re
import time

from google import genai
from google.genai import errors, types

from .base import LLMError, LLMProvider

log = logging.getLogger("seablings.llm")


def _wait_time(msg: str) -> str | None:
    m = re.search(r"retry in ((?:\d+h)?(?:\d+m)?\d+)(?:\.\d+)?s", msg)
    return f"{m.group(1)}s" if m else None


def _explain(e: Exception, model: str) -> LLMError:
    code, msg = getattr(e, "code", None), str(e)
    if code == 400 and "API key" in msg:
        return LLMError("The Gemini API key is invalid. Check GEMINI_API_KEY in the app's Secrets.")
    if code in (401, 403):
        return LLMError("The Gemini API key was rejected. Create a new key in Google AI Studio.")
    if code == 404:
        return LLMError(f"Model '{model}' isn't available to this key. Edit GEMINI_MODELS in Secrets "
                        "to use model names listed in Google AI Studio.", try_other_model=True)
    if code == 429:
        wait = _wait_time(msg)
        if "PerDay" in msg:
            when = f" It resets in about {wait}." if wait else ""
            return LLMError(f"The free daily quota for {model} is used up.{when} "
                            "Pick another model in the sidebar.", try_other_model=True)
        return LLMError(f"{model} is rate-limited right now. Wait {wait or 'a minute'} or pick "
                        "another model.", try_other_model=True)
    if code in (500, 503):
        return LLMError(f"{model} is temporarily overloaded. Try again shortly or pick another model.",
                        try_other_model=True)
    return LLMError(f"Gemini request failed ({code}): {msg[:300]}")


def call_gemini(fn, what: str, model: str):
    """Run one Gemini call; retry once on a brief overload; never wait on a daily quota."""
    for attempt in range(2):
        start = time.time()
        try:
            result = fn()
            log.info("%s ok in %.1fs (model=%s)", what, time.time() - start, model)
            return result
        except errors.APIError as e:
            log.warning("%s failed in %.1fs (model=%s): %s", what, time.time() - start, model, e)
            transient = getattr(e, "code", None) == 503 or (
                getattr(e, "code", None) == 429 and "PerDay" not in str(e))
            if transient and attempt == 0:
                time.sleep(5)
                continue
            raise _explain(e, model) from e
        except Exception as e:  # timeouts and network errors
            log.warning("%s failed in %.1fs (model=%s): %r", what, time.time() - start, model, e)
            if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
                raise LLMError(f"{model} didn't respond in time while {what}. Try again or pick "
                               "another model.", try_other_model=True) from e
            raise LLMError(f"Couldn't reach Gemini while {what}: {e}", try_other_model=True) from e


def make_client(api_key: str, timeout_s: int) -> genai.Client:
    if not api_key:
        raise LLMError("GEMINI_API_KEY is missing. Add it in App settings > Secrets, then reboot.",
                       try_other_model=True)
    # The SDK has no timeout by default, so a slow call could hang forever. Set one.
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))


class GeminiProvider(LLMProvider):
    batch_jobs = True

    def __init__(self, api_key: str, model: str, timeout_s: int = 60):
        self.client = make_client(api_key, timeout_s)
        self.model = model
        self.name = model

    def complete(self, prompt: str, *, json_mode: bool, temperature: float, what: str,
                 schema: dict | None = None) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature, response_mime_type="application/json" if json_mode else None)
        resp = call_gemini(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt, config=config), what, self.model)
        return resp.text or ""
