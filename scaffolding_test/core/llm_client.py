"""Thin wrapper around the Gemini API: JSON generation and embeddings.
Every call has a timeout and fails with a clear, actionable message."""
import logging
import time
from typing import TypeVar

import numpy as np
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

log = logging.getLogger("seablings.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised with a message that can be shown directly to the user."""


def _explain(e: Exception, model: str) -> str:
    code = getattr(e, "code", None)
    msg = str(e)
    if code == 400 and "API key" in msg:
        return "The Gemini API key is invalid. Check GEMINI_API_KEY in the app's Secrets."
    if code in (401, 403):
        return "The Gemini API key was rejected. Create a new key in Google AI Studio and update Secrets."
    if code == 404:
        return (f"Model '{model}' isn't available to this key. Set GEMINI_MODEL in Secrets to a "
                "model listed in Google AI Studio.")
    if code == 429:
        return ("Gemini free-tier rate limit reached. Wait a minute and try again, or switch "
                "GEMINI_MODEL to a Flash-Lite model, which has higher free limits.")
    return f"Gemini request failed ({code}): {msg[:300]}"


class LLMClient:
    def __init__(self, api_key: str, model: str, embed_model: str, timeout_s: int = 60):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is missing. Add it in App settings > Secrets, then reboot the app.")
        # The SDK has no timeout by default, so a slow call can hang forever. Set one.
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
        self.model = model
        self.embed_model = embed_model

    def _call(self, fn, what: str, model: str, retries: int = 2):
        for attempt in range(retries):
            start = time.time()
            try:
                result = fn()
                log.info("%s ok in %.1fs (model=%s)", what, time.time() - start, model)
                return result
            except errors.APIError as e:
                log.warning("%s failed in %.1fs: %s", what, time.time() - start, e)
                if getattr(e, "code", None) in (429, 500, 503) and attempt < retries - 1:
                    time.sleep(10)
                    continue
                raise LLMError(_explain(e, model)) from e
            except Exception as e:  # timeouts and network errors
                log.warning("%s failed in %.1fs: %r", what, time.time() - start, e)
                if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower():
                    raise LLMError(f"Gemini didn't respond in time while {what}. Try again, or "
                                   "set GEMINI_MODEL to a faster Flash-Lite model.") from e
                raise LLMError(f"Couldn't reach Gemini while {what}: {e}") from e

    def generate_json(self, prompt: str, schema: type[T], what: str = "generating",
                      temperature: float = 0.1) -> T:
        full_prompt = (f"{prompt}\n\nReturn only JSON that validates against this JSON Schema:\n"
                       f"{schema.model_json_schema()}")
        config = types.GenerateContentConfig(temperature=temperature,
                                             response_mime_type="application/json")
        last_err = None
        for _ in range(2):  # one repair attempt if the JSON is invalid
            resp = self._call(lambda: self.client.models.generate_content(
                model=self.model, contents=full_prompt, config=config), what, self.model)
            try:
                return schema.model_validate_json(resp.text or "")
            except ValidationError as e:
                last_err = e
        raise LLMError(f"Gemini returned an unexpected format while {what}. Try again. ({last_err})")

    def generate_text(self, prompt: str, what: str = "writing advice", temperature: float = 0.4) -> str:
        config = types.GenerateContentConfig(temperature=temperature)
        resp = self._call(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt, config=config), what, self.model)
        return resp.text or ""

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        vectors = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            resp = self._call(lambda: self.client.models.embed_content(
                model=self.embed_model, contents=chunk,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")),
                "comparing skills", self.embed_model)
            vectors.extend(e.values for e in resp.embeddings)
        arr = np.array(vectors, dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)

    def ping(self) -> tuple[float, float]:
        """Quick connectivity check: returns (generate_seconds, embed_seconds)."""
        t0 = time.time()
        self._call(lambda: self.client.models.generate_content(
            model=self.model, contents="Reply with OK."), "testing the connection", self.model)
        t1 = time.time()
        self.embed(["test"])
        return t1 - t0, time.time() - t1
