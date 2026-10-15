"""Thin wrapper around the Gemini API: JSON generation and embeddings, with retries."""
import time
from typing import TypeVar

import numpy as np
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, api_key: str, model: str, embed_model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Add it to Streamlit secrets.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = embed_model

    def _call(self, fn, retries: int = 3):
        for attempt in range(retries):
            try:
                return fn()
            except errors.APIError as e:
                # 429 = free-tier rate limit; 5xx = transient. Back off and retry.
                code = getattr(e, "code", None)
                if code in (429, 500, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise

    def generate_json(self, prompt: str, schema: type[T], temperature: float = 0.1) -> T:
        """Ask for JSON matching a Pydantic schema and validate it."""
        full_prompt = (
            f"{prompt}\n\nReturn only JSON that validates against this JSON Schema:\n"
            f"{schema.model_json_schema()}"
        )
        config = types.GenerateContentConfig(
            temperature=temperature, response_mime_type="application/json"
        )
        last_err = None
        for _ in range(2):  # one repair attempt if the JSON is invalid
            resp = self._call(lambda: self.client.models.generate_content(
                model=self.model, contents=full_prompt, config=config))
            try:
                return schema.model_validate_json(resp.text)
            except ValidationError as e:
                last_err = e
        raise ValueError(f"Model returned invalid JSON: {last_err}")

    def generate_text(self, prompt: str, temperature: float = 0.4) -> str:
        config = types.GenerateContentConfig(temperature=temperature)
        resp = self._call(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt, config=config))
        return resp.text or ""

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        vectors = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            resp = self._call(lambda: self.client.models.embed_content(
                model=self.embed_model, contents=chunk,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")))
            vectors.extend(e.values for e in resp.embeddings)
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)
