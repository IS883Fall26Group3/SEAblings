"""Text embeddings for skill matching and profile similarity.
Each embedder carries its own thresholds: cosine values differ a lot between models."""
import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from google.genai import types

from .llm.base import LLMError
from .llm.gemini import call_gemini, make_client

log = logging.getLogger("seablings.llm")


def _normalize(arr: np.ndarray) -> np.ndarray:
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)


class Embedder(ABC):
    name: str = "embedder"
    skill_threshold: float = 0.8     # cosine above which two skills count as the same
    cos_low: float = 0.5             # cosine range stretched to 0-1 for profile similarity
    cos_high: float = 0.9

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized vectors, shape (len(texts), dim)."""


class GeminiEmbedder(Embedder):
    skill_threshold, cos_low, cos_high = 0.80, 0.5, 0.9

    def __init__(self, api_key: str, model: str, timeout_s: int = 60):
        self.client = make_client(api_key, timeout_s)
        self.model = model
        self.name = model

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            resp = call_gemini(lambda: self.client.models.embed_content(
                model=self.model, contents=chunk,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")),
                "comparing skills", self.model)
            vectors.extend(e.values for e in resp.embeddings)
        return _normalize(np.array(vectors, dtype=np.float32))


class LocalEmbedder(Embedder):
    """Small sentence-embedding model (~90 MB) run on CPU. No API quota involved.
    Thresholds below are starting points: tune them with your evaluation set."""
    skill_threshold, cos_low, cos_high = 0.70, 0.2, 0.7

    def __init__(self, model_id: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise LLMError("Local embeddings need torch and transformers installed.") from e
        start = time.time()
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.name = f"Local: {model_id.split('/')[-1]}"
        log.info("loaded %s in %.1fs", model_id, time.time() - start)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
            with self.torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).numpy())
        return _normalize(np.vstack(out).astype(np.float32))
