"""Text embeddings for skill matching and profile similarity.
Each embedder carries its own thresholds: cosine values differ a lot between models."""
import logging
import threading
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
    """Small GGUF embedding model (bge-small, ~35 MB) run with llama-cpp-python. No API quota.
    Thresholds are starting points for bge-small: tune them with your evaluation set."""
    skill_threshold, cos_low, cos_high = 0.85, 0.55, 0.9

    def __init__(self, repo_id: str, filename: str, n_threads: int = 2):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise LLMError("Local embeddings need llama-cpp-python installed.") from e
        start = time.time()
        self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, embedding=True,
                                         n_ctx=512, n_batch=512, n_threads=n_threads, verbose=False)
        self.lock = threading.Lock()
        self.name = f"Local: {repo_id.split('/')[-1].removesuffix('-gguf')}"
        log.info("loaded %s in %.1fs", repo_id, time.time() - start)

    def embed(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            vectors = [self.llm.embed(t, truncate=True) for t in texts]
        return _normalize(np.array(vectors, dtype=np.float32))
