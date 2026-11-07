"""Text embeddings for skill matching and profile similarity.
Each embedder carries its own thresholds: cosine values differ a lot between models."""
import logging
import re
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
from google.genai import types

from .llm.base import LLMError
from .llm.gemini import call_gemini, make_client

log = logging.getLogger("seablings.llm")


class EmbeddingQuotaError(LLMError):
    """Gemini's free daily embedding quota is used up. Picking another chat model doesn't help;
    the local embedding model does (streamlit_app.py switches to it)."""


def _normalize(arr: np.ndarray) -> np.ndarray:
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8, None)


class Embedder(ABC):
    name: str = "embedder"
    skill_threshold: float = 0.8     # cosine above which two skills count as the same
    cos_low: float = 0.5             # cosine range stretched to 0-1 for profile similarity
    cos_high: float = 0.9
    role_threshold: float = 0.8      # cosine above which a CV role counts toward a job (EXPERIENCE_FIT="roles")
    # cosine range stretched to 0-1 for the other PROFILE_SIMILARITY modes (document uses cos_low/high)
    sim_bands: dict[str, tuple[float, float]] = {}
    max_chars: int = 7_000           # CV and posting are cut to this before embedding (profile similarity)

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized vectors, shape (len(texts), dim)."""


def embed_with_wait(embed, texts: list[str], on_wait=None, max_wait_s: float = 60) -> np.ndarray:
    """Run embed(texts); on a per-minute rate limit, wait as long as Gemini asks (at most
    max_wait_s), tell the user through on_wait(seconds), and try once more.
    Embeddings have no other model to fall back to, so waiting beats failing."""
    try:
        return embed(texts)
    except LLMError as e:
        if e.retry_after_s is None:
            raise
        wait = min(max_wait_s, e.retry_after_s + 2)
    if on_wait:
        on_wait(round(wait))
    time.sleep(wait)
    try:
        return embed(texts)
    except LLMError as e:
        if e.retry_after_s is None:
            raise
        raise LLMError("Still over Gemini's per-minute limit. Wait a minute and try again, "
                       "or analyze fewer jobs.") from e


def memo_embed(embed):
    """Wrap an embed function so each distinct text is embedded once: retrieval and scoring embed the
    same skills, and Gemini's free tier allows about 100 texts a minute."""
    cache: dict[str, np.ndarray] = {}

    def run(texts: list[str]) -> np.ndarray:
        todo = list(dict.fromkeys(t for t in texts if t not in cache))
        if todo:
            cache.update(zip(todo, embed(todo)))
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, 1), dtype=np.float32)

    return run


class GeminiEmbedder(Embedder):
    # Gemini cosines are high and compressed: at 0.80 every skill matched (eval/README.md, step 4);
    # the document band is the p5-p95 of CV-job cosines over the 176 labeled pairs at 7,000 chars
    # (eval/doc_band.py, step 6 revisited; it was 0.731-0.795 at 3,000 chars).
    skill_threshold, cos_low, cos_high = 0.925, 0.709, 0.786
    # summary compares short LLM summaries, so max_chars doesn't change it. chunks embeds the
    # posting at max_chars and was measured at 3,000: re-run eval/similarity.py before using it.
    sim_bands = {"summary": (0.783, 0.859), "chunks": (0.747, 0.810)}
    # The model reads at most 2,048 tokens (~8,000 chars of English, fewer with numbers and bullets).
    max_chars = 7_000

    def __init__(self, api_key: str, model: str, timeout_s: int = 60):
        self.client = make_client(api_key, timeout_s)
        self.model = model
        self.name = model

    def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            try:
                resp = call_gemini(lambda: self.client.models.embed_content(
                    model=self.model, contents=chunk,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")),
                    "comparing skills", self.model)
            except LLMError as e:
                if "daily quota" in str(e):  # no other Gemini model helps: the app switches to local
                    when = re.search(r"resets in about ([0-9hms]+)", str(e))
                    raise EmbeddingQuotaError(
                        "Gemini's free daily quota for embeddings is used up"
                        + (f" (it resets in about {when[1]})." if when else ".")) from e
                raise
            vectors.extend(e.values for e in resp.embeddings)
        return _normalize(np.array(vectors, dtype=np.float32))


class LocalEmbedder(Embedder):
    """Small GGUF embedding model (bge-small, ~35 MB) run with llama-cpp-python. No API quota.
    Thresholds are starting points for bge-small: tune them with your evaluation set."""
    skill_threshold, cos_low, cos_high = 0.85, 0.586, 0.740  # document band: p5-p95 (step 6; was 0.55-0.9)
    sim_bands = {"summary": (0.560, 0.729), "chunks": (0.590, 0.697)}  # p5-p95, eval/README.md step 6
    # The bands above were measured at 3,000 chars. bge-small reads only 512 tokens (n_ctx below),
    # so a higher limit wouldn't add anything; a lower one would need the bands re-measured.
    max_chars = 3_000

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
