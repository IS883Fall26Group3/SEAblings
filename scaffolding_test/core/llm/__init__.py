"""Model registry: which models are offered, how to build them, and Auto fallback."""
import importlib.util
import logging
from typing import Callable, TypeVar

from ..config import Settings
from .base import LLMError, LLMProvider
from .gemini import GeminiProvider

log = logging.getLogger("seablings.llm")
R = TypeVar("R")
AUTO = "auto"


def local_available() -> bool:
    return importlib.util.find_spec("llama_cpp") is not None


def model_options(s: Settings) -> dict[str, str]:
    """Sidebar choices as {key: label}. Keys look like 'gemini:<model>' or 'local:<repo id>'."""
    opts = {AUTO: "Auto (switch models when one is unavailable)"}
    opts.update({f"gemini:{m}": f"Gemini: {m}" for m in s.gemini_models})
    if local_available():
        name = s.local_llm_repo.split("/")[-1].removesuffix("-GGUF")
        opts[f"local:{s.local_llm_repo}"] = f"Offline: {name} (slower, no quota)"
    return opts


def auto_chain(s: Settings) -> list[str]:
    return [k for k in model_options(s) if k != AUTO]  # Gemini models first, local last


def build_provider(key: str, s: Settings) -> LLMProvider:
    kind, _, name = key.partition(":")
    if kind == "gemini":
        return GeminiProvider(s.gemini_api_key, name, s.llm_timeout_s)
    if kind == "local":
        from .local_llama import LlamaCppProvider
        return LlamaCppProvider(name, s.local_llm_file, s.local_n_ctx, s.local_threads,
                                max_cv_chars=s.local_max_cv_chars)
    raise ValueError(f"Unknown model key: {key}")


def build_embedder(s: Settings):
    """Local embeddings when available (no quota); Gemini embeddings otherwise or if loading fails."""
    from ..embeddings import GeminiEmbedder, LocalEmbedder
    use_local = s.embeddings == "local" or (s.embeddings == "auto" and local_available())
    if use_local:
        try:
            return LocalEmbedder(s.local_embed_repo, s.local_embed_file, s.local_threads)
        except Exception as e:
            if s.embeddings == "local":
                raise
            log.warning("local embeddings unavailable, using Gemini instead: %r", e)
    return GeminiEmbedder(s.gemini_api_key, s.gemini_embed_model, s.llm_timeout_s)


def with_fallback(chain: list[str], fn: Callable[[str], R],
                  on_switch: Callable[[str, LLMError], None] | None = None) -> tuple[R, str]:
    """Run fn(model_key) on each model in turn until one succeeds.
    Moves on only for errors where another model could help (quota, overload, timeout)."""
    last: LLMError | None = None
    for key in chain:
        try:
            return fn(key), key
        except LLMError as e:
            last = e
            if not e.try_other_model or key == chain[-1]:
                raise
            log.info("switching from %s: %s", key, e)
            if on_switch:
                on_switch(key, e)
    raise last or LLMError("No model is configured.")


__all__ = ["AUTO", "LLMError", "LLMProvider", "auto_chain", "build_embedder", "build_provider",
           "local_available", "model_options", "with_fallback"]
