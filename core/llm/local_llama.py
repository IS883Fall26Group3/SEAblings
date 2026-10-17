"""Offline provider: a small GGUF model from Hugging Face, run on CPU with llama-cpp-python
(same setup as the course's llama_demo). The file downloads on first use, then no API is involved.
Expect slower responses and lower extraction quality than Gemini."""
import logging
import threading
import time

from .base import LLMError, LLMProvider

log = logging.getLogger("seablings.llm")


class LlamaCppProvider(LLMProvider):
    batch_jobs = False  # small models are more reliable one job at a time

    def __init__(self, repo_id: str, filename: str, n_ctx: int = 8192, n_threads: int = 2,
                 max_tokens: int = 512, max_cv_chars: int = 8000):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise LLMError("Offline mode isn't installed (needs llama-cpp-python).") from e
        start = time.time()
        try:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx,
                                             n_threads=n_threads, verbose=False)
        except Exception as e:
            raise LLMError(f"Couldn't load the offline model {repo_id} ({filename}): {e}") from e
        self.lock = threading.Lock()  # one model shared by all visitors, one request at a time
        self.max_tokens = max_tokens
        self.max_cv_chars = max_cv_chars
        self.name = f"Offline: {repo_id.split('/')[-1].removesuffix('-GGUF')}"
        log.info("loaded %s in %.1fs", repo_id, time.time() - start)

    def complete(self, prompt: str, *, json_mode: bool, temperature: float, what: str,
                 schema: dict | None = None) -> str:
        messages = [{"role": "system", "content": "You are a precise assistant."
                     + (" Reply with valid JSON only." if json_mode else "")},
                    {"role": "user", "content": prompt}]
        start = time.time()
        try:
            with self.lock:
                reply = self._chat(messages, json_mode, temperature, schema)
        except Exception as e:
            log.warning("%s failed on %s: %r", what, self.name, e)
            raise LLMError(f"The offline model failed while {what}: {e}") from e
        log.info("%s ok in %.1fs (model=%s)", what, time.time() - start, self.name)
        return reply["choices"][0]["message"]["content"] or ""

    def _chat(self, messages, json_mode, temperature, schema):
        kwargs = dict(messages=messages, max_tokens=self.max_tokens, temperature=temperature)
        if not json_mode:
            return self.llm.create_chat_completion(**kwargs)
        try:
            # Grammar-constrained decoding: output is forced to match the schema, which makes
            # a 1B model far more reliable at structured extraction.
            return self.llm.create_chat_completion(
                **kwargs, response_format={"type": "json_object", "schema": schema})
        except Exception as e:  # schema-to-grammar conversion failed: fall back to plain JSON mode
            log.warning("schema grammar unavailable (%r); using plain JSON mode", e)
            return self.llm.create_chat_completion(**kwargs, response_format={"type": "json_object"})
