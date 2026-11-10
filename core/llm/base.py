"""Provider interface shared by Gemini and local models, plus JSON helpers."""
import json
import logging
import types
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("seablings.llm")

# Who to tell when a call is tried again, so the user isn't left wondering why a step is slow.
# The app sets it (Streamlit); core/ only calls announce_retry. One value per thread, so visitors
# never see each other's messages.
_on_retry: ContextVar = ContextVar("on_retry", default=None)


def announce_retry(message: str) -> None:
    log.info("retry: %s", message)
    callback = _on_retry.get()
    if callback:
        try:
            callback(message)
        except Exception:  # showing the message must never break the call itself
            log.exception("couldn't show the retry message")


@contextmanager
def retry_notices(callback):
    """Inside this block, retry messages go to callback(message)."""
    token = _on_retry.set(callback)
    try:
        yield
    finally:
        _on_retry.reset(token)


class LLMError(RuntimeError):
    """User-facing error. try_other_model=True means another model may still work
    (quota, rate limit, temporary outage), so Auto mode moves to the next model.
    retry_after_s is set for a per-minute rate limit: waiting that long should be enough."""

    def __init__(self, message: str, try_other_model: bool = False, retry_after_s: float | None = None):
        super().__init__(message)
        self.try_other_model = try_other_model
        self.retry_after_s = retry_after_s


def _skeleton(tp):
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return {name: _skeleton(f.annotation) for name, f in tp.model_fields.items()}
    origin = get_origin(tp)
    if origin is list:
        return [_skeleton(get_args(tp)[0])]
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(tp) if a is not type(None)]
        return _skeleton(inner[0]) if inner else None
    return {str: "string", float: 0.0, int: 0, bool: False}.get(tp, "string")


def schema_hint(schema: type[BaseModel]) -> str:
    """A compact example of the expected JSON. Shorter than a full JSON Schema,
    which matters for small local models (and saves tokens on Gemini)."""
    return json.dumps(_skeleton(schema))


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply, tolerating code fences or extra text."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    # raw_decode reads the first complete object and ignores anything after it,
    # e.g. a second JSON block or a closing remark some models append.
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


class LLMProvider(ABC):
    name: str = "model"
    batch_jobs: bool = True             # can it extract requirements for many jobs in one call?
    max_cv_chars: int | None = None     # optional tighter CV limit (slow local models)

    @abstractmethod
    def complete(self, prompt: str, *, json_mode: bool, temperature: float, what: str,
                 schema: dict | None = None) -> str:
        """schema: JSON Schema of the expected reply; providers may use it to constrain output."""

    def generate_json(self, prompt: str, schema: type[T], what: str = "generating",
                      temperature: float = 0.1) -> T:
        full = (f"{prompt}\n\nRespond with only a JSON object in exactly this shape, "
                f"no commentary:\n{schema_hint(schema)}")
        last = None
        for attempt in range(2):  # one retry if the JSON is malformed
            if attempt:
                announce_retry(f"{self.name} sent an answer the app couldn't read while {what}. "
                               "Asking again…")
            raw = self.complete(full, json_mode=True, temperature=temperature, what=what,
                                schema=schema.model_json_schema())
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                last = e
        raise LLMError(f"{self.name} returned an unexpected format while {what}. "
                       f"Try again or pick another model. ({str(last)[:200]})", try_other_model=True)

    def generate_text(self, prompt: str, what: str = "writing advice", temperature: float = 0.4) -> str:
        return self.complete(prompt, json_mode=False, temperature=temperature, what=what)

    supports_tools: bool = False  # can the model call our Python functions (tool calling)?

    def run_with_tools(self, system: str, prompt: str, tools: dict, what: str = "writing advice",
                       temperature: float = 0.4, max_steps: int = 6):
        """Let the model call tools (name -> Python function) until it answers. Returns
        (answer text, list of core.schemas.ToolCall). Only providers with supports_tools have it."""
        raise NotImplementedError(f"{self.name} can't call tools")
