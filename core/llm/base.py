"""Provider interface shared by Gemini and local models, plus JSON helpers."""
import json
import types
from abc import ABC, abstractmethod
from typing import TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """User-facing error. try_other_model=True means another model may still work
    (quota, rate limit, temporary outage), so Auto mode moves to the next model."""

    def __init__(self, message: str, try_other_model: bool = False):
        super().__init__(message)
        self.try_other_model = try_other_model


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
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


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
        for _ in range(2):  # one retry if the JSON is malformed
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
