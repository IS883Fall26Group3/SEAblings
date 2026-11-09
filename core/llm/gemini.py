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
        seconds = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
        return LLMError(f"{model} is rate-limited right now. Wait {wait or 'a minute'} or pick "
                        "another model.", try_other_model=True,
                        retry_after_s=float(seconds[1]) if seconds else 60.0)
    if code in (500, 503):
        return LLMError(f"{model} is temporarily overloaded. Try again shortly or pick another model.",
                        try_other_model=True)
    if code == 504:  # Google stopped the call at our deadline (llm_timeout_s): long prompts, slow model
        return LLMError(f"{model} took too long to answer. Try again or pick a faster model "
                        "(Flash Lite).", try_other_model=True)
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
    schema_unsupported = False

    def __init__(self, api_key: str, model: str, timeout_s: int = 60):
        self.client = make_client(api_key, timeout_s)
        self.model = model
        self.name = model

    def complete(self, prompt: str, *, json_mode: bool, temperature: float, what: str,
                 schema: dict | None = None) -> str:
        use_schema = json_mode and schema is not None and not self.schema_unsupported
        config = types.GenerateContentConfig(
            temperature=temperature, response_mime_type="application/json" if json_mode else None,
            # Constrains the reply to the expected JSON shape, so it can't add extra blocks.
            response_json_schema=schema if use_schema else None)
        try:
            resp = call_gemini(lambda: self.client.models.generate_content(
                model=self.model, contents=prompt, config=config), what, self.model)
        except LLMError as e:
            if not (use_schema and getattr(e.__cause__, "code", None) == 400):
                raise
            # Some models reject a JSON schema; fall back to plain JSON mode for this model.
            log.warning("%s rejected the JSON schema; retrying without it", self.model)
            self.schema_unsupported = True
            return self.complete(prompt, json_mode=json_mode, temperature=temperature, what=what)
        return resp.text or ""

    supports_tools = True

    def run_with_tools(self, system: str, prompt: str, tools: dict, what: str = "writing advice",
                       temperature: float = 0.4, max_steps: int = 6):
        """Tool calling as in the course's ReAct loop (Session 5): Gemini sees each Python function's
        name, arguments and docstring, decides which to call, we run it and send back the result,
        until it answers in text. The last step allows no more calls, so the loop always ends."""
        from ..schemas import ToolCall
        base = dict(system_instruction=system, temperature=temperature, tools=list(tools.values()),
                    # we run the functions ourselves, so each call can be checked and shown
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        made: list[ToolCall] = []
        for step in range(max_steps):
            last = step == max_steps - 1
            config = types.GenerateContentConfig(**base, tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE" if last else "AUTO")))
            resp = call_gemini(lambda: self.client.models.generate_content(
                model=self.model, contents=contents, config=config), what, self.model)
            calls = resp.function_calls or []
            if not calls:
                return resp.text or "", made
            contents.append(resp.candidates[0].content)  # the model's request, kept in the conversation
            parts = []
            for call in calls:
                args = dict(call.args or {})
                fn = tools.get(call.name)
                try:
                    result = fn(**args) if fn else {"error": f"There is no tool named {call.name}."}
                except Exception as e:  # a bad argument: tell the model instead of failing the advice
                    result = {"error": str(e)}
                log.info("tool %s(%s) called while %s", call.name, args, what)
                made.append(ToolCall(name=call.name, args=args))
                parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=parts))
        return "", made
