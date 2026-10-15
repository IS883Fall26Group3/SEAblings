"""Offline provider: runs a small Hugging Face instruct model on CPU with transformers.
Weights download from the Hugging Face Hub on first use, then run without any API.
Expect lower extraction quality and slower responses than Gemini."""
import logging
import os
import time

from .base import LLMError, LLMProvider

log = logging.getLogger("seablings.llm")


class LocalHFProvider(LLMProvider):
    batch_jobs = False  # small models are more reliable one job at a time

    def __init__(self, model_id: str, dtype: str = "bfloat16", max_new_tokens: int = 512):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LLMError("Local model support isn't installed (needs torch and transformers).") from e
        start = time.time()
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # bfloat16 halves memory (about 1 GB for a 0.5B model), which matters on the free tier.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.name = f"Local: {model_id.split('/')[-1]}"
        log.info("loaded %s in %.1fs", model_id, time.time() - start)

    def complete(self, prompt: str, *, json_mode: bool, temperature: float, what: str) -> str:
        system = "You are a precise assistant."
        if json_mode:
            system += " Reply with valid JSON only."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        start = time.time()
        try:
            inputs = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                          do_sample=False, pad_token_id=self.tok.eos_token_id)
            prompt_len = inputs["input_ids"].shape[1]
            text = self.tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        except Exception as e:
            log.warning("%s failed on %s: %r", what, self.name, e)
            raise LLMError(f"The local model failed while {what}: {e}") from e
        log.info("%s ok in %.1fs (model=%s)", what, time.time() - start, self.name)
        return text
