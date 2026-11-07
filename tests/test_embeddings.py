"""Rate-limit handling for embeddings. No API key needed: the embed function is scripted."""
import numpy as np
import pytest

import core.embeddings as embeddings
from core.embeddings import embed_with_wait
from core.llm.base import LLMError

LIMIT = LLMError("rate-limited", try_other_model=True, retry_after_s=40)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waited = []
    monkeypatch.setattr(embeddings.time, "sleep", waited.append)
    return waited


def scripted(*outcomes):
    calls = list(outcomes)

    def embed(texts):
        out = calls.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    return embed


def test_waits_tells_the_user_and_retries(no_sleep):
    told = []
    out = embed_with_wait(scripted(LIMIT, np.ones((1, 2))), ["cv"], told.append)
    assert out.shape == (1, 2)
    assert no_sleep == [42] and told == [42]  # Gemini's wait plus a 2 s margin


def test_wait_is_capped(no_sleep):
    long = LLMError("rate-limited", retry_after_s=300)
    embed_with_wait(scripted(long, np.ones((1, 2))), ["cv"])
    assert no_sleep == [60]


def test_second_limit_gives_a_clear_message():
    with pytest.raises(LLMError, match="Still over Gemini's per-minute limit"):
        embed_with_wait(scripted(LIMIT, LIMIT), ["cv"])


def test_other_errors_are_not_retried(no_sleep):
    with pytest.raises(LLMError, match="daily quota"):
        embed_with_wait(scripted(LLMError("daily quota used up")), ["cv"])
    assert no_sleep == []
