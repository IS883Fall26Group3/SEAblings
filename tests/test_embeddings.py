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


def test_daily_embedding_quota_becomes_its_own_error(monkeypatch):
    """The app switches to the local embedder on this error, so it must not say to pick another model."""
    def spent(fn, what, model):
        raise LLMError("The free daily quota for gemini-embedding-001 is used up. It resets in about "
                       "4h43m31s. Pick another model in the sidebar.", try_other_model=True)
    monkeypatch.setattr(embeddings, "call_gemini", spent)
    emb = embeddings.GeminiEmbedder.__new__(embeddings.GeminiEmbedder)
    emb.client, emb.model = None, "gemini-embedding-001"
    with pytest.raises(embeddings.EmbeddingQuotaError) as e:
        emb.embed(["sql"])
    assert "4h43m31s" in str(e.value) and "sidebar" not in str(e.value)


def test_memo_embed_embeds_each_text_once():
    seen = []

    def embed(texts):
        seen.extend(texts)
        return np.eye(3)[:len(texts)]

    run = embeddings.memo_embed(embed)
    run(["a", "b"])
    out = run(["b", "c", "a"])
    assert seen == ["a", "b", "c"] and out.shape == (3, 3)
