"""Provider plumbing tests: JSON parsing, schema hints, and Auto fallback. No API keys needed."""
import pytest

from core.llm import with_fallback
from core.llm.base import LLMError, LLMProvider, extract_json, schema_hint
from core.llm.gemini import _explain
from core.schemas import CandidateProfile, JobRequirementsBatch


class Scripted(LLMProvider):
    def __init__(self, replies):
        self.replies, self.name = list(replies), "scripted"

    def complete(self, prompt, *, json_mode, temperature, what, schema=None):
        return self.replies.pop(0)


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_schema_hint_is_compact_example():
    assert '"items": [{"job_id": "string"' in schema_hint(JobRequirementsBatch)


def test_generate_json_retries_once_on_bad_json():
    good = ('{"summary": "x", "years_experience": 3, "seniority": "mid", '
            '"skills": ["SQL"], "job_titles": []}')
    p = Scripted(["not json", good])
    assert p.generate_json("cv", CandidateProfile).skills == ["SQL"]


def test_fallback_moves_on_quota_but_not_on_bad_key():
    def fn(key):
        if key == "a":
            raise LLMError("quota", try_other_model=True)
        return "ok"
    assert with_fallback(["a", "b"], fn) == ("ok", "b")

    def bad_key(key):
        raise LLMError("invalid key")
    with pytest.raises(LLMError):
        with_fallback(["a", "b"], bad_key)


def test_daily_quota_message_names_model_and_wait():
    class E(Exception):
        code = 429
    e = E("Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel-FreeTier "
          "Please retry in 1h20m8.889822866s.")
    err = _explain(e, "gemini-flash-latest")
    assert err.try_other_model and "1h20m8s" in str(err) and "daily quota" in str(err)


def test_local_style_provider_extracts_one_job_at_a_time():
    from core.job_sources.sample import SampleSource
    from core.skill_extractor import extract_requirements
    jobs = SampleSource().search("analyst", "", 2)
    reply = '{"required_skills": ["SQL"], "nice_to_have_skills": [], "min_years_experience": null}'
    p = Scripted([reply, reply])
    p.batch_jobs = False
    out = extract_requirements(p, jobs)
    assert [j.requirements.job_id for j in out] == [j.id for j in jobs]
