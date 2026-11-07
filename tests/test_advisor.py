"""Tool calling for advice (core/advisor.py, core/llm/gemini.py: run_with_tools). No API keys:
the Gemini client is replaced by a fake that first asks for a tool, then answers."""
from types import SimpleNamespace

from google.genai import types

from core.advisor import match_tools
from core.config import get_settings
from core.llm.base import LLMProvider
from core.llm.gemini import GeminiProvider
from core.matcher import score_jobs
from core.schemas import CandidateProfile, RequirementCheck
from core.skill_extractor import explain_match
from test_matcher import FakeEmbedder, fake_embed
from test_requirements import CV, job_with

PROFILE = CandidateProfile(summary="analyst", years_experience=3, seniority="mid", skills=["SQL"], job_titles=[])


def result():
    job = job_with(["SQL", "Power BI"])
    checks = {job.id: [RequirementCheck(requirement="SQL", status="met", quote="Wrote SQL queries",
                                        quote_verified=True),
                       RequirementCheck(requirement="Power BI", status="partial")]}
    return score_jobs(PROFILE, CV, [job], FakeEmbedder(), get_settings(), requirement_checks=checks)[0]


def test_score_tool_runs_the_apps_formula():
    r = result()
    tools = match_tools(r, CV, fake_embed, get_settings())
    out = tools["compute_match_score"](r.job.id)
    assert out["match_score"] == r.score
    assert out["requirements"] == [
        {"requirement": "SQL", "status": "met", "cv_quote": "Wrote SQL queries"},
        {"requirement": "Power BI", "status": "partial", "cv_quote": ""}]
    assert "error" in tools["compute_match_score"]("another-job")


def test_search_cv_returns_lines_copied_from_the_cv():
    tools = match_tools(result(), CV, fake_embed, get_settings())
    lines = tools["search_cv"]("inventory warehouses")["cv_lines"]
    assert lines[0] == "Managed inventory for three warehouses" and len(lines) == 3


class FakeModels:
    """Answers like Gemini: the first reply asks for compute_match_score, the second is text."""

    def __init__(self, job_id):
        self.job_id, self.requests = job_id, []

    def generate_content(self, model, contents, config):
        self.requests.append((list(contents), config))
        if len(self.requests) == 1:
            call = SimpleNamespace(name="compute_match_score", args={"job_id": self.job_id})
            content = types.Content(role="model", parts=[types.Part.from_function_call(
                name="compute_match_score", args={"job_id": self.job_id})])
            return SimpleNamespace(function_calls=[call], candidates=[SimpleNamespace(content=content)], text=None)
        return SimpleNamespace(function_calls=None, candidates=[], text="**Fit:** 62.5")


def gemini_with(models):
    llm = GeminiProvider.__new__(GeminiProvider)  # no API key: the client is the fake
    llm.client, llm.model, llm.name = SimpleNamespace(models=models), "fake-gemini", "fake-gemini"
    return llm


def test_the_model_calls_the_tool_and_gets_its_result_back():
    r = result()
    models = FakeModels(r.job.id)
    text, calls = explain_match(gemini_with(models), PROFILE, r, match_tools(r, CV, fake_embed, get_settings()))
    assert text == "**Fit:** 62.5"
    assert [(c.name, c.args) for c in calls] == [("compute_match_score", {"job_id": r.job.id})]
    first, config = models.requests[0]
    assert "Never suggest that the candidate claim" in config.system_instruction
    assert {t.__name__ for t in config.tools} == {"compute_match_score", "search_cv"}
    assert config.automatic_function_calling.disable  # our loop runs the tools, as in Session 5
    assert str(r.score) not in first[0].parts[0].text  # the prompt has no score: only the tool gives it
    sent_back = models.requests[1][0][-1].parts[0].function_response
    assert sent_back.name == "compute_match_score" and sent_back.response["result"]["match_score"] == r.score


class NoToolsLLM(LLMProvider):
    name = "offline"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, *, json_mode, temperature, what, schema=None):
        self.prompts.append(prompt)
        return "advice"


def test_models_without_tools_get_the_score_from_the_app():
    r = result()
    llm = NoToolsLLM()
    text, calls = explain_match(llm, PROFILE, r, match_tools(r, CV, fake_embed, get_settings()))
    assert text == "advice" and calls[0].args["by"] == "app"
    assert f'"match_score": {r.score}' in llm.prompts[0]
