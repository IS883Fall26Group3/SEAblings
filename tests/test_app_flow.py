"""Runs streamlit_app.py end to end with fake models: no API key, no quota."""
import json
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import core.llm
from core.llm.base import LLMError, LLMProvider

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
JD = "Data Analyst\n\nWe are looking for an analyst. Requirements: SQL. 2 years of experience."
RECIPE = "Chocolate chip cookies. Preheat the oven to 180C. Mix butter and sugar, then add flour. " * 5


class FakeProvider(LLMProvider):
    name = "fake"

    def complete(self, prompt, *, json_mode, temperature, what, schema=None):
        if not json_mode:
            return "Some advice."
        if what == "extracting skills from your CV":
            return json.dumps({"summary": "An analyst.", "years_experience": 4, "seniority": "mid",
                               "skills": ["SQL", "Python"], "job_titles": ["Analyst"]})
        job_id = prompt.split("### job_id: ")[1].split("\n")[0]
        return json.dumps({"items": [{"job_id": job_id, "required_skills": ["SQL"], "nice_to_have_skills": []}]})


class FakeEmbedder:
    name = "fake-embedder"
    skill_threshold, cos_low, cos_high, role_threshold = 0.8, 0.5, 0.9, 0.8
    error: Exception | None = None

    def embed(self, texts):
        if FakeEmbedder.error:
            raise FakeEmbedder.error
        return np.array([[1.0, 0.0]] * len(texts), dtype=np.float32)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(core.llm, "build_provider", lambda key, s: FakeProvider())
    monkeypatch.setattr(core.llm, "build_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(FakeEmbedder, "error", None)
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=30).run()


def analyze(app, jd=JD):
    app.text_area[1].set_value(jd)  # same request as the click, as with a quick click after typing
    next(b for b in app.button if b.label == "Analyze my match").click().run()
    assert not app.exception
    return app


def text(elements):
    return " ".join(e.value for e in elements)


def test_full_run_shows_ranked_results_and_unlocks_the_page(app):
    analyze(app)
    assert not app.session_state["running"]
    assert len(app.session_state["results"]) == 1
    assert app.text_area[1].value == JD  # kept, though typed just before the click
    assert "Your matches" in text(app.markdown)
    assert "Analyzed 1 job." in text(app.caption)
    assert not app.error


def test_advice_button_works_on_the_results_page(app):
    analyze(app)
    next(b for b in app.button if b.label == "Get advice for this role").click().run()
    assert not app.exception
    assert "Some advice." in text(app.markdown)
    assert len(app.session_state["results"]) == 1


def test_search_mode_ranks_the_jobs_it_finds(app):
    app.radio[0].set_value("Search live jobs").run()
    app.text_input[0].set_value("analyst")
    next(b for b in app.button if b.label == "Find matching jobs").click().run()
    assert not app.exception
    assert not app.session_state["running"]
    assert len(app.session_state["results"]) >= 3
    assert "Ranked" in text(app.caption)


def test_scoring_failure_keeps_earlier_stages_and_says_try_again(app):
    FakeEmbedder.error = LLMError("embed-model is temporarily overloaded. Pick another model.", try_other_model=True)
    analyze(app)
    assert not app.session_state["running"]
    assert app.error[0].value == "Scoring failed, try again."
    assert "results" not in app.session_state
    assert "SQL" in text(app.markdown)  # the job's requirements
    assert "The jobs we're looking at" in text(app.markdown)
    assert "An analyst." in text(app.markdown)  # the profile


def test_scoring_failure_on_daily_quota_does_not_say_try_again(app):
    FakeEmbedder.error = LLMError("The free daily quota for embed-model is used up. It resets in about 3h0s. "
                                  "Pick another model in the sidebar.", try_other_model=True)
    analyze(app)
    assert app.error[0].value == ("Scoring failed. The free daily quota for embed-model is used up. "
                                  "It resets in about 3h0s.")


def test_failed_run_does_not_bring_old_results_back(app):
    analyze(app)
    FakeEmbedder.error = LLMError("overloaded", try_other_model=True)
    analyze(app, JD + " Apply today.")  # a new posting, so its embeddings aren't cached yet
    assert app.error[0].value == "Scoring failed, try again."
    app.run()  # any later click reruns the page
    assert "results" not in app.session_state
    assert "Your matches" not in text(app.markdown)


def test_retry_after_scoring_failure_succeeds(app):
    FakeEmbedder.error = LLMError("overloaded", try_other_model=True)
    analyze(app)
    FakeEmbedder.error = None
    analyze(app)
    assert len(app.session_state["results"]) == 1
    assert not app.error


def test_invalid_job_description_warns_and_unlocks_the_page(app):
    analyze(app, RECIPE)
    assert not app.session_state["running"]
    assert "doesn't look like a job posting" in app.warning[0].value
    assert "profile" not in app.session_state


def test_empty_job_description_asks_for_one(app):
    analyze(app, "")
    assert not app.session_state["running"]
    assert app.sidebar.error[0].value == "Upload or paste at least one job description."
