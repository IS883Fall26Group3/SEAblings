"""Runs streamlit_app.py end to end with fake models: no API key, no quota."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import core.job_sources
import core.llm
from core.job_sources.sample import SampleSource
from core.llm.base import LLMError, LLMProvider

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
JD = "Data Analyst\n\nWe are looking for an analyst. Requirements: SQL. 2 years of experience."
CV = (Path(APP).parent / "data" / "sample" / "sample_cv.txt").read_text(encoding="utf-8")
RECIPE = "Chocolate chip cookies. Preheat the oven to 180C. Mix butter and sugar, then add flour. " * 5


class FakeProvider(LLMProvider):
    name = "fake"
    cv_type = "cv"

    def complete(self, prompt, *, json_mode, temperature, what, schema=None):
        if what == "extracting skills from your CV":
            return json.dumps({"summary": "An analyst.", "years_experience": 4, "seniority": "mid",
                               "skills": ["SQL", "Python"], "job_titles": ["Analyst"],
                               "document_type": FakeProvider.cv_type})
        ids = re.findall(r"job_id: (\S+)", prompt)
        if what == "judging your relevant experience":
            items = [{"job_id": i, "field_match": "same", "work_match": "same"} for i in ids]
        elif what == "checking each requirement against your CV":
            items = [{"job_id": i, "labels": [{"requirement": "SQL", "status": "met"}]} for i in ids]
        else:
            items = [{"job_id": i, "required_skills": ["SQL"], "nice_to_have_skills": []} for i in ids]
        return json.dumps({"items": items})


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
    monkeypatch.setattr(FakeProvider, "cv_type", "cv")
    monkeypatch.setattr(core.job_sources, "get_job_source", lambda s: SampleSource())  # whatever secrets say
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=30).run()


def click(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception
    return app


def analyze(app, jd=JD):
    app.text_area[1].set_value(jd)  # same request as the click, as with a quick click after typing
    return click(app, "Analyze my match")


def text(elements):
    return " ".join(e.value for e in elements)


def test_inputs_on_the_main_page_and_model_and_limitations_in_the_sidebar(app):
    assert [h.value for h in app.main.subheader] == ["1. Your CV", "2. Jobs to match"]
    assert [h.value for h in app.sidebar.subheader] == ["AI model", "Limitations"]
    assert "See how well your CV fits the job" in text(app.markdown)


def test_full_run_shows_ranked_results_and_unlocks_the_page(app):
    analyze(app)
    assert not app.session_state["running"]
    assert len(app.session_state["results"]) == 1
    assert app.text_area[1].value == JD  # kept, though typed just before the click
    assert "Your matches" in text(app.markdown)
    assert "Analyzed 1 job." in text(app.caption)
    assert not app.error


def test_sample_postings_are_used_without_any_job_description(app):
    analyze(app, "")
    assert len(app.session_state["results"]) >= 3


def test_search_mode_ranks_the_jobs_it_finds(app):
    app.radio[0].set_value("Search live jobs").run()
    app.text_input[0].set_value("analyst")
    click(app, "Find matching jobs")
    assert not app.session_state["running"]
    assert len(app.session_state["results"]) >= 3
    assert "Ranked" in text(app.caption)


def test_empty_role_asks_for_one_under_the_button(app):
    app.radio[0].set_value("Search live jobs").run()
    click(app, "Find matching jobs")
    assert app.main.error[0].value == "Enter the role you're looking for."
    assert not app.session_state["running"]


def test_junk_role_warns_like_the_other_inputs_before_any_model_call(app):
    app.radio[0].set_value("Search live jobs").run()
    app.text_input[0].set_value("???")
    click(app, "Find matching jobs")
    assert app.warning[0].value == "**CV:** Looks good.  \n**Role:** Enter a job role, such as Data analyst."
    assert "profile" not in app.session_state
    assert not app.session_state["running"]


def test_search_runs_without_a_location_and_explains_the_default(app):
    app.radio[0].set_value("Search live jobs").run()
    location = app.text_input[1]
    assert location.label == "Location (optional)"
    assert location.help == "If left empty, we search for this role across the US."
    app.text_input[0].set_value("teddy bear")
    click(app, "Find matching jobs")
    assert not app.warning
    assert app.session_state["results"]


def test_invalid_job_description_warns_before_any_model_call(app):
    analyze(app, RECIPE)
    assert not app.session_state["running"]
    assert app.warning[0].value == ("**CV:** Looks good.  \n**Pasted job description:** This doesn't look "
                                    "like a job posting. Please upload or paste a job description.")
    assert "profile" not in app.session_state
    assert not any(b.label == "Try again" for b in app.button)
    assert app.text_area[1].value == RECIPE


def test_pasted_text_that_is_not_a_resume_warns(app):
    app.text_area[0].set_value(RECIPE)
    analyze(app)
    assert app.warning[0].value.startswith("**CV:** This doesn't look like a resume.")
    assert "profile" not in app.session_state


def test_model_check_still_stops_a_cv_the_rules_let_through(app):
    FakeProvider.cv_type = "cover letter"
    analyze(app)
    assert app.error[0].value.startswith("This doesn't look like a CV (it reads like a cover letter).")
    assert "results" not in app.session_state
    assert not any(b.label == "Try again" for b in app.button)


def test_scoring_failure_keeps_earlier_stages_and_offers_try_again(app):
    FakeEmbedder.error = LLMError("embed-model is temporarily overloaded.", try_other_model=True)
    analyze(app)
    assert not app.session_state["running"]
    assert app.error[0].value == "embed-model is temporarily overloaded."
    assert "results" not in app.session_state
    assert "The jobs we're looking at" in text(app.markdown)
    assert "SQL" in text(app.markdown)  # the job's requirements
    assert "An analyst." in text(app.markdown)  # the profile
    assert any(b.label == "Try again" for b in app.button)


def test_failed_run_does_not_bring_old_results_back(app):
    analyze(app)
    FakeEmbedder.error = LLMError("overloaded", try_other_model=True)
    analyze(app, JD + " Apply today.")  # a new posting, so its embeddings aren't cached yet
    assert app.error[0].value == "overloaded"
    app.run()  # any later click reruns the page
    assert "results" not in app.session_state
    assert "Your matches" not in text(app.markdown)


def test_retry_after_scoring_failure_succeeds(app):
    FakeEmbedder.error = LLMError("overloaded", try_other_model=True)
    analyze(app)
    FakeEmbedder.error = None
    click(app, "Try again")
    assert len(app.session_state["results"]) == 1
    assert not app.error


def test_new_search_keeps_the_cv_and_empties_the_job_inputs(app):
    app.text_area[0].set_value(CV)
    analyze(app)
    assert "cv_text" in app.session_state
    click(app, "New search")
    assert "results" not in app.session_state and "cv_text" not in app.session_state
    assert "See how well your CV fits the job" in text(app.markdown)
    assert app.text_area[0].value == CV
    assert app.text_area[1].value == ""
