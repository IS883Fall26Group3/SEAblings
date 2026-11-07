"""Input checks (core/input_checks.py): a non-CV or a non-posting is never scored."""
from core.input_checks import cv_problem, posting_problem
from core.job_sources.sample import SampleSource
from core.schemas import CandidateProfile, JobRequirements


def profile(kind):
    return CandidateProfile(document_type=kind, summary="", years_experience=0, seniority="entry",
                            skills=[], job_titles=[])


def job(kind="job posting", skills=("SQL",), years=None):
    j = SampleSource().search("analyst", "", 1)[0]
    j.requirements = JobRequirements(document_type=kind, required_skills=list(skills),
                                     nice_to_have_skills=[], min_years_experience=years)
    return j


def test_cv_check():
    assert cv_problem(profile("cv")) is None
    assert cv_problem(profile("Resume")) is None
    assert cv_problem(profile("")) is None                      # older cached profiles: no field
    assert "recipe" in cv_problem(profile("recipe"))


def test_posting_check():
    assert posting_problem(job()) is None
    assert posting_problem(job("Job description", skills=(), years=2)) is None  # years alone count
    assert "news article" in posting_problem(job("news article"))
    assert "no job requirements" in posting_problem(job(skills=()))
    j = job(); j.requirements = None
    assert posting_problem(j) is None                            # not extracted: matcher decides
