import json
from pathlib import Path

from core.validation import looks_like_job_description, looks_like_resume

DATA = Path(__file__).resolve().parents[1] / "data" / "sample"
CV = (DATA / "sample_cv.txt").read_text(encoding="utf-8")
JDS = [j["description"] for j in json.loads((DATA / "jobs.json").read_text(encoding="utf-8"))]
RECIPE = "Chocolate chip cookies. Preheat the oven to 180C. Mix butter and sugar, then add flour. " * 5


def test_sample_cv_is_a_resume_not_a_job_description():
    assert looks_like_resume(CV)
    assert not looks_like_job_description(CV)


def test_sample_jobs_are_job_descriptions_not_resumes():
    for jd in JDS:
        assert looks_like_job_description(jd)
        assert not looks_like_resume(jd)


def test_unrelated_text_is_neither():
    assert not looks_like_resume(RECIPE)
    assert not looks_like_job_description(RECIPE)


def test_posting_with_resume_style_headings_is_not_a_resume():
    jd = ("Data Analyst\n\nAbout the role\nWe are looking for an analyst to join our team.\n\n"
          "Responsibilities\nBuild dashboards.\n\nRequirements\n3+ years of experience in SQL.\n\n"
          "Education\nBachelor's degree required.\n\nSkills\nSQL, Python\n\nBenefits\nHealth insurance.")
    assert looks_like_job_description(jd)
    assert not looks_like_resume(jd)
