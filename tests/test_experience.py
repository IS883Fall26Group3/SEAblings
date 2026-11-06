"""Relevant-experience tests: date arithmetic, both techniques, and the matcher hook. No API keys."""
from datetime import date

from core.config import get_settings
from core.experience import experience_fit, relevant_years_by_roles, role_years
from core.job_sources.sample import SampleSource
from core.matcher import score_jobs
from core.schemas import CandidateProfile, JobRequirements, Role
from core.skill_extractor import judge_experience
from test_llm import Scripted
from test_matcher import FakeEmbedder, fake_embed

TODAY = date(2026, 10, 31)


def test_role_years_counts_overlap_once_and_uses_durations():
    roles = [Role(title="a", start="2020-01", end="2021-12"),       # 24 months
             Role(title="b", start="2021-01", end="2022-12"),       # overlaps a by 12
             Role(title="c", duration_months=6),                     # no dates, stated duration
             Role(title="d")]                                        # nothing usable
    assert role_years(roles, TODAY) == (36 + 6) / 12


def test_role_years_present_and_year_only():
    assert role_years([Role(title="x", start="2026-01", end="present")], TODAY) == 10 / 12
    assert role_years([Role(title="x", start="2020", end="2020")], TODAY) == 1 / 12


def test_experience_fit_uses_minimum_or_one_year():
    assert experience_fit(2, 4) == 0.5
    assert experience_fit(6, 4) == 1.0
    assert experience_fit(0.5, None) == 0.5


def test_roles_technique_counts_only_similar_roles():
    jobs = SampleSource().search("analyst", "", 1)
    jobs[0].requirements = JobRequirements(required_skills=["sql", "dashboards"], nice_to_have_skills=[])
    title = jobs[0].title
    roles = [Role(title=title, summary="sql dashboards", start="2020-01", end="2021-12"),
             Role(title="pastry chef", summary="bread cakes", start="2015-01", end="2019-12")]
    years = relevant_years_by_roles(roles, jobs, fake_embed, threshold=0.5, today=TODAY)
    assert years == {jobs[0].id: 2.0}
    assert relevant_years_by_roles([], jobs, fake_embed, 0.5) == {}  # no roles: fall back


def test_judge_technique_returns_roles_per_job():
    jobs = SampleSource().search("analyst", "", 2)
    reply = ('{"items": [{"job_id": "%s", "reasoning": "analytics role", "relevant_roles": '
             '[{"title": "Analyst", "start": "2022-01", "end": "2023-12"}]}, '
             '{"job_id": "%s", "reasoning": "none", "relevant_roles": []}]}' % (jobs[0].id, jobs[1].id))
    judged = judge_experience(Scripted([reply]), "cv text", jobs)
    assert role_years(judged[jobs[0].id].relevant_roles, TODAY) == 2.0
    assert judged[jobs[1].id].relevant_roles == []


def test_judge_reads_the_posting_like_extraction_does():
    class Recording(Scripted):
        def complete(self, prompt, **kw):
            self.prompts.append(prompt)
            return super().complete(prompt, **kw)

    [job] = SampleSource().search("analyst", "", 1)
    job.description = "x" * 9_000 + " REQUIREMENTS"
    reply = '{"items": [{"job_id": "%s", "reasoning": "", "relevant_roles": []}]}' % job.id
    gemini_like = Recording([reply])
    gemini_like.prompts = []
    judge_experience(gemini_like, "cv", [job])
    assert "REQUIREMENTS" in gemini_like.prompts[0]  # 10,000 chars: the end of the posting is read

    local_like = Recording([reply])
    local_like.prompts, local_like.batch_jobs = [], False
    judge_experience(local_like, "cv", [job])
    assert "REQUIREMENTS" not in local_like.prompts[0] and "x" * 1_500 in local_like.prompts[0]


def test_score_jobs_uses_relevant_years_when_given():
    jobs = SampleSource().search("analyst", "", 1)
    jobs[0].requirements = JobRequirements(required_skills=["sql"], nice_to_have_skills=[], min_years_experience=4)
    profile = CandidateProfile(summary="", years_experience=10, seniority="senior",
                               skills=["sql"], job_titles=[])
    s = get_settings()
    total = score_jobs(profile, "cv", jobs, FakeEmbedder(), s)[0]
    relevant = score_jobs(profile, "cv", jobs, FakeEmbedder(), s, relevant_years={jobs[0].id: 1.0})[0]
    assert total.seniority_fit == 1.0 and relevant.seniority_fit == 0.25
