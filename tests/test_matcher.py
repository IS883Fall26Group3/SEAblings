"""Matcher tests use a fake embedding (bag of words), so they need no API key."""
import numpy as np

from core.config import get_settings
from core.job_sources.sample import SampleSource
from core.matcher import match_skills, score_jobs, seniority_fit
from core.schemas import CandidateProfile, JobRequirements

VOCAB = {}


def fake_embed(texts):
    rows = []
    for t in texts:
        v = np.zeros(512)
        for w in t.lower().split():
            v[VOCAB.setdefault(w, len(VOCAB) % 512)] += 1
        rows.append(v / (np.linalg.norm(v) or 1))
    return np.array(rows)


class FakeEmbedder:
    name, skill_threshold, cos_low, cos_high = "fake", 0.8, 0.0, 1.0

    def embed(self, texts):
        return fake_embed(texts)


def test_exact_and_missing_skills():
    vec = {s.lower(): fake_embed([s])[0] for s in ["sql", "python", "tableau"]}
    matched, missing = match_skills(["SQL", "Python"], ["sql", "Tableau"], vec, 0.8)
    assert matched == ["sql"] and missing == ["Tableau"]


def test_seniority_fit():
    assert seniority_fit(4, 2) == 1.0
    assert seniority_fit(2, 4) == 0.5
    assert seniority_fit(1, None) == 1.0


def test_similarity_texts_are_cut_at_the_embedders_limit():
    seen = []

    class SmallEmbedder(FakeEmbedder):
        max_chars = 50

        def embed(self, texts):
            seen.extend(texts)
            return fake_embed(texts)

    profile = CandidateProfile(summary="", years_experience=1, seniority="mid", skills=[], job_titles=[])
    [job] = SampleSource().search("", "", 1)
    job.description = "x" * 500
    score_jobs(profile, "y" * 500, [job], SmallEmbedder(), get_settings())
    assert "x" * 50 in seen and "y" * 50 in seen
    assert all(len(t) <= 50 for t in seen)


def test_better_skill_overlap_ranks_higher():
    profile = CandidateProfile(summary="planning analyst", years_experience=4, seniority="mid",
                               skills=["Financial modeling", "Excel", "Budgeting", "SQL"], job_titles=[])
    jobs = SampleSource().search("planning", "", 6)
    for j in jobs:
        if j.id == "s6":
            j.requirements = JobRequirements(job_id=j.id, required_skills=["Financial modeling", "Budgeting", "Excel"],
                                             nice_to_have_skills=[], min_years_experience=3)
        else:
            j.requirements = JobRequirements(job_id=j.id, required_skills=["PyTorch", "Docker", "Kubernetes"],
                                             nice_to_have_skills=[], min_years_experience=5)
    results = score_jobs(profile, "financial modeling budgeting excel", jobs, FakeEmbedder(), get_settings())
    assert results[0].job.id == "s6"
    assert all(0 <= r.score <= 100 for r in results)
