"""Profile-similarity modes: chunking, summary vs document, fallback, per-mode bands. No API keys."""
from dataclasses import replace

import pytest

from core.config import get_settings
from core.job_sources.sample import SampleSource
from core.matcher import score_jobs, similarity_band
from core.schemas import CandidateProfile, JobRequirements
from core.similarity import chunk_text, profile_cosines
from test_matcher import FakeEmbedder, fake_embed

PROFILE = CandidateProfile(summary="sql dashboards analyst", years_experience=3, seniority="mid",
                           skills=["sql"], job_titles=[])


def jobs_with(summaries):
    jobs = SampleSource().search("analyst", "", len(summaries))
    for j, sm in zip(jobs, summaries):
        j.requirements = JobRequirements(required_skills=["sql"], nice_to_have_skills=[], role_summary=sm)
    return jobs


def test_chunk_text_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), words_per_chunk=40, overlap=10)
    assert [c.split()[0] for c in chunks] == ["w0", "w30", "w60"]
    assert chunks[-1].split()[-1] == "w99"
    assert chunk_text("short text") == ["short text"]


def test_summary_mode_compares_summaries_and_falls_back_per_job():
    jobs = jobs_with(["sql dashboards analyst", ""])  # second job has no role_summary
    out = profile_cosines("summary", PROFILE, "unrelated cv words", jobs, fake_embed)
    assert out[0] == (pytest.approx(1.0), "summary")
    assert out[1][1] == "document"


def test_chunks_mode_uses_best_chunks():
    jobs = jobs_with(["x"])
    cv = " ".join(["filler"] * 200) + " " + jobs[0].description
    best = profile_cosines("chunks", PROFILE, cv, jobs, fake_embed, top_k=1)[0]
    blurry = profile_cosines("document", PROFILE, cv, jobs, fake_embed)[0]
    assert best[1] == "chunks" and best[0] > blurry[0]


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError):
        profile_cosines("magic", PROFILE, "cv", jobs_with(["x"]), fake_embed)


def test_each_mode_uses_its_own_band():
    emb = FakeEmbedder()
    emb.sim_bands = {"summary": (0.2, 0.4)}
    assert similarity_band(emb, "summary") == (0.2, 0.4)
    assert similarity_band(emb, "document") == (emb.cos_low, emb.cos_high)
    jobs = jobs_with(["sql dashboards analyst"])
    s = replace(get_settings(), profile_similarity="summary")
    assert score_jobs(PROFILE, "cv", jobs, emb, s)[0].semantic_similarity == 1.0
