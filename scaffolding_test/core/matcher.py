"""Hybrid match score: skill coverage + semantic similarity + seniority fit.
Deterministic given embeddings, so scores are consistent and explainable."""
from typing import Callable

import numpy as np

from .config import Settings
from .schemas import CandidateProfile, Job, MatchResult

EmbedFn = Callable[[list[str]], np.ndarray]


def _norm(s: str) -> str:
    return s.strip().lower()


def match_skills(cv_skills: list[str], job_skills: list[str],
                 vec: dict[str, np.ndarray], threshold: float) -> tuple[list[str], list[str]]:
    """A job skill is matched if it equals, or is semantically close to, any CV skill."""
    if not job_skills:
        return [], []
    if not cv_skills:
        return [], list(job_skills)
    cv_norm = {_norm(s) for s in cv_skills}
    cv_mat = np.stack([vec[_norm(s)] for s in cv_skills])
    matched, missing = [], []
    for skill in job_skills:
        hit = _norm(skill) in cv_norm or float((cv_mat @ vec[_norm(skill)]).max()) >= threshold
        (matched if hit else missing).append(skill)
    return matched, missing


def seniority_fit(cv_years: float, min_years: float | None) -> float:
    if not min_years:
        return 1.0
    return float(min(1.0, cv_years / min_years))


def rescale_cosine(cos: float, low: float = 0.5, high: float = 0.9) -> float:
    """Embedding cosines cluster in a narrow band; stretch it to 0-1. Tune low/high."""
    return float(np.clip((cos - low) / (high - low), 0.0, 1.0))


def score_jobs(profile: CandidateProfile, cv_text: str, jobs: list[Job],
               embed: EmbedFn, s: Settings) -> list[MatchResult]:
    if not jobs:
        return []
    # Two embedding calls total: one for documents, one for all unique skills.
    doc_vecs = embed([cv_text[:s.max_jd_chars]] + [j.description[:s.max_jd_chars] for j in jobs])
    cv_vec, job_vecs = doc_vecs[0], doc_vecs[1:]

    all_skills = list(profile.skills)
    for j in jobs:
        if j.requirements:
            all_skills += j.requirements.required_skills
    unique = list(dict.fromkeys(_norm(x) for x in all_skills if x.strip()))
    vec = dict(zip(unique, embed(unique))) if unique else {}

    results = []
    for job, jv in zip(jobs, job_vecs):
        req = job.requirements
        required = [x for x in (req.required_skills if req else []) if x.strip()]
        cv_skills = [x for x in profile.skills if x.strip()]
        matched, missing = match_skills(cv_skills, required, vec, s.skill_match_threshold)
        coverage = len(matched) / len(required) if required else 0.5  # neutral if unknown
        semantic = rescale_cosine(float(jv @ cv_vec))
        senior = seniority_fit(profile.years_experience, req.min_years_experience if req else None)
        score = 100 * (s.w_skill_coverage * coverage + s.w_semantic * semantic
                       + s.w_seniority * senior)
        results.append(MatchResult(
            job=job, score=round(score, 1), skill_coverage=coverage,
            semantic_similarity=semantic, seniority_fit=senior,
            matched_skills=matched, missing_skills=missing))
    return sorted(results, key=lambda r: r.score, reverse=True)
