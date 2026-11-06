"""Hybrid match score: skill coverage + semantic similarity + seniority fit.
Deterministic given embeddings, so scores are consistent and explainable."""
from typing import Callable

import numpy as np

from .config import Settings
from .experience import experience_fit, role_years
from .similarity import profile_cosines
from .schemas import CandidateProfile, Job, MatchResult, Role

EmbedFn = Callable[[list[str]], np.ndarray]


def _norm(s: str) -> str:
    return s.strip().lower()


def match_skills_detail(cv_skills: list[str], job_skills: list[str], vec: dict[str, np.ndarray],
                        threshold: float) -> tuple[list[str], list[str], dict[str, str]]:
    """match_skills, plus the closest CV skill for each job skill matched by meaning, not name."""
    if not job_skills:
        return [], [], {}
    if not cv_skills:
        return [], list(job_skills), {}
    cv_norm = {_norm(s) for s in cv_skills}
    cv_mat = np.stack([vec[_norm(s)] for s in cv_skills])
    matched, missing, close = [], [], {}
    for skill in job_skills:
        if _norm(skill) in cv_norm:
            matched.append(skill)
            continue
        sims = cv_mat @ vec[_norm(skill)]
        if float(sims.max()) >= threshold:
            matched.append(skill)
            close[skill] = cv_skills[int(sims.argmax())]
        else:
            missing.append(skill)
    return matched, missing, close


def match_skills(cv_skills: list[str], job_skills: list[str],
                 vec: dict[str, np.ndarray], threshold: float) -> tuple[list[str], list[str]]:
    """A job skill is matched if it equals, or is semantically close to, any CV skill."""
    matched, missing, _ = match_skills_detail(cv_skills, job_skills, vec, threshold)
    return matched, missing


def seniority_fit(cv_years: float, min_years: float | None) -> float:
    if not min_years:
        return 1.0
    return float(min(1.0, cv_years / min_years))


def rescale_cosine(cos: float, low: float = 0.5, high: float = 0.9) -> float:
    """Embedding cosines cluster in a narrow band; stretch it to 0-1.
    The band depends on the embedding model, so low/high come from the embedder."""
    return float(np.clip((cos - low) / (high - low), 0.0, 1.0))


def similarity_band(embedder, mode: str) -> tuple[float, float]:
    """Each PROFILE_SIMILARITY mode has its own cosine range; 'document' uses cos_low/cos_high."""
    return getattr(embedder, "sim_bands", {}).get(mode, (embedder.cos_low, embedder.cos_high))


def score_jobs(profile: CandidateProfile, cv_text: str, jobs: list[Job], embedder,
               s: Settings, embed_fn: EmbedFn | None = None,
               relevant_years: dict[str, float] | None = None,
               relevant_roles: dict[str, list[Role]] | None = None) -> list[MatchResult]:
    """embedder supplies thresholds (and embed() unless embed_fn, e.g. a cached version, is given).
    relevant_roles (job id -> CV roles relevant to it, see core/experience.py), or relevant_years
    (job id -> their years), replaces total years in the experience fit; jobs missing from it fall
    back to total years."""
    if not jobs:
        return []
    embed = embed_fn or embedder.embed
    # Two embedding calls total: one for profile similarity (core/similarity.py), one for all skills.
    sims = profile_cosines(s.profile_similarity, profile, cv_text, jobs, embed,
                           getattr(embedder, "max_chars", 7_000))

    all_skills = list(profile.skills)
    for j in jobs:
        if j.requirements:
            all_skills += j.requirements.required_skills
    unique = list(dict.fromkeys(_norm(x) for x in all_skills if x.strip()))
    vec = dict(zip(unique, embed(unique))) if unique else {}

    results = []
    for job, (cos, mode) in zip(jobs, sims):
        req = job.requirements
        required = [x for x in (req.required_skills if req else []) if x.strip()]
        cv_skills = [x for x in profile.skills if x.strip()]
        matched, missing, close = match_skills_detail(cv_skills, required, vec, embedder.skill_threshold)
        coverage = len(matched) / len(required) if required else 0.5  # neutral if unknown
        semantic = rescale_cosine(cos, *similarity_band(embedder, mode))
        min_years = req.min_years_experience if req else None
        counted, relevant = [], None
        if relevant_roles and relevant_roles.get(job.id) is not None:
            counted = relevant_roles[job.id]
            relevant = role_years(counted)
        elif relevant_years and relevant_years.get(job.id) is not None:
            relevant = relevant_years[job.id]
        if relevant is not None:
            # capped at total years: a role given both company dates and a duration can double count
            years = min(relevant, profile.years_experience)
            senior = experience_fit(years, min_years)
        else:
            years = profile.years_experience
            senior = seniority_fit(years, min_years)
        score = 100 * (s.w_skill_coverage * coverage + s.w_semantic * semantic
                       + s.w_seniority * senior)
        results.append(MatchResult(
            job=job, score=round(score, 1), skill_coverage=coverage,
            semantic_similarity=semantic, seniority_fit=senior,
            matched_skills=matched, missing_skills=missing, close_matches=close,
            years_counted=years, total_years=profile.years_experience, min_years=min_years,
            relevant_only=relevant is not None, counted_roles=counted))
    return sorted(results, key=lambda r: r.score, reverse=True)
