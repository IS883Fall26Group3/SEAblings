"""Hybrid match score: skill coverage + semantic similarity + seniority fit.
Deterministic given embeddings, so scores are consistent and explainable."""
from typing import Callable

import numpy as np

from .config import Settings
from .experience import experience_fit, role_years
from .similarity import profile_cosines
from .skill_names import skill_key
from .schemas import CandidateProfile, Job, MatchResult, RequirementCheck, Role

EmbedFn = Callable[[list[str]], np.ndarray]


def _norm(s: str) -> str:
    return s.strip().lower()


def match_skills_detail(cv_skills: list[str], job_skills: list[str], vec: dict[str, np.ndarray],
                        threshold: float) -> tuple[list[str], list[str], dict[str, str]]:
    """match_skills, plus the CV skill behind each job skill matched under another wording: the same
    standard name ("MS Excel" for "Excel", core/skill_names.py) or a close meaning."""
    if not job_skills:
        return [], [], {}
    if not cv_skills:
        return [], list(job_skills), {}
    cv_by_key = {}
    for s in cv_skills:
        cv_by_key.setdefault(skill_key(s), s)
    cv_mat = np.stack([vec[_norm(s)] for s in cv_skills])
    matched, missing, close = [], [], {}
    for skill in job_skills:
        same = cv_by_key.get(skill_key(skill))
        if same is not None:
            matched.append(skill)
            if _norm(same) != _norm(skill):
                close[skill] = same
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
    """A job skill is matched if it has the same standard name as, or is semantically close to, any CV skill."""
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


# Best of 6 factor sets on the 175 rated labeled pairs (eval/background.py, eval/README.md step 7):
# similarity alone vs labels, Spearman 0.288 -> 0.459, NDCG@5 0.577 -> 0.710; picked on every
# leave-one-CV-out split.
BACKGROUND_VALUES = {"same": 1.0, "related": 0.5, "different": 0.0}
BACKGROUND_COMBINE = "mean"


def background_factor(field_match: str, work_match: str, values: dict[str, float] | None = None,
                      combine: str | None = None) -> float:
    """Multiplier for profile similarity from the judge's same / related / different rating of the
    field and the kind of work. The embedding sees two career documents sharing generic wording
    ("management", "operations") as alike even across fields; this rating can only lower the score.
    An unrated dimension counts as no evidence (1.0)."""
    values = values or BACKGROUND_VALUES
    v = [values.get(x.strip().lower(), 1.0) for x in (field_match, work_match)]
    how = combine or BACKGROUND_COMBINE
    return float(min(v) if how == "min" else max(v) if how == "max" else sum(v) / 2)


PARTIAL_CREDIT = 0.5  # a requirement labeled partial counts half toward skill coverage


def combine_score(coverage: float, semantic: float, experience: float, s: Settings) -> float:
    """The match score, 0-100. Also the advice model's compute_match_score tool (core/advisor.py)."""
    return round(100 * (s.w_skill_coverage * coverage + s.w_semantic * semantic
                        + s.w_seniority * experience), 1)


def apply_checks(required: list[str], checks: list[RequirementCheck], matched: list[str],
                 close: dict[str, str]) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Use the per-requirement labels (met / partial / missing) where there is one; the skill
    matcher's verdict for any requirement the labels skipped. Returns met, partial, missing and the
    matcher's close matches that still apply."""
    by_req = {_norm(c.requirement): c.status for c in checks}
    met, partial, missing = [], [], []
    for skill in required:
        status = by_req.get(_norm(skill)) or ("met" if skill in matched else "missing")
        {"met": met, "partial": partial}.get(status, missing).append(skill)
    close = {k: v for k, v in close.items() if k in met and _norm(k) not in by_req}
    return met, partial, missing, close


def unlabeled_skills(job: Job, checks: list[RequirementCheck]) -> list[str]:
    """The job's required skills that have no label, so the backup skill matcher decides them."""
    req = job.requirements
    labeled = {_norm(c.requirement) for c in checks}
    return [x for x in (req.required_skills if req else []) if x.strip() and _norm(x) not in labeled]


def similarity_band(embedder, mode: str) -> tuple[float, float]:
    """Each PROFILE_SIMILARITY mode has its own cosine range; 'document' uses cos_low/cos_high."""
    return getattr(embedder, "sim_bands", {}).get(mode, (embedder.cos_low, embedder.cos_high))


def score_jobs(profile: CandidateProfile, cv_text: str, jobs: list[Job], embedder,
               s: Settings, embed_fn: EmbedFn | None = None,
               relevant_years: dict[str, float] | None = None,
               relevant_roles: dict[str, list[Role]] | None = None,
               background: dict[str, tuple[str, str]] | None = None,
               requirement_checks: dict[str, list[RequirementCheck]] | None = None) -> list[MatchResult]:
    """embedder supplies thresholds (and embed() unless embed_fn, e.g. a cached version, is given).
    relevant_roles (job id -> CV roles relevant to it, see core/experience.py), or relevant_years
    (job id -> their years), replaces total years in the experience fit; jobs missing from it fall
    back to total years. background (job id -> the judge's field and kind-of-work rating) scales
    profile similarity down (background_factor); jobs missing from it keep the embedding score.
    requirement_checks (job id -> met / partial / missing labels from retrieval + LLM,
    core/requirements.py) decide skill coverage, a partial counting half; without them, or for a
    requirement they skip, the skill matcher decides by name and embedding."""
    if not jobs:
        return []
    embed = embed_fn or embedder.embed
    # Two embedding calls total: one for profile similarity (core/similarity.py), one for all skills.
    sims = profile_cosines(s.profile_similarity, profile, cv_text, jobs, embed,
                           getattr(embedder, "max_chars", 7_000))

    # The backup skill matcher only decides requirements the labels skipped, so only those (and the
    # CV skills) are embedded. When every requirement is labeled, this embedding call is skipped.
    cv_skills = [x for x in profile.skills if x.strip()]
    unlabeled = {j.id: unlabeled_skills(j, (requirement_checks or {}).get(j.id) or []) for j in jobs}
    todo = [x for skills in unlabeled.values() for x in skills]
    unique = list(dict.fromkeys(_norm(x) for x in (cv_skills + todo) if x.strip())) if todo else []
    vec = dict(zip(unique, embed(unique))) if unique else {}

    results = []
    for job, (cos, mode) in zip(jobs, sims):
        req = job.requirements
        required = [x for x in (req.required_skills if req else []) if x.strip()]
        matched, missing, close = match_skills_detail(cv_skills, unlabeled[job.id], vec,
                                                      embedder.skill_threshold)
        checks = (requirement_checks or {}).get(job.id) or []
        partial = []
        if checks:
            matched, partial, missing, close = apply_checks(required, checks, matched, close)
        coverage = ((len(matched) + PARTIAL_CREDIT * len(partial)) / len(required)
                    if required else 0.5)  # neutral if unknown
        embedded = rescale_cosine(cos, *similarity_band(embedder, mode))
        field_match, work_match = (background or {}).get(job.id, ("", ""))
        semantic = embedded * background_factor(field_match, work_match)
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
        results.append(MatchResult(
            job=job, score=combine_score(coverage, semantic, senior, s), skill_coverage=coverage,
            semantic_similarity=semantic, seniority_fit=senior,
            matched_skills=matched, missing_skills=missing, close_matches=close,
            years_counted=years, total_years=profile.years_experience, min_years=min_years,
            relevant_only=relevant is not None, counted_roles=counted, embedding_similarity=embedded,
            field_match=field_match.strip().lower(), work_match=work_match.strip().lower(),
            partial_skills=partial, requirement_checks=checks))
    return sorted(results, key=lambda r: r.score, reverse=True)
