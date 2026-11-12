"""Tools the advice model can call (tool calling, core/llm/gemini.py: run_with_tools).

The advice prompt holds no score and no skill lists. To say anything about the fit, the model has to
call compute_match_score, which runs the app's own formula (core/matcher.py: combine_score); to suggest
rewording, it calls search_cv, which retrieves the CV lines closest to a phrase. So the score is never
a number the model makes up, and a suggestion can only start from something the CV really says.
Each tool's docstring is what the model reads to decide when to call it."""
from typing import Callable

import numpy as np

from .config import Settings
from .matcher import combine_score
from .requirements import cv_lines, norm
from .schemas import MatchResult

EmbedFn = Callable[[list[str]], np.ndarray]


def requirement_rows(r: MatchResult) -> list[dict]:
    """Every required skill with its label, the labeler's reason and, when verified, the CV words
    behind it."""
    checks = {norm(c.requirement): c for c in r.requirement_checks}
    rows = [(s, "met") for s in r.matched_skills] + [(s, "partial") for s in r.partial_skills] \
        + [(s, "missing") for s in r.missing_skills]
    out = []
    for s, st in rows:
        c = checks.get(norm(s))
        out.append({"requirement": s, "status": st, "cv_quote": c.quote if c and c.quote_verified else "",
                    "reason": c.reason if c else ""})
    return out


def match_tools(r: MatchResult, cv_text: str, embed: EmbedFn, s: Settings) -> dict[str, Callable]:
    def compute_match_score(job_id: str) -> dict:
        """Compute the match score (0-100) between the candidate's CV and the job with SEAblings'
        formula: skill coverage, profile similarity and experience fit, weighted. Returns the score,
        each part, and every required skill labeled met, partial or missing with the CV words that
        show it. Call this before saying anything about the fit."""
        if job_id.strip() != r.job.id:
            return {"error": f"Unknown job_id '{job_id}'. This job's id is '{r.job.id}'."}
        return {
            "job_id": r.job.id,
            "match_score": combine_score(r.skill_coverage, r.semantic_similarity, r.seniority_fit, s),
            "formula": (f"{s.w_skill_coverage:.0%} skill coverage + {s.w_semantic:.0%} profile similarity "
                        f"+ {s.w_seniority:.0%} experience fit; a partly met skill counts half"),
            "skill_coverage": round(r.skill_coverage, 2),
            "profile_similarity": round(r.semantic_similarity, 2),
            "experience_fit": round(r.seniority_fit, 2),
            "years_counted": round(r.years_counted, 1),
            "min_years_required": r.min_years,
            "field_match": r.field_match or "not rated",
            "work_match": r.work_match or "not rated",
            "requirements": requirement_rows(r),
        }

    def search_cv(phrase: str) -> dict:
        """Search the candidate's CV for the lines closest in meaning to a phrase, such as a missing
        skill. Use it to check whether the CV already shows related experience that could be worded
        better. Returns up to 3 lines copied from the CV, closest first."""
        lines = cv_lines(cv_text)
        if not lines or not phrase.strip():
            return {"phrase": phrase, "cv_lines": []}
        vecs = embed(lines + [norm(phrase)])
        order = np.argsort(-(vecs[:-1] @ vecs[-1]))[:3]
        return {"phrase": phrase, "cv_lines": [lines[i] for i in order]}

    return {"compute_match_score": compute_match_score, "search_cv": search_cv}
