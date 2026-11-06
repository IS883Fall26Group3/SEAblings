"""Experience fit from *relevant* years instead of total years. Two ways to pick the relevant roles:
  roles: embed each CV role and each job; a role counts when their cosine reaches role_threshold
         (no extra LLM call, deterministic).
  judge: the LLM picks the relevant roles per job with guided steps (skill_extractor.judge_experience).
Either way the years are added up here, in Python, from the roles' dates: LLMs are poor calculators."""
import re
from datetime import date
from typing import Callable

import numpy as np

from .schemas import Job, Role

EmbedFn = Callable[[list[str]], np.ndarray]


def _month(value: str, today: date) -> int | None:
    """Months since year 0 for 'YYYY-MM', 'YYYY' (taken as mid-year) or 'present'; None if unknown."""
    v = (value or "").strip().lower()
    if v in ("present", "current", "now"):
        return today.year * 12 + today.month - 1
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})", v):
        return int(m[1]) * 12 + int(m[2]) - 1
    if m := re.fullmatch(r"(\d{4})", v):
        return int(m[1]) * 12 + 5
    return None


def role_years(roles: list[Role], today: date | None = None) -> float:
    """Years covered by these roles, counting overlapping periods once.
    Roles without usable dates fall back to duration_months; roles with neither count as zero."""
    today = today or date.today()
    spans, loose = [], 0.0
    for r in roles:
        a, b = _month(r.start, today), _month(r.end, today)
        if a is not None and b is not None and b >= a:
            spans.append((a, b + 1))  # both months inclusive
        elif r.duration_months:
            loose += r.duration_months
    covered, reach = 0, None
    for a, b in sorted(spans):
        if reach is None or a >= reach:
            covered, reach = covered + b - a, b
        elif b > reach:
            covered, reach = covered + b - reach, b
    return (covered + loose) / 12


def experience_fit(relevant_years: float, min_years: float | None) -> float:
    """Share of the required experience the candidate has in relevant roles.
    With no stated minimum, one relevant year counts as a full fit."""
    return float(min(1.0, relevant_years / (min_years or 1.0)))


def role_text(r: Role) -> str:
    return f"{r.title}: {r.summary}".strip(": ")


def job_text(j: Job) -> str:
    skills = ", ".join(j.requirements.required_skills) if j.requirements else ""
    return f"{j.title}: {skills}".strip(": ")


def relevant_roles_by_roles(roles: list[Role], jobs: list[Job], embed: EmbedFn,
                            threshold: float) -> dict[str, list[Role]]:
    """Technique A: one embedding call for all roles and jobs; a role counts toward a job
    when cosine(role title + summary, job title + required skills) >= threshold."""
    roles = [r for r in roles if r.title.strip()]
    if not roles or not jobs:
        return {}  # nothing to judge by: score_jobs falls back to total years
    vecs = embed([role_text(r) for r in roles] + [job_text(j) for j in jobs])
    sims = vecs[len(roles):] @ vecs[:len(roles)].T  # (jobs, roles)
    return {j.id: [r for r, s in zip(roles, row) if s >= threshold] for j, row in zip(jobs, sims)}


def relevant_years_by_roles(roles: list[Role], jobs: list[Job], embed: EmbedFn, threshold: float,
                            today: date | None = None) -> dict[str, float]:
    """Years in the roles relevant_roles_by_roles keeps, per job."""
    return {jid: role_years(rs, today)
            for jid, rs in relevant_roles_by_roles(roles, jobs, embed, threshold).items()}
