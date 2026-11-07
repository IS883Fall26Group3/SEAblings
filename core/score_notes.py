"""Plain-language reasons for the three scores on a job card, built from what the matcher recorded.
No LLM calls: the text always agrees with the numbers. The field / type of work labels describe the
CV and the posting (written by the LLM during extraction); they explain profile similarity but
don't calculate it."""
from dataclasses import dataclass, field

from .schemas import CandidateProfile, MatchResult, Role

SKILL_HELP = ("Share of the job's required skills your CV shows. Each skill is checked against the CV "
              "lines closest to it; a partly met skill (a related tool or a lower level) counts half.")
SIMILARITY_HELP = ("Compares your whole CV with the whole posting. The field, type of work, level and "
                   "tools all count together.")
EXPERIENCE_HELP = ("Years in your roles that are relevant to this job, compared with the minimum the "
                   "posting asks for. Each role's years are added up from its dates on your CV.")
TOTAL_HELP = "Years of experience on your CV, compared with the minimum the posting asks for."

SIMILARITY_LEVELS = [  # (lowest percent, label, sentence)
    (100, "Very strong match", "Your CV and this posting describe very similar work."),
    (67, "Strong match", "Your CV and this posting have a lot in common."),
    (34, "Partial match", "Your CV overlaps with parts of this posting."),
    (1, "Weak match", "Your CV has little in common with this posting."),
    (0, "No clear match", "Your CV describes quite different work from this posting."),
]


# The judge's rating as shown in the Match column. "same" reads as "100%" beside a lower profile
# similarity (the similarity also depends on wording), so it is shown as "Closely related".
MATCH_LABELS = {"same": "Closely related", "related": "Related", "different": "Different"}


@dataclass
class Note:
    label: str
    text: str
    bullets: list[str] = field(default_factory=list)
    table: list[tuple[str, ...]] = field(default_factory=list)  # (row, your CV, this job[, match])
    after: str = ""


def _pct(x: float) -> int:
    return round(100 * x)


def _num(y: float) -> str:
    return f"{round(y, 1):g}"


def _years(y: float) -> str:
    return f"{_num(y)} year" if round(y, 1) == 1 else f"{_num(y)} years"


def _them(k: int) -> str:
    return "the one" if k == 1 else f"any of the {k}"


def role_line(r: Role) -> str:
    """'Sales Branch Manager (2021-2025)', or with a stated duration when the CV gives no dates."""
    start = r.start.strip()[:4]
    end = "present" if r.end.strip().lower() in ("present", "current", "now") else r.end.strip()[:4]
    if start and end:
        span = start if start == end else f"{start}–{end}"
    elif r.duration_months:
        span = f"{round(r.duration_months)} months" if r.duration_months < 24 else _years(r.duration_months / 12)
    else:
        span = ""
    return f"{r.title} ({span})" if span else r.title


def evidence_bullets(r: MatchResult, limit: int = 6) -> list[str]:
    """'SQL: "Wrote SQL queries to consolidate sales data"', for each met or partly met requirement
    whose quote was found word for word in the CV."""
    out = [f'{c.requirement}{" (partly)" if c.status == "partial" else ""}: "{c.quote}"'
           for c in r.requirement_checks if c.quote_verified and c.status in ("met", "partial")]
    return out[:limit] + ([f"{len(out) - limit} more."] if len(out) > limit else [])


def skill_note(r: MatchResult, profile: CandidateProfile) -> Note:
    m, p, k = len(r.matched_skills), len(r.partial_skills), len(r.missing_skills)
    n = m + p + k
    if n == 0:
        return Note("No clear requirements", "This posting doesn't list clear required skills, so this "
                    "part is set to a neutral 50%.")
    if p:
        note = _skill_note_with_partial(r, m, p, k, n)
        note.bullets = evidence_bullets(r)
        return note
    note = _skill_note(r, profile, m, k, n)
    note.bullets = evidence_bullets(r)
    return note


def _skill_note_with_partial(r: MatchResult, m: int, p: int, k: int, n: int) -> Note:
    pct = _pct(r.skill_coverage)
    label = "Most required skills" if pct >= 50 else "Some required skills"
    parts = [f"You have {m} of the {n} skills this job requires, and {p} more partly." if m
             else f"You partly have {p} of the {n} skills this job requires."]
    parts.append("A partly met skill (a related tool, a lower level or a nearby industry) counts half.")
    if k:
        parts.append('The ones under "To build" weren\'t found in your CV.' if k > 1
                     else 'The one under "To build" wasn\'t found in your CV.')
    return Note(label, " ".join(parts))


def _skill_note(r: MatchResult, profile: CandidateProfile, m: int, k: int, n: int) -> Note:
    pct = _pct(m / n)
    label = ("All required skills" if pct == 100 else "Most required skills" if pct >= 50
             else "Some required skills" if pct > 0 else "No required skills found")
    if not any(s.strip() for s in profile.skills):
        return Note(label, f"We couldn't read any skills from your CV, so none of the {n} required "
                    "skills could be matched. Add a skills section and try again.")
    if m == n:
        parts = ["You have the one skill this job requires." if n == 1
                 else f"You have all {n} skills this job requires."]
    elif m == 0:
        parts = [f"None of the {n} skills this job requires were found in your CV."]
    else:
        parts = [f"You have {m} of the {n} skills this job requires."]
    close = list(r.close_matches.items())
    parts += [f'"{job}" counted because your CV lists "{cv}", a close match.' for job, cv in close[:2]]
    if len(close) > 2:
        parts.append(f"{len(close) - 2} more counted as close matches.")
    if k and pct < 50:
        parts.append(f'If you have {_them(k)} under "To build", name them in your CV; skills described '
                     "only in your experience may be missed.")
    elif k and m:
        parts.append('The one under "To build" wasn\'t found in your CV.' if k == 1
                     else f'The {k} under "To build" weren\'t found in your CV.')
    return Note(label, " ".join(parts))


def _cap(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:]


def _adjustment(r: MatchResult, pct: int) -> str:
    """Why the rating lowered the embedding score, e.g. 'The wording overlaps (36%), but the field
    and type of work are different, so this counts as 0%.' Empty when nothing was lowered."""
    before = _pct(r.embedding_similarity)
    if before <= pct:
        return ""
    reasons = {"different": [], "related": []}
    for name, match in (("field", r.field_match), ("type of work", r.work_match)):
        if match in reasons:
            reasons[match].append(name)
    parts = []
    for match, names in reasons.items():
        if names:
            what = " and ".join(names)
            verb = "are" if len(names) > 1 else "is"
            parts.append(f"the {what} {verb} {'different' if match == 'different' else 'only related'}")
    return (f"The wording of your CV and this posting overlaps ({before}%), but {' and '.join(parts)}, "
            f"so this counts as {pct}%.")


def similarity_note(r: MatchResult, profile: CandidateProfile) -> Note:
    pct = _pct(r.semantic_similarity)
    label, sentence = next((lab, s) for low, lab, s in SIMILARITY_LEVELS if pct >= low)
    req = r.job.requirements
    rated = {"Field": r.field_match, "Type of work": r.work_match}
    show_match = any(rated.values())
    rows = []
    if req:
        for name, cv, job in [("Field", profile.field, req.field),
                              ("Type of work", profile.work_type, req.work_type),
                              ("Level", profile.seniority, req.level)]:
            if cv.strip() and job.strip():  # a row only when both sides have it
                row = (name, _cap(cv), _cap(job))
                rows.append(row + (MATCH_LABELS.get(rated.get(name, ""), ""),) if show_match else row)
    if len(rows) == 1 and rows[0][0] == "Level":
        rows = []  # level alone says too little: show it only beside field or type of work
    return Note(label, _adjustment(r, pct) or sentence, table=rows)


def experience_note(r: MatchResult) -> Note:
    pct = _pct(r.seniority_fit)
    label = ("Meets the requirement" if pct == 100 else "Partly meets the requirement" if pct >= 50
             else "Below the requirement" if pct > 0 else "No relevant experience")
    y, mn = r.years_counted, r.min_years
    if not r.relevant_only:
        if mn:
            return Note(label, f"This job asks for {_num(mn)}+ years; we counted all {_years(y)} on your CV.")
        return Note(label, "This posting doesn't state a minimum, so any experience counts as a full fit.")

    roles = [role_line(x) for x in r.counted_roles if x.title.strip()]
    lead = ":" if roles else "."
    others = "Your other roles weren't counted as experience for this job." if 0 < y < r.total_years - 0.05 else ""
    if y <= 0:
        none = "None of your roles were counted as experience for this job."
        if mn:
            return Note("No relevant experience", f"This job asks for {_num(mn)}+ years. {none}")
        return Note("No relevant experience", f"This posting doesn't state a minimum, but {none[0].lower()}{none[1:]}")
    if mn:
        if others:
            text = f"This job asks for {_num(mn)}+ years; {_num(y)} of your {_years(r.total_years)} count as relevant"
        else:
            text = f"This job asks for {_num(mn)}+ years; we counted {_years(y)} of relevant experience"
    elif pct == 100:
        text = "This posting doesn't state a minimum; you have relevant experience, so this counts as a full fit"
    else:
        text = (f"This posting doesn't state a minimum; you have {_years(y)} of relevant experience, and "
                "a full fit needs at least one")
    return Note(label, text + lead, bullets=roles, after=others)
