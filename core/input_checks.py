"""Checks that the inputs are what the app expects, before anything is scored. A recipe uploaded
as a CV, or a news item pasted as a job, would otherwise get a match score as if it were real.
The LLM says what kind of text it read (document_type, filled in the same extraction call, so no
extra call), and a rule backs it up: a "posting" with no requirements at all is never scored."""
from .schemas import CandidateProfile, Job

_CV_WORDS = ("cv", "resume", "résumé", "curriculum vitae")
_POSTING_WORDS = ("job", "posting", "vacancy", "position", "role", "opening")


def cv_problem(profile: CandidateProfile) -> str | None:
    """Why this doesn't look like a CV, or None if it does."""
    kind = profile.document_type.strip().lower()
    if not kind or any(w in kind for w in _CV_WORDS):
        return None
    return f"This doesn't look like a CV (it reads like a {kind})."


def posting_problem(job: Job) -> str | None:
    """Why this job can't be scored, or None if it can. A job whose requirements weren't extracted
    (None) is left to the matcher as before."""
    req = job.requirements
    if req is None:
        return None
    kind = req.document_type.strip().lower()
    if kind and not any(w in kind for w in _POSTING_WORDS):
        return f"it doesn't read like a job posting (it reads like a {kind})"
    if not any(s.strip() for s in req.required_skills) and not req.min_years_experience:
        return "no job requirements were found in it"
    return None
