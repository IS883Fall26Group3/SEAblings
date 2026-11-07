"""Jobs from descriptions the user pastes or uploads, instead of a live search."""
import hashlib
import re

from ..schemas import Job

SOURCE = "Your job description"
# A LinkedIn copy starts with the logo's caption, then the company name: "Company logo for, ATC."
_LOGO = re.compile(r"company logo for,?\s*(.*?)\.?$", re.IGNORECASE)


def _first_title_line(text: str) -> str:
    """The first line that isn't LinkedIn's logo caption or the company name right after it."""
    company = ""
    for line in text.splitlines():
        line = line.strip(" #*-:\t\xa0")
        if not line:
            continue
        if m := _LOGO.fullmatch(line):
            company = m[1].strip().lower()
            continue
        if company and line.lower().rstrip(".") == company.rstrip("."):
            continue
        return line
    return ""


def jobs_from_descriptions(items: list[tuple[str, str]]) -> list[Job]:
    """items: (name, text) pairs, e.g. (filename, extracted text). Empty texts are skipped.
    The title is the first line of the text, which is usually the role name, until
    name_from_requirements replaces it with the title the LLM read from the posting."""
    jobs = []
    for name, text in items:
        text = text.strip()
        if not text:
            continue
        first = _first_title_line(text)
        title = first[:80] if 0 < len(first) <= 80 else name
        jobs.append(Job(
            id="jd-" + hashlib.sha256(text.encode()).hexdigest()[:12],
            title=title or name, company="", location="", description=text, source=SOURCE))
    # The same JD added twice (e.g. pasted and uploaded) is analyzed once.
    return list({j.id: j for j in jobs}.values())


def name_from_requirements(jobs: list[Job]) -> list[Job]:
    """Use the title and company the LLM extracted, where it found them, for the user's own jobs."""
    for j in jobs:
        req = j.requirements
        if j.source != SOURCE or not req:
            continue
        if 0 < len(req.job_title.strip()) <= 80:
            j.title = req.job_title.strip()
        if req.company.strip():
            j.company = req.company.strip()
    return jobs
