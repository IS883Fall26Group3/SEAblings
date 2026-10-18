"""Jobs from descriptions the user pastes or uploads, instead of a live search."""
import hashlib

from ..schemas import Job

SOURCE = "Your job description"


def jobs_from_descriptions(items: list[tuple[str, str]]) -> list[Job]:
    """items: (name, text) pairs, e.g. (filename, extracted text). Empty texts are skipped.
    The title is the first line of the text, which is usually the role name."""
    jobs = []
    for name, text in items:
        text = text.strip()
        if not text:
            continue
        first = next(line.strip(" #*-:\t") for line in text.splitlines() if line.strip())
        title = first[:80] if 0 < len(first) <= 80 else name
        jobs.append(Job(
            id="jd-" + hashlib.sha256(text.encode()).hexdigest()[:12],
            title=title or name, company="", location="", description=text, source=SOURCE))
    # The same JD added twice (e.g. pasted and uploaded) is analyzed once.
    return list({j.id: j for j in jobs}.values())
