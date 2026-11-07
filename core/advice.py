"""Advice for one job, on demand: strongest match, main gap, next step. Each requirement in the posting
retrieves its closest CV chunks (Session 5 RAG), the LLM rates it, and Python checks the LLM's quotes
against the CV. The two LLM calls run only on click; nothing here changes a score or a chip."""
import logging
import re
from typing import Callable, NamedTuple, Optional

import numpy as np
from pydantic import BaseModel, Field

from .llm.base import LLMError, LLMProvider
from .schemas import CandidateProfile, MatchResult, Role
from .similarity import chunk_text
from .skill_extractor import _prompt

log = logging.getLogger("seablings.llm")

EmbedFn = Callable[[list[str]], np.ndarray]
MAX_REQUIREMENTS = 15
MAX_QUOTE_WORDS = 15
MAX_NOTES = 3
SECTION = re.compile(r"summary|profile|objective|experience|employment|work history|education|skills|projects"
                     r"|certifications?|languages|awards|publications|activities|volunteer\w*|interests", re.I)
# Sections where a CV describes the person rather than the work; "skills and experience" still counts as work
SELF_DESCRIBED = re.compile(r"summary|profile|objective|about|skills|competenc", re.I)
WORK_SECTION = re.compile(r"experience|employment|work history|projects", re.I)
DATE = re.compile(r"\b(?:19|20)\d{2}\b|\bpresent\b", re.I)
BULLETS = ("-", "*", "•", "–")
# Words too general to show that a next step is about a particular gap
GENERIC_WORDS = {"ability", "and", "any", "best", "demonstrated", "experience", "for", "knowledge", "proven",
                 "relevant", "skill", "skills", "strong", "the", "understanding", "use", "using", "with", "work",
                 "working", "year", "years"}


class ListedRequirement(BaseModel):
    text: str = Field(description="One requirement, in the posting's own words")
    chip: str = Field(default="", description="The item from the 'To build' list this covers, copied "
                                              "exactly, or empty")
    kind: str = Field(default="work", description="basic for a condition of being hired (age, driving licence, "
                                                  "schedule, work authorization); otherwise work")


class RequirementList(BaseModel):
    items: list[ListedRequirement]


class RequirementRating(BaseModel):
    id: int
    reasoning: str = Field(default="", description="One sentence: what the CV does or doesn't show")
    rating: str = Field(description="One of: strong, weak, not_shown")
    quote: str = Field(default="", description="Exact words from one line of the CV; empty for not_shown")


class AdviceReply(BaseModel):
    """Reasoning comes first: each requirement is rated before anything is picked."""
    ratings: list[RequirementRating]
    match_id: Optional[int] = Field(default=None, description="id of the strongest match, or null if none")
    gap_id: Optional[int] = Field(default=None, description="id of the most important gap, or null if none")
    next_step: str = Field(default="", description="One sentence: the single most useful step, naming the gap")


class CheckedRating(NamedTuple):
    rating: str
    quote: str
    lowered: bool = False  # rated strong on the CV's own description, so it counts as weak


def _titles(roles: list[Role] | None) -> list[str]:
    return [r.title.strip().lower() for r in roles or [] if r.title.strip()]


def _is_role(line: str, titles: list[str]) -> bool:
    """A line naming a CV role. A long one needs a date, so a summary sentence with a job title doesn't count."""
    if line.startswith(BULLETS) or line.endswith(".") or not any(title in line.lower() for title in titles):
        return False
    words = len(line.split())
    return words <= 15 or (words <= 30 and bool(DATE.search(line)))


def _is_section(line: str) -> bool:
    text = line.strip(" :")
    if not text or line.startswith(BULLETS) or text.endswith(".") or text[0].islower():
        return False
    words = len(text.split())
    return (words <= 6 and text.isupper()) or (words <= 4 and bool(SECTION.search(text)))


def _continues(previous: str, line: str) -> bool:
    """A PDF wraps long lines: the rest of a sentence starts in lowercase, or follows a trailing comma."""
    return not line.startswith(BULLETS) and (line[0].islower() or previous.endswith(","))


def _cv_lines(cv_text: str, titles: list[str]) -> list[tuple[str, str]]:
    """(kind, line) for each line of the CV, with wrapped lines joined. Kinds: blank, role, section, text."""
    lines = []
    for raw in cv_text.splitlines():
        line = raw.strip()
        if not line:
            lines.append(("blank", ""))
        elif _is_role(line, titles):
            lines.append(("role", line))
        elif _is_section(line):
            lines.append(("section", line))
        elif lines and lines[-1][0] == "text" and _continues(lines[-1][1], line):
            lines[-1] = ("text", f"{lines[-1][1]} {line}")
        else:
            lines.append(("text", line))
    return lines


def _sentences(line: str, max_words: int) -> list[str]:
    if len(line.split()) <= max_words:
        return [line]
    return re.split(r"(?<=[.;!?])\s+(?=[A-Z])", line)


def chunk_cv(cv_text: str, roles: list[Role] | None = None, max_words: int = 40,
             overlap_words: int = 10) -> list[str]:
    """Chunks of up to max_words, in whole lines, that never mix roles or sections. Each role's line is
    repeated on its chunks in [brackets], not counted in max_words. A role that has to split repeats its last
    line, about overlap_words. A long line is split into sentences; a longer sentence falls back to chunk_text."""
    chunks, group, header, header_used = [], [], "", False

    def add(text: str) -> None:
        nonlocal header_used
        chunks.append(f"[{header}] {text}" if header else text)
        header_used = True

    def flush() -> None:
        nonlocal group
        if group:
            add(" ".join(group))
        group = []

    def new_block(new_header: str) -> None:
        nonlocal header, header_used
        flush()
        if header and not header_used:  # a role with no lines of its own still counts as evidence
            chunks.append(f"[{header}]")
        header, header_used = new_header, False

    for kind, line in _cv_lines(cv_text, _titles(roles)):
        if kind == "blank":
            flush()
        elif kind == "role":
            new_block(line)
        elif kind == "section":
            new_block("")
            group = [line]
        else:
            for piece in _sentences(line, max_words):
                words = len(piece.split())
                if words > max_words:
                    flush()
                    for window in chunk_text(piece, max_words, overlap_words):
                        add(window)
                    continue
                if group and sum(len(g.split()) for g in group) + words > max_words:
                    last = group[-1]
                    flush()
                    if len(last.split()) + words <= max_words:
                        group = [last]
                group.append(piece)
    new_block("")
    return list(dict.fromkeys(chunks))


def _text_by_source(cv_text: str, roles: list[Role] | None) -> tuple[str, str]:
    """(what the CV says about the person, everything else), split by its headings. With no headings to go
    by, everything counts as everything else, so no rating gets lowered."""
    titles = _titles(roles)
    described, rest, in_described = [], [], False
    for kind, line in _cv_lines(cv_text, titles):
        if kind == "section":
            in_described = bool(SELF_DESCRIBED.search(line)) and not WORK_SECTION.search(line)
        elif kind == "role":
            in_described = False
        (described if in_described else rest).append(line)
    return _squash(" ".join(described)), _squash(" ".join(rest))


def list_requirements(llm: LLMProvider, result: MatchResult, max_jd_chars: int = 10_000,
                      local_max_jd_chars: int = 1_500) -> list[ListedRequirement]:
    """The posting's requirements, structured by the LLM. Every 'To build' chip is kept, so each one gets checked."""
    limit = max_jd_chars if llm.batch_jobs else local_max_jd_chars
    chips = result.missing_skills
    reply = llm.generate_json(_prompt("list_requirements.md", max_requirements=MAX_REQUIREMENTS,
                                      title=result.job.title, chips=", ".join(chips) or "none",
                                      description=result.job.description[:limit]),
                              RequirementList, what="listing the job's requirements")
    items = [i for i in reply.items if i.text.strip()]
    for item in items:
        item.chip = next((c for c in chips if c.lower() == item.chip.strip().lower()), "")
        item.kind = "basic" if item.kind.strip().lower() == "basic" else "work"
    covered = {i.chip for i in items}
    items += [ListedRequirement(text=c, chip=c) for c in chips if c not in covered]
    linked = [i for i in items if i.chip]
    return linked + [i for i in items if not i.chip][:max(0, MAX_REQUIREMENTS - len(linked))]


def retrieve_evidence(requirements: list[str], chunks: list[str], embed: EmbedFn, top_k: int = 3) -> list[list[str]]:
    """The top_k CV chunks closest to each requirement, by cosine (vectors are normalized)."""
    if not requirements or not chunks:
        return [[] for _ in requirements]
    vecs = embed(requirements + chunks)
    sims = vecs[:len(requirements)] @ vecs[len(requirements):].T  # (requirements, chunks)
    return [[chunks[i] for i in np.argsort(row)[::-1][:top_k]] for row in sims]


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def _names(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text.lower()))


def _checked_ratings(items: list[ListedRequirement], reply: AdviceReply, cv_text: str,
                     roles: list[Role] | None = None) -> dict[int, CheckedRating]:
    """id -> the rating, keeping only what the CV backs up. A quote the CV doesn't contain drops that rating,
    so it can't be shown as a match or a gap. A strong rating needs real work: on the CV's summary or skills
    list alone it counts as weak."""
    cv = _squash(cv_text)
    described, rest = _text_by_source(cv_text, roles)
    checked = {}
    for r in reply.ratings:
        if not 1 <= r.id <= len(items):
            continue
        rating, quote = r.rating.strip().lower(), r.quote.strip().lstrip("".join(BULLETS)).strip()
        if rating in ("strong", "weak") and quote and _squash(quote) in cv:
            lowered = rating == "strong" and _squash(quote) in described and _squash(quote) not in rest
            checked[r.id] = CheckedRating("weak" if lowered else rating, quote, lowered)
        elif rating == "not_shown":
            chip = items[r.id - 1].chip
            # A false gap is the worst error here: never call a chip missing when the CV names it
            checked[r.id] = (CheckedRating("weak", chip) if chip and _names(cv, chip)
                             else CheckedRating("not_shown", ""))
    return checked


def _pick(wanted: int | None, allowed: dict[int, CheckedRating], ratings: tuple[str, ...]) -> int | None:
    """The LLM's choice if it's allowed; otherwise the first in the posting's order, by the order of ratings."""
    if wanted in allowed:
        return wanted
    for rating in ratings:
        for i in sorted(allowed):
            if allowed[i].rating == rating:
                return i
    return None


def _pick_match(items: list[ListedRequirement], checked: dict[int, CheckedRating], wanted: int | None) -> int | None:
    """Only a strong rating, on the job's actual work. Years come from role dates, which no single line proves,
    and the card's Experience fit already covers them."""
    allowed = {i: c for i, c in checked.items() if c.rating == "strong" and items[i - 1].kind == "work"
               and not _names(items[i - 1].text, "years")}
    return _pick(wanted, allowed, ("strong",))


def _pick_gap(items: list[ListedRequirement], checked: dict[int, CheckedRating], wanted: int | None) -> int | None:
    """A condition like age or a driving licence is a gap only if no gap in the job's actual work is left."""
    open_gaps = {i: c for i, c in checked.items() if c.rating in ("not_shown", "weak") and not c.lowered
                 and (items[i - 1].chip or not _names(items[i - 1].text, "years"))}
    work = {i: c for i, c in open_gaps.items() if items[i - 1].kind == "work"}
    return _pick(wanted, work or open_gaps, ("not_shown", "weak"))


def _short(quote: str) -> str:
    words = quote.split()
    return quote if len(words) <= MAX_QUOTE_WORDS else " ".join(words[:MAX_QUOTE_WORDS]) + "…"


def _is_about(step: str, requirement: str) -> bool:
    words = {w for w in re.findall(r"[a-z0-9+#]+", requirement.lower()) if len(w) > 1 and w not in GENERIC_WORDS}
    return any(w in step.lower() for w in words)


def _chip_notes(items: list[ListedRequirement], checked: dict[int, CheckedRating], match: int | None,
                gap: int | None) -> list[str]:
    """Where the CV disagrees with a 'To build' chip, unless a bullet already covers that requirement. Real work
    earns a note; a weak link only if the CV names the chip, since the words could be unrelated (Java, JavaScript)."""
    notes = []
    for i, item in enumerate(items, 1):
        found = checked.get(i)
        if not item.chip or not found or i in (match, gap):
            continue
        if found.rating == "strong":
            notes.append(f'{item.chip} is listed under "To build", but your CV shows "{_short(found.quote)}"')
        elif found.rating == "weak" and _names(found.quote, item.chip):
            notes.append(f'{item.chip} is listed under "To build", and your CV mentions it. '
                         "Add a specific example of using it.")
    extra = len(notes) - MAX_NOTES
    return notes[:MAX_NOTES] + ([f'{extra} more "To build" {"item" if extra == 1 else "items"} '
                                 "may also be covered by your CV."] if extra > 0 else [])


def build_advice(items: list[ListedRequirement], reply: AdviceReply, cv_text: str,
                 roles: list[Role] | None = None, poor_fit: bool = False) -> str:
    """Strongest match, main gap, any 'To build' chip the CV disagrees with, then the next step. The requirement
    and the quote are filled in by Python, so the text can't claim what the checks rejected."""
    checked = _checked_ratings(items, reply, cv_text, roles)
    match = _pick_match(items, checked, reply.match_id)
    gap = _pick_gap(items, checked, reply.gap_id)

    def wanted(i: int) -> str:
        return items[i - 1].text.strip().rstrip(".;:")

    first = (f'{wanted(match)}. Your CV shows: "{_short(checked[match].quote)}"' if match
             else "No strong match found between this posting and your CV.")
    if gap is None:
        second, step = "No clear gap found between this posting and your CV.", ""
    elif checked[gap].rating == "weak":
        second = f'{wanted(gap)}. Your CV only shows a weak link: "{_short(checked[gap].quote)}"'
        step = f"Add a specific example to your CV if you have done this, or start building it: {wanted(gap)}."
    else:
        second = f"{wanted(gap)}. Your CV doesn't show this."
        step = f"Add a specific example to your CV if you have done this, or start building it: {wanted(gap)}."
    # The LLM's step only stands if it's about the gap shown here
    written = reply.next_step.strip()
    if written and reply.gap_id == gap and (gap is None or _is_about(written, wanted(gap))):
        step = written
    if poor_fit:  # a score of near zero and a step that polishes the CV for this job would disagree
        step = "This role is far from your background, so focus on roles closer to your field."
    step = step or "Lead your application with your strongest match, using a specific result from your CV."

    notes = [f"- **May not be a real gap:** {note}" for note in _chip_notes(items, checked, match, gap)]
    text = "\n".join([f"- **Strongest match:** {first}", f"- **Main gap:** {second}", *notes,
                      f"- **Most useful next step:** {step}"])
    return text.replace("$", r"\$")  # a pair of dollar signs would render as a math formula


def generate_advice(llm: LLMProvider, cv_text: str, profile: CandidateProfile, result: MatchResult, embed: EmbedFn,
                    max_jd_chars: int = 10_000, local_max_jd_chars: int = 1_500) -> str:
    items = list_requirements(llm, result, max_jd_chars, local_max_jd_chars)
    if not items:
        return "We couldn't find clear requirements in this posting. Open the posting to read it directly."
    try:
        evidence = retrieve_evidence([i.text for i in items], chunk_cv(cv_text, profile.roles), embed)
    except LLMError as e:  # the full CV still goes to the LLM, so the advice can go ahead without retrieval
        log.warning("advice continues without retrieval: %s", e)
        evidence = [[] for _ in items]
    blocks = "\n\n".join(
        f"[{n}] ({item.kind}) {item.text}\nClosest CV lines:\n" + ("\n".join(f"- {c}" for c in found)
                                                                   or "- (none retrieved)")
        for n, (item, found) in enumerate(zip(items, evidence), 1))
    reply = llm.generate_json(_prompt("write_advice.md", title=result.job.title, company=result.job.company,
                                      requirements=blocks, cv_text=cv_text[:llm.max_cv_chars or len(cv_text)]),
                              AdviceReply, what="writing advice")
    poor_fit = result.field_match == "different" and result.work_match == "different"
    return build_advice(items, reply, cv_text, profile.roles, poor_fit)
