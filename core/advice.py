"""Advice for one job, on demand: strongest match, main gap, next step. Each requirement in the posting
retrieves its closest CV chunks (Session 5 RAG), the LLM rates it, and Python checks the LLM's quotes
against the CV. The two LLM calls run only on click; nothing here changes a score or a chip."""
import logging
import re
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, Field

from .llm.base import LLMError, LLMProvider
from .schemas import CandidateProfile, MatchResult, Role
from .similarity import chunk_text
from .skill_extractor import _prompt

log = logging.getLogger("seablings.llm")

EmbedFn = Callable[[list[str]], np.ndarray]
MAX_REQUIREMENTS = 15
SECTION = re.compile(r"summary|profile|objective|experience|employment|work history|education|skills|projects"
                     r"|certifications?|languages|awards|publications|activities|volunteer\w*|interests", re.I)
BULLETS = ("-", "*", "•", "–")


class ListedRequirement(BaseModel):
    text: str = Field(description="One requirement, in the posting's own words")
    chip: str = Field(default="", description="The item from the 'To build' list this covers, copied "
                                              "exactly, or empty")


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
    next_step: str = Field(default="", description="One sentence: the single most useful step")


def _word_count(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def _is_role(line: str, titles: list[str]) -> bool:
    return (not line.startswith(BULLETS) and len(line.split()) <= 15
            and any(title in line.lower() for title in titles))


def _is_section(line: str) -> bool:
    text = line.strip(" :")
    return len(text.split()) <= 4 and (text.isupper() or bool(SECTION.search(text)))


def chunk_cv(cv_text: str, roles: list[Role] | None = None, max_words: int = 40,
             overlap_words: int = 10) -> list[str]:
    """Chunks of up to max_words, in whole lines, that never mix roles or sections. Each role's line is
    repeated on its chunks (not counted in max_words). A role that has to split repeats its last line,
    about overlap_words. A longer line, as in a CV pasted without line breaks, falls back to chunk_text."""
    titles = [r.title.strip().lower() for r in roles or [] if r.title.strip()]
    chunks, group, header, header_used = [], [], "", False

    def add(text: str) -> None:
        nonlocal header_used
        chunks.append(f"{header} | {text}" if header else text)
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
            chunks.append(header)
        header, header_used = new_header, False

    for raw in cv_text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif _is_role(line, titles):
            new_block(line)
        elif _is_section(line):
            new_block("")
            group = [line]
        elif len(line.split()) > max_words:
            flush()
            for piece in chunk_text(line, max_words, overlap_words):
                add(piece)
        else:
            if group and _word_count(group) + len(line.split()) > max_words:
                last = group[-1]
                flush()
                if len(last.split()) + len(line.split()) <= max_words:
                    group = [last]
            group.append(line)
    new_block("")
    return list(dict.fromkeys(chunks))


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


def _checked_ratings(items: list[ListedRequirement], reply: AdviceReply, cv_text: str) -> dict[int, tuple[str, str]]:
    """id -> (rating, quote), keeping only what the CV backs up. A quote the CV doesn't contain drops
    that rating, so it can't be shown as a match or a gap."""
    cv = _squash(cv_text)
    checked = {}
    for r in reply.ratings:
        if not 1 <= r.id <= len(items):
            continue
        rating, quote = r.rating.strip().lower(), r.quote.strip().lstrip("".join(BULLETS)).strip()
        if rating in ("strong", "weak") and quote and _squash(quote) in cv:
            checked[r.id] = (rating, quote)
        elif rating == "not_shown":
            chip = items[r.id - 1].chip
            # A false gap is the worst error here: never call a chip missing when the CV names it
            named = chip and re.search(rf"(?<!\w){re.escape(chip.lower())}(?!\w)", cv)
            checked[r.id] = ("weak", chip) if named else ("not_shown", "")
    return checked


def _pick(wanted: int | None, ratings: tuple[str, ...], checked: dict[int, tuple[str, str]]) -> int | None:
    """The LLM's choice if it holds up; otherwise the first in the posting's order, by the order of ratings."""
    if wanted in checked and checked[wanted][0] in ratings:
        return wanted
    for rating in ratings:
        for i in sorted(checked):
            if checked[i][0] == rating:
                return i
    return None


def build_advice(items: list[ListedRequirement], reply: AdviceReply, cv_text: str) -> str:
    """Three bullets, then a note on any 'To build' chip the CV disagrees with. The requirement and the
    quote are filled in by Python, so the text can't claim what the check rejected."""
    checked = _checked_ratings(items, reply, cv_text)
    match = _pick(reply.match_id, ("strong",), checked)
    gap = _pick(reply.gap_id, ("not_shown", "weak"), checked)

    first = (f'{items[match - 1].text}. Your CV shows: "{checked[match][1]}"' if match
             else "No strong match found between this posting and your CV.")
    if gap is None:
        second, step = "No clear gap found between this posting and your CV.", ""
    else:
        rating, quote = checked[gap]
        wanted = items[gap - 1].text
        second = (f'{wanted}. Your CV only shows a weak link: "{quote}"' if rating == "weak"
                  else f"{wanted}. Your CV doesn't show this.")
        step = f"If you have done this, add a specific example of it to your CV; if not, start building it: {wanted}."
    # The LLM's step only stands if it was written for the gap shown here
    if reply.next_step.strip() and reply.gap_id == gap:
        step = reply.next_step.strip()
    step = step or "Lead your application with your strongest match, using a specific result from your CV."

    notes = []
    for i, item in enumerate(items, 1):
        rating, quote = checked.get(i, ("", ""))
        if item.chip and rating and rating != "not_shown" and not (rating == "weak" and i == gap):
            notes.append(f'"To build" lists {item.chip}, but your CV shows "{quote}", so this may not be a real gap.')
    text = (f"- **Strongest match:** {first}\n- **Main gap:** {second}\n- **Most useful next step:** {step}")
    if notes:
        text += "\n\n" + "  \n".join(notes)
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
        f"[{n}] {item.text}\nClosest CV lines:\n" + ("\n".join(f"- {c}" for c in found) or "- (none retrieved)")
        for n, (item, found) in enumerate(zip(items, evidence), 1))
    reply = llm.generate_json(_prompt("write_advice.md", title=result.job.title, company=result.job.company,
                                      requirements=blocks, cv_text=cv_text[:llm.max_cv_chars or len(cv_text)]),
                              AdviceReply, what="writing advice")
    return build_advice(items, reply, cv_text)
