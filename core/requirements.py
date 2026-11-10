"""Retrieval for per-requirement labels: met, partial or missing, with a quote from the CV.

For each required skill of each job, two kinds of text are retrieved by embedding similarity:
  - the k CV lines closest to the requirement (the candidate's evidence), and
  - the k closest families in the skills knowledge base (core/knowledge.py), which tell the LLM
    which other skills are related enough to count as partial.
Only those retrieved lines, never the whole CV, go into the labeling prompt
(prompts/label_requirements.md, run by core/skill_extractor.py: label_requirements). The LLM's
quotes are then checked word for word against the CV (quote_in_cv): a deterministic check."""
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .knowledge import KBEntry, SkillKB
from .schemas import Job, RequirementCheck, RequirementLabel

EmbedFn = Callable[[list[str]], np.ndarray]
STATUSES = ("met", "partial", "missing")

_BULLET = re.compile(r"^[\s•●▪◦·*>\-–—]+")
_SENTENCE = re.compile(r"(?<=[.;])\s+")


@dataclass
class Retrieved:
    requirement: str
    lines: list[str] = field(default_factory=list)       # closest CV lines, best first
    related: list[KBEntry] = field(default_factory=list)  # closest knowledge-base families


def norm(s: str) -> str:
    """Same key as core/matcher.py, so a memoized embed function reuses the skill vectors."""
    return s.strip().lower()


def cv_lines(cv_text: str, max_lines: int = 80, max_words: int = 40) -> list[str]:
    """The CV as short lines to retrieve from: one per line or bullet, long paragraphs split into
    sentences, section headings ("EXPERIENCE") and repeats dropped."""
    out: list[str] = []
    for raw in cv_text.splitlines():
        line = _BULLET.sub("", raw).strip()
        words = line.split()
        if not words or (line.isupper() and len(words) <= 3):
            continue
        parts = _SENTENCE.split(line) if len(words) > max_words else [line]
        for part in parts:
            w = part.split()
            out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
    return list(dict.fromkeys(x for x in out if x))[:max_lines]


def normalize(text: str) -> str:
    t = text.lower().translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    t = re.sub(r"[•●▪◦·]", " ", t)
    return re.sub(r"\s+", " ", t).strip(" .;,:")


def quote_in_cv(quote: str, cv_text: str) -> bool:
    """True if the quote appears word for word in the CV (ignoring case, spacing and bullet marks)."""
    q = normalize(quote)
    return len(q) >= 3 and q in normalize(cv_text)


def required_skills(job: Job) -> list[str]:
    req = job.requirements
    return [s for s in (req.required_skills if req else []) if s.strip()]


def _words(text: str) -> str:
    """Lowercase words with spaces around them, for whole-word matching ("r" doesn't match "rigs")."""
    return " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip() + " "


def _terms(requirement: str) -> list[str]:
    """The names a requirement goes by: "GenAI / LLMs" -> ["genai / llms", "genai", "llms"]."""
    parts = re.split(r"\s*(?:/|,|;|\band\b|\bor\b)\s*", requirement, flags=re.I)
    return [t for t in dict.fromkeys([requirement, *parts]) if _words(t).strip()]


def _naming(lines_w: list[str], terms: list[str]) -> list[int]:
    """Indexes of the CV lines that name any of the terms as whole words."""
    tw = [_words(t) for t in terms]
    return [i for i, lw in enumerate(lines_w) if any(t in lw for t in tw)]


def retrieve_evidence(jobs: list[Job], cv_text: str, embed: EmbedFn, kb: SkillKB | None,
                      k_lines: int = 3, k_kb: int = 2) -> dict[str, list[Retrieved]]:
    """One embedding call for the CV lines and every distinct requirement, then, for each requirement:
    the closest knowledge-base families, and the CV lines to show the LLM. Those are, in order: up to
    2 lines that name the requirement itself, 1 line that names a related skill from those families,
    then the k_lines closest lines by embedding. The word matches come first because embedding search
    alone misses short names: a long skills line that lists "Python" among ten tools is often not
    among the closest lines to "Python"."""
    queries = list(dict.fromkeys(norm(s) for j in jobs for s in required_skills(j)))
    if not queries:
        return {j.id: [] for j in jobs}
    lines = cv_lines(cv_text)
    vecs = embed(lines + queries)
    line_vecs, q_vecs = vecs[:len(lines)], vecs[len(lines):]
    near_kb = kb.search(q_vecs, k_kb) if kb else [[] for _ in queries]
    near_lines = [[] for _ in queries]
    if lines:
        lines_w = [_words(x) for x in lines]
        order = np.argsort(-(q_vecs @ line_vecs.T), axis=1)[:, :k_lines]
        for qi, q in enumerate(queries):
            named = _naming(lines_w, _terms(q))[:2]
            related = [s for e in near_kb[qi] for s in e.skills if norm(s) != q]
            near = [i for i in _naming(lines_w, related) if i not in named][:1]
            near_lines[qi] = [lines[i] for i in dict.fromkeys([*named, *near, *order[qi]])]
    found = {q: (ln, kbe) for q, ln, kbe in zip(queries, near_lines, near_kb)}
    return {j.id: [Retrieved(s, *found[norm(s)]) for s in required_skills(j)] for j in jobs}


def mention(r: Retrieved) -> str:
    """The first retrieved CV line that names the requirement, or "". Used on a requirement the LLM
    labeled missing: a line naming it means the label may be wrong, so the card says it may not be a
    real gap."""
    named = _naming([_words(x) for x in r.lines], _terms(r.requirement))
    return r.lines[named[0]] if named else ""


def to_checks(retrieved: list[Retrieved], labels: list[RequirementLabel], cv_text: str) -> list[RequirementCheck]:
    """Match the LLM's labels to the job's requirements and verify each quote. A requirement the LLM
    skipped, or labeled with an unknown status, is left out: the skill matcher decides it instead.
    A missing requirement keeps the CV line that names it, if one was retrieved (mention)."""
    by_req = {norm(x.requirement): x for x in labels}
    out = []
    for r in retrieved:
        label = by_req.get(norm(r.requirement))
        status = label.status.strip().lower() if label else ""
        if status not in STATUSES:
            continue
        quote = label.quote.strip().strip('"').strip() if status != "missing" else ""
        verified = bool(quote) and quote_in_cv(quote, cv_text)
        out.append(RequirementCheck(requirement=r.requirement, status=status,
                                    quote=quote if verified else "", quote_verified=verified,
                                    reason=label.reason.strip(),
                                    mention=mention(r) if status == "missing" else ""))
    return out


def job_block(job: Job, retrieved: list[Retrieved]) -> str:
    """One job's requirements with the text retrieved for each, as the labeling prompt shows them."""
    out = [f"### job_id: {job.id}", f"Title: {job.title}"]
    for r in retrieved:
        out.append(f"- Requirement: {r.requirement}")
        out += [f'    CV line: "{line}"' for line in r.lines] or ["    CV lines: none found"]
        if r.related:
            out.append("    Related skills (knowledge base): "
                       + " | ".join(f"{e.family}: {', '.join(e.skills)}" for e in r.related))
    return "\n".join(out)
