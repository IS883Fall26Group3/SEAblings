"""Profile similarity between the CV and each job, three ways (PROFILE_SIMILARITY):
  summary   cosine of the CV summary and the job's role_summary, both written by the LLM during
            extraction (no extra calls; shortest texts, no company pitch or legal boilerplate)
  chunks    the CV split into overlapping chunks; each job retrieves its k closest chunks and the
            score is their mean cosine (Session 5 chunking + retrieval: no "blurry average")
  document  cosine of the whole CV and the whole posting, each cut to max_chars (the original)
Each mode's cosines fall in a different range, so the embedder keeps one band per mode."""
from typing import Callable

import numpy as np

from .schemas import CandidateProfile, Job

EmbedFn = Callable[[list[str]], np.ndarray]
MODES = ("summary", "chunks", "document")


def chunk_text(text: str, words_per_chunk: int = 40, overlap: int = 10) -> list[str]:
    """Overlapping chunks of about words_per_chunk words (as in the Session 5 notebook)."""
    words = text.split()
    step = max(1, words_per_chunk - overlap)
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, max(len(words) - overlap, 1), step)]


def profile_cosines(mode: str, profile: CandidateProfile, cv_text: str, jobs: list[Job], embed: EmbedFn,
                    max_chars: int = 7_000, top_k: int = 3) -> list[tuple[float, str]]:
    """Raw cosine per job and the mode actually used for it. 'summary' falls back to 'document'
    for a job without a role_summary (or a CV without a summary), so it never scores blank text."""
    if mode not in MODES:
        raise ValueError(f"Unknown PROFILE_SIMILARITY: {mode}")
    docs = [j.description[:max_chars] for j in jobs]
    if mode == "chunks":
        chunks = chunk_text(cv_text[:4 * max_chars])
        vecs = embed(chunks + docs)
        sims = vecs[len(chunks):] @ vecs[:len(chunks)].T  # (jobs, chunks)
        k = min(top_k, len(chunks))
        return [(float(np.sort(row)[-k:].mean()), "chunks") for row in sims]

    use_summary = [mode == "summary" and bool(profile.summary.strip()) and bool(
        j.requirements and j.requirements.role_summary.strip()) for j in jobs]
    texts = [cv_text[:max_chars]] if not all(use_summary) else []
    if any(use_summary):
        texts.append(profile.summary)
    texts += [j.requirements.role_summary if u else d for j, d, u in zip(jobs, docs, use_summary)]
    vecs = embed(texts)
    cv_doc = vecs[0] if not all(use_summary) else None
    cv_sum = vecs[1 if cv_doc is not None else 0] if any(use_summary) else None
    job_vecs = vecs[len(texts) - len(jobs):]
    return [(float(jv @ cv_sum), "summary") if u else (float(jv @ cv_doc), "document")
            for jv, u in zip(job_vecs, use_summary)]
