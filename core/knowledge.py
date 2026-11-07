"""Skills knowledge base: the corpus the app retrieves from (built by scripts/build_skill_kb.py).

Each entry is a family of related skills ("Data analysis and business intelligence: Tableau, Power BI,
Looker, ..."). For every job requirement the app retrieves the closest entries by embedding similarity
and puts them in the prompt that labels the requirement met / partial / missing (core/requirements.py),
so the LLM knows which CV skills are related enough to count as a partial match.

The entries are embedded once, ahead of time, and saved per embedding model in data/skill_kb_vectors/,
with a hash of the texts so a stale file is never used. Without a saved file they are embedded at
runtime and kept in memory."""
import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("seablings.knowledge")

DATA = Path(__file__).resolve().parent.parent / "data"
KB_CSV = DATA / "skill_kb.csv"
VECTORS = DATA / "skill_kb_vectors"


@dataclass(frozen=True)
class KBEntry:
    id: str
    family: str
    description: str
    skills: tuple[str, ...]
    source: str  # "hand" (team) or "onet"


def load_kb(path: Path = KB_CSV) -> list[KBEntry]:
    with path.open(encoding="utf-8") as f:
        return [KBEntry(r["id"], r["family"], r["description"],
                        tuple(s.strip() for s in r["skills"].split(";") if s.strip()), r["source"])
                for r in csv.DictReader(f)]


def entry_text(e: KBEntry) -> str:
    """What is embedded, and what the labeling prompt shows."""
    return f"{e.family}: {', '.join(e.skills)}"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in name)


def _digest(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def save_vectors(embedder_name: str, texts: list[str], vecs: np.ndarray) -> Path:
    VECTORS.mkdir(exist_ok=True)
    path = VECTORS / f"{_slug(embedder_name)}.npy"
    np.save(path, vecs.astype(np.float16))  # half precision: a third of the size, same ranking
    path.with_suffix(".sha256").write_text(_digest(texts), encoding="utf-8")
    return path


def _saved_vectors(embedder_name: str, texts: list[str]) -> np.ndarray | None:
    path = VECTORS / f"{_slug(embedder_name)}.npy"
    sha = path.with_suffix(".sha256")
    if not path.exists() or not sha.exists() or sha.read_text(encoding="utf-8").strip() != _digest(texts):
        return None
    vecs = np.load(path).astype(np.float32)
    return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)


class SkillKB:
    def __init__(self, entries: list[KBEntry], vectors: np.ndarray):
        self.entries, self.vectors = entries, vectors

    @classmethod
    def build(cls, embedder_name: str, embed: Callable[[list[str]], np.ndarray],
              entries: list[KBEntry] | None = None) -> "SkillKB":
        entries = load_kb() if entries is None else entries
        texts = [entry_text(e) for e in entries]
        vecs = _saved_vectors(embedder_name, texts)
        if vecs is None:
            log.info("no saved knowledge-base vectors for %s; embedding %d entries", embedder_name, len(texts))
            vecs = embed(texts)
        return cls(entries, vecs)

    def search(self, query_vecs: np.ndarray, k: int = 2) -> list[list[KBEntry]]:
        """The k closest entries for each query vector (rows of query_vecs)."""
        if not len(self.entries) or not len(query_vecs):
            return [[] for _ in range(len(query_vecs))]
        sims = query_vecs @ self.vectors.T
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[self.entries[i] for i in row] for row in top]
