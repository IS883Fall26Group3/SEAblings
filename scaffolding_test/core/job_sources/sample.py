"""Offline source backed by data/sample/jobs.json. Use for development, tests, and as a
fallback so a live demo never depends on API quota."""
import json
from pathlib import Path

from ..schemas import Job
from .base import JobSource

DATA = Path(__file__).resolve().parents[2] / "data" / "sample" / "jobs.json"


class SampleSource(JobSource):
    name = "sample"

    def search(self, query: str, location: str, limit: int = 10) -> list[Job]:
        jobs = [Job(**j) for j in json.loads(DATA.read_text(encoding="utf-8"))]
        terms = [t for t in query.lower().split() if len(t) > 2]
        ranked = sorted(jobs, key=lambda j: -sum(t in (j.title + j.description).lower() for t in terms))
        return ranked[:limit]
