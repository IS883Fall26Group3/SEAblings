"""Fictional sample CVs and job postings, so anyone can try the app without their own files.
Every person and company in them is made up (data/sample/)."""
from pathlib import Path

from .job_sources.sample import SampleSource
from .schemas import Job

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"

# Dropdown label -> CV file. The first is the default.
SAMPLE_CVS = {
    "Alex Rivera: corporate planning, energy": SAMPLE / "sample_cv.txt",
    "Jordan Lee: FP&A, software": SAMPLE / "cvs" / "finance_jordan_lee.txt",
    "Priya Raman: marketing and consulting": SAMPLE / "cvs" / "marketing_priya_raman.txt",
    "Diego Morales: supply chain, manufacturing": SAMPLE / "cvs" / "operations_diego_morales.txt",
    "Mei Chen: product analytics, fintech": SAMPLE / "cvs" / "product_mei_chen.txt",
}


def sample_cv(label: str) -> str:
    return SAMPLE_CVS.get(label, next(iter(SAMPLE_CVS.values()))).read_text(encoding="utf-8")


def sample_jobs() -> list[Job]:
    """All sample postings, in file order."""
    return SampleSource().search("", "", 100)
