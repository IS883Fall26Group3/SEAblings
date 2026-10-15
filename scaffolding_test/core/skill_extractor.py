"""LLM-based extraction of skills from the CV and from job descriptions."""
from pathlib import Path

from .llm_client import LLMClient
from .schemas import CandidateProfile, Job, JobRequirementsBatch

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str, **kwargs) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").format(**kwargs)


def extract_profile(llm: LLMClient, cv_text: str) -> CandidateProfile:
    return llm.generate_json(_prompt("extract_profile.md", cv_text=cv_text), CandidateProfile,
                             what="extracting skills from your CV")


def extract_requirements(llm: LLMClient, jobs: list[Job], max_jd_chars: int = 3_000) -> list[Job]:
    """One batched LLM call for all jobs (saves free-tier quota versus one call per job)."""
    if not jobs:
        return jobs
    blocks = "\n\n".join(
        f"### job_id: {j.id}\nTitle: {j.title}\n{j.description[:max_jd_chars]}" for j in jobs
    )
    batch = llm.generate_json(_prompt("extract_requirements.md", jobs=blocks), JobRequirementsBatch,
                              what="reading the job descriptions")
    by_id = {r.job_id: r for r in batch.items}
    for j in jobs:
        j.requirements = by_id.get(j.id)
    return jobs


def explain_match(llm: LLMClient, profile: CandidateProfile, result) -> str:
    """Optional per-job explanation, generated only when the user asks (saves quota)."""
    return llm.generate_text(_prompt(
        "explain_match.md",
        summary=profile.summary,
        skills=", ".join(profile.skills),
        title=result.job.title,
        company=result.job.company,
        matched=", ".join(result.matched_skills) or "none",
        missing=", ".join(result.missing_skills) or "none",
    ))
