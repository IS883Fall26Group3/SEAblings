"""LLM-based extraction of skills from the CV and from job descriptions."""
from pathlib import Path

from .llm.base import LLMProvider
from .schemas import CandidateProfile, Job, JobRequirements, JobRequirementsBatch

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str, **kwargs) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").format(**kwargs)


def extract_profile(llm: LLMProvider, cv_text: str) -> CandidateProfile:
    if llm.max_cv_chars:
        cv_text = cv_text[:llm.max_cv_chars]
    return llm.generate_json(_prompt("extract_profile.md", cv_text=cv_text), CandidateProfile,
                             what="extracting skills from your CV")


def extract_requirements(llm: LLMProvider, jobs: list[Job], max_jd_chars: int = 3_000,
                         local_max_jd_chars: int = 1_500) -> list[Job]:
    if not jobs:
        return jobs
    if llm.batch_jobs:
        # One call for all jobs: saves free-tier quota versus one call per job.
        blocks = "\n\n".join(
            f"### job_id: {j.id}\nTitle: {j.title}\n{j.description[:max_jd_chars]}" for j in jobs)
        batch = llm.generate_json(_prompt("extract_requirements.md", jobs=blocks),
                                  JobRequirementsBatch, what="reading the job descriptions")
        by_id = {r.job_id: r for r in batch.items}
        for j in jobs:
            j.requirements = by_id.get(j.id)
    else:
        # Small local models: one shorter job at a time is slower but far more reliable.
        for j in jobs:
            req = llm.generate_json(
                _prompt("extract_requirements_single.md", title=j.title,
                        description=j.description[:local_max_jd_chars]),
                JobRequirements, what=f"reading the posting for {j.title}")
            j.requirements = req.model_copy(update={"job_id": j.id})
    return jobs


def explain_match(llm: LLMProvider, profile: CandidateProfile, result) -> str:
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
