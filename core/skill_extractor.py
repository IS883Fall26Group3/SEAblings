"""LLM-based extraction of skills from the CV and from job descriptions."""
from pathlib import Path

from .llm.base import LLMProvider
from .schemas import (CandidateProfile, Job, JobRequirements, JobRequirementsBatch, RelevantExperience,
                      RelevantExperienceBatch)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str, **kwargs) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").format(**kwargs)


def extract_profile(llm: LLMProvider, cv_text: str) -> CandidateProfile:
    if llm.max_cv_chars:
        cv_text = cv_text[:llm.max_cv_chars]
    return llm.generate_json(_prompt("extract_profile.md", cv_text=cv_text), CandidateProfile,
                             what="extracting skills from your CV")


def extract_requirements(llm: LLMProvider, jobs: list[Job], max_jd_chars: int = 10_000,
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


def judge_experience(llm: LLMProvider, cv_text: str, jobs: list[Job], max_jd_chars: int = 10_000,
                     local_max_jd_chars: int = 1_500) -> dict[str, RelevantExperience]:
    """Technique B: the LLM picks the CV roles relevant to each job, following guided steps and
    using only the CV. One call for all jobs on Gemini; one per job on small local models.
    Reads as much of each posting as extract_requirements, so it sees the requirements section."""
    if llm.max_cv_chars:
        cv_text = cv_text[:llm.max_cv_chars]
    if not llm.batch_jobs:
        max_jd_chars = local_max_jd_chars  # small local models: shorter input, more reliable

    def block(j: Job) -> str:
        req = j.requirements
        return (f"### job_id: {j.id}\nTitle: {j.title}\n"
                f"Required skills: {', '.join(req.required_skills) if req else 'not extracted'}\n"
                f"Minimum years: {req.min_years_experience if req and req.min_years_experience else 'not stated'}\n"
                f"{j.description[:max_jd_chars]}")

    groups = [jobs] if llm.batch_jobs else [[j] for j in jobs]
    out = {}
    for group in groups:
        batch = llm.generate_json(_prompt("judge_experience.md", cv_text=cv_text,
                                          jobs="\n\n".join(block(j) for j in group)),
                                  RelevantExperienceBatch, what="judging your relevant experience")
        items = batch.items
        if len(group) == 1 and len(items) == 1:
            items[0].job_id = group[0].id  # small models sometimes garble the id of a single job
        out |= {i.job_id: i for i in items}
    return out
