"""Shared data contracts. Agree on these first; every module codes against them."""
from typing import Optional
from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    summary: str = Field(description="2-3 sentence professional summary")
    years_experience: float = Field(description="Total years of professional experience")
    seniority: str = Field(description="One of: entry, mid, senior, lead")
    skills: list[str] = Field(description="Hard and soft skills, each 1-4 words, deduplicated")
    job_titles: list[str] = Field(description="Past job titles, most recent first")


class JobRequirements(BaseModel):
    job_id: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_years_experience: Optional[float] = None


class JobRequirementsBatch(BaseModel):
    items: list[JobRequirements]


class Job(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str = ""
    source: str = ""          # e.g. "Indeed", "LinkedIn" (publisher reported by the API)
    employment_type: str = ""
    posted_at: str = ""
    requirements: Optional[JobRequirements] = None


class MatchResult(BaseModel):
    job: Job
    score: float              # 0-100
    skill_coverage: float     # 0-1
    semantic_similarity: float  # 0-1
    seniority_fit: float      # 0-1
    matched_skills: list[str]
    missing_skills: list[str]
