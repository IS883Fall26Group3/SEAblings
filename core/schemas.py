"""Shared data contracts. Agree on these first; every module codes against them."""
from typing import Optional
from pydantic import BaseModel, Field


class Role(BaseModel):
    """One position from the CV. Dates are copied, never computed: Python adds up the years."""
    title: str = Field(description="Position title as written in the CV")
    summary: str = Field(default="", description="One sentence: the role's main work and domain")
    start: str = Field(default="", description="Start month as YYYY-MM (YYYY if only the year is given), "
                                               "or empty if the CV gives no date")
    end: str = Field(default="", description="End month as YYYY-MM, 'present' if current, or empty")
    duration_months: Optional[float] = Field(default=None, description="Duration in months only when the "
                                             "CV states a duration instead of dates, e.g. '(11 months)'")


class CandidateProfile(BaseModel):
    summary: str = Field(description="2-3 sentence professional summary")
    years_experience: float = Field(description="Total years of professional experience")
    seniority: str = Field(description="One of: entry, mid, senior, lead")
    skills: list[str] = Field(description="Hard and soft skills, each 1-4 words, deduplicated")
    job_titles: list[str] = Field(description="Past job titles, most recent first")
    roles: list[Role] = Field(default_factory=list, description="Every professional position, most recent "
                              "first; one entry per position, not per company")
    # Shown beside the job's own labels to explain profile similarity; not used in any score
    field: str = Field(default="", description="Industry or domain of the work, 2-5 words")
    work_type: str = Field(default="", description="Main kind of work, 2-5 words")


class JobRequirements(BaseModel):
    job_id: str = ""
    required_skills: list[str]
    nice_to_have_skills: list[str]
    min_years_experience: Optional[float] = None
    role_summary: str = Field(default="", description="2-3 sentences describing the ideal candidate the "
                              "way a CV summary would: main kind of work, domain, and seniority")
    # Used to name jobs the user pastes or uploads (core/job_sources/manual.py)
    job_title: str = Field(default="", description="The role's title as the posting states it")
    company: str = Field(default="", description="The hiring company's name, or empty if not stated")
    # Shown beside the CV's labels to explain profile similarity; not used in any score
    field: str = Field(default="", description="Industry or domain of the role, 2-5 words")
    work_type: str = Field(default="", description="Main kind of work in the role, 2-5 words")
    level: str = Field(default="", description="One of: entry, mid, senior, lead")


class JobRequirementsBatch(BaseModel):
    items: list[JobRequirements]


class RelevantExperience(BaseModel):
    """The LLM judge's view of which CV roles count toward one job (reasoning comes first)."""
    job_id: str = ""
    reasoning: str = Field(default="", description="One or two sentences: what experience the job needs and "
                                                   "which of the candidate's roles provide it")
    # The candidate's whole background vs the job; adjusts profile similarity (core/matcher.py)
    field_match: str = Field(default="", description="Industry or domain: same, related or different")
    work_match: str = Field(default="", description="Kind of work: same, related or different")
    relevant_roles: list[Role] = Field(default_factory=list, description="The candidate's roles that count "
                                       "toward this job's required experience, dates copied from the CV")


class RelevantExperienceBatch(BaseModel):
    items: list[RelevantExperience]


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
    # What went into the scores, so the card can explain them (core/score_notes.py)
    close_matches: dict[str, str] = {}  # job skill -> the CV skill that covered it, when not the same name
    years_counted: float = 0.0          # years compared with the posting's minimum
    total_years: float = 0.0
    min_years: Optional[float] = None
    relevant_only: bool = False         # True when only roles relevant to the job were counted
    counted_roles: list[Role] = []      # those roles
    embedding_similarity: float = 0.0   # profile similarity from the embedding alone, before the rating
    field_match: str = ""               # the judge's same / related / different, empty when not rated
    work_match: str = ""
