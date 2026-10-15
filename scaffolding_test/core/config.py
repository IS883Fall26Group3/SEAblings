"""Central settings. Reads environment variables only, so core/ never depends on Streamlit.
streamlit_app.py copies st.secrets into the environment at startup."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    rapidapi_key: str
    job_source: str
    # Model names change over time; confirm current names in Google AI Studio.
    gemini_model: str
    gemini_embed_model: str
    llm_timeout_s: int = 60
    max_cv_chars: int = 15_000
    max_jd_chars: int = 3_000
    default_num_jobs: int = 10
    # Scoring weights (must sum to 1.0). Tune with your evaluation set.
    w_skill_coverage: float = 0.5
    w_semantic: float = 0.3
    w_seniority: float = 0.2
    skill_match_threshold: float = 0.80  # cosine similarity for "same skill"


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        gemini_api_key=env("GEMINI_API_KEY", ""),
        rapidapi_key=env("RAPIDAPI_KEY", ""),
        job_source=env("JOB_SOURCE", "sample"),
        gemini_model=env("GEMINI_MODEL", "gemini-flash-latest"),
        gemini_embed_model=env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
    )
