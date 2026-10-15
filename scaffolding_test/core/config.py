"""Central settings. Reads environment variables only, so core/ never depends on Streamlit.
streamlit_app.py copies st.secrets into the environment at startup."""
import os
from dataclasses import dataclass


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    rapidapi_key: str
    job_source: str
    # Gemini models offered in the sidebar. Each model has its own free-tier daily quota.
    # Model names change over time; confirm current names in Google AI Studio.
    gemini_models: list[str]
    gemini_embed_model: str
    # Local (offline) Hugging Face models; used only if torch + transformers are installed.
    local_llm_model: str
    local_llm_dtype: str
    local_embed_model: str
    embeddings: str              # "auto" (local if installed, else Gemini), "local", or "gemini"
    llm_timeout_s: int = 60
    max_cv_chars: int = 15_000
    max_jd_chars: int = 3_000
    local_max_jd_chars: int = 1_500  # small local models are slow on long inputs
    default_num_jobs: int = 10
    # Scoring weights (must sum to 1.0). Tune with your evaluation set.
    w_skill_coverage: float = 0.5
    w_semantic: float = 0.3
    w_seniority: float = 0.2


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        gemini_api_key=env("GEMINI_API_KEY", ""),
        rapidapi_key=env("RAPIDAPI_KEY", ""),
        job_source=env("JOB_SOURCE", "sample"),
        gemini_models=_list(env("GEMINI_MODELS", env("GEMINI_MODEL",
                                "gemini-flash-latest,gemini-flash-lite-latest"))),
        gemini_embed_model=env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        local_llm_model=env("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        local_llm_dtype=env("LOCAL_LLM_DTYPE", "bfloat16"),
        local_embed_model=env("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embeddings=env("EMBEDDINGS", "auto"),
    )
