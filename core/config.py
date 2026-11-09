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
    jsearch_url: str             # copy from the RapidAPI playground's code snippet if it changes
    # Gemini models offered in the sidebar. Each model has its own free-tier daily quota.
    # Model names change over time; confirm current names in Google AI Studio.
    gemini_models: list[str]
    gemini_embed_model: str
    # Offline GGUF models from Hugging Face, run with llama-cpp-python (no API, no quota).
    local_llm_repo: str
    local_llm_file: str          # filename or glob within the repo (case-sensitive)
    local_embed_repo: str
    local_embed_file: str
    embeddings: str              # "auto" (Gemini if a key is set, else local), "local", or "gemini"
    # Experience fit: "judge" (years in roles the LLM judges relevant; one extra LLM call per search,
    # best in eval/README.md step 5), "roles" (years in roles similar to the job, by embedding)
    # or "total" (all years). Any other value counts all years, like "total".
    experience_fit: str = "judge"
    # Profile similarity: "document" (whole CV vs whole posting), "summary" (CV summary vs the job's
    # role_summary) or "chunks" (best CV chunks per job). See core/similarity.py, eval/README.md step 6.
    profile_similarity: str = "document"
    # Required skills: "llm" (retrieve the closest CV lines and skills-knowledge-base families, then
    # the LLM labels each met / partial / missing with a quote; core/requirements.py) or "match"
    # (matched or missing by name and embedding only, core/matcher.py).
    requirement_labels: str = "llm"
    llm_timeout_s: int = 60
    max_cv_chars: int = 15_000
    # What the LLM reads of each posting (skills, years). Covers ~95% of postings in full
    # (eval/data/jobs: median 4,300 chars, p95 8,900). The embedding limit is set per embedder.
    max_jd_chars: int = 10_000
    max_jd_file_chars: int = 50_000  # text kept from an uploaded posting, before the limits above
    local_max_cv_chars: int = 8_000  # the offline model runs on 2 CPUs: shorter input = faster
    local_max_jd_chars: int = 1_500
    local_n_ctx: int = 8_192         # llama-cpp's default context (512 tokens) is too small for a CV
    local_threads: int = 2           # Streamlit Community Cloud allows up to 2 CPUs
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
        jsearch_url=env("JSEARCH_URL", "https://jsearch.p.rapidapi.com/search-v2"),
        # Flash Lite first: every eval ran on it, and it answers in seconds. Flash thinks longer and,
        # on 10 postings of 10,000 characters, often runs past the 60 s deadline (504).
        gemini_models=_list(env("GEMINI_MODELS", env("GEMINI_MODEL",
                                "gemini-flash-lite-latest,gemini-flash-latest"))),
        gemini_embed_model=env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        local_llm_repo=env("LOCAL_LLM_REPO", "bartowski/Llama-3.2-1B-Instruct-GGUF"),
        local_llm_file=env("LOCAL_LLM_FILE", "*Q4_K_M.gguf"),
        local_embed_repo=env("LOCAL_EMBED_REPO", "CompendiumLabs/bge-small-en-v1.5-gguf"),
        local_embed_file=env("LOCAL_EMBED_FILE", "*q8_0.gguf"),
        embeddings=env("EMBEDDINGS", "auto"),
        experience_fit=env("EXPERIENCE_FIT", "judge"),
        profile_similarity=env("PROFILE_SIMILARITY", "document"),
        requirement_labels=env("REQUIREMENT_LABELS", "llm"),
    )
