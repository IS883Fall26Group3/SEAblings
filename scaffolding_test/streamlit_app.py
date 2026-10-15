"""SEAblings: CV-to-job matching. This file wires UI to core/ and holds no business logic."""
import hashlib
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SEAblings | Job match", page_icon="🌊", layout="wide")


def load_secrets_into_env() -> None:
    try:
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ.setdefault(k, v)
    except Exception:
        pass  # no secrets file locally: core falls back to the sample job source


load_secrets_into_env()

from core.config import get_settings  # noqa: E402
from core.cv_parser import extract_text  # noqa: E402
from core.job_sources import get_job_source  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from core.matcher import score_jobs  # noqa: E402
from core.skill_extractor import explain_match, extract_profile, extract_requirements  # noqa: E402
from ui import components as ui  # noqa: E402

S = get_settings()
SAMPLE_CV = Path(__file__).parent / "data" / "sample" / "sample_cv.txt"


# ---------- cached resources and calls (protect free-tier quotas) ----------
@st.cache_resource
def get_llm() -> LLMClient:
    return LLMClient(S.gemini_api_key, S.gemini_model, S.gemini_embed_model)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_profile(cv_hash: str, cv_text: str):
    return extract_profile(get_llm(), cv_text)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_jobs(source: str, query: str, location: str, limit: int):
    jobs = get_job_source(S).search(query, location, limit)
    return extract_requirements(get_llm(), jobs, S.max_jd_chars)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_embed(texts: tuple[str, ...]):
    return get_llm().embed(list(texts))


def handle_explain(result) -> None:
    text = explain_match(get_llm(), st.session_state["profile"], result)
    st.session_state.setdefault("explanations", {})[result.job.id] = text


# ---------- layout ----------
ui.inject_css()

with st.sidebar:
    st.subheader("Your search")
    upload = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    if upload is None:
        st.caption("No CV yet? We'll use a fictional sample CV so you can try the app.")
    query = st.text_input("Role you're looking for", placeholder="e.g. Data analyst")
    location = st.text_input("Location", placeholder="e.g. Boston, MA or Remote")
    limit = st.slider("Number of jobs", 3, 10, S.default_num_jobs)
    run = st.button("Find matching jobs", type="primary", width="stretch")
    st.caption("Your CV is processed in memory and not saved by this app. "
               "Its text is sent to Google's Gemini API for analysis.")
    if S.job_source == "sample":
        st.info("Using sample job listings. Set JOB_SOURCE to \"jsearch\" in secrets for live jobs.")

if run:
    if not query.strip():
        st.sidebar.error("Enter the role you're looking for.")
        st.stop()
    st.session_state.pop("explanations", None)
    with st.status("Matching your CV to jobs", expanded=True) as status:
        try:
            st.write("Reading your CV")
            if upload is not None:
                cv_text = extract_text(upload.getvalue(), upload.name, S.max_cv_chars)
            else:
                cv_text = SAMPLE_CV.read_text(encoding="utf-8")
            cv_hash = hashlib.sha256(cv_text.encode()).hexdigest()

            st.write("Extracting your skills")
            profile = cached_profile(cv_hash, cv_text)

            st.write(f"Searching jobs for \"{query}\"")
            jobs = cached_jobs(S.job_source, query.strip(), location.strip(), limit)
            if not jobs:
                status.update(label="No jobs found", state="error")
                st.warning("No openings matched this search. Try a broader role or location.")
                st.stop()

            st.write("Scoring each job")
            results = score_jobs(profile, cv_text, jobs, lambda t: cached_embed(tuple(t)), S)
            st.session_state.update(profile=profile, results=results)
            status.update(label=f"Ranked {len(results)} jobs", state="complete", expanded=False)
        except Exception as e:  # show a clear next step instead of a stack trace
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            st.stop()

results = st.session_state.get("results")
if not results:
    ui.hero()
else:
    st.markdown("## Your matches")
    ui.profile_panel(st.session_state["profile"])
    for i, r in enumerate(results):
        ui.match_card(r, i, handle_explain)
