"""SEAblings: CV-to-job matching. This file wires UI to core/ and holds no business logic."""
import hashlib
import time
import logging
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SEAblings | Job match", page_icon="🌊", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def load_secrets_into_env() -> None:
    try:
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v.strip()  # overwrite so edited secrets take effect
    except Exception:
        pass  # no secrets file locally: core falls back to the sample job source


load_secrets_into_env()

from core.config import get_settings  # noqa: E402
from core.cv_parser import clean_text, extract_text  # noqa: E402
from core.job_sources import get_job_source, jobs_from_descriptions  # noqa: E402
from core.llm import (AUTO, auto_chain, build_embedder, build_provider,  # noqa: E402
                      model_options, with_fallback)
from core.matcher import score_jobs  # noqa: E402
from core.skill_extractor import explain_match, extract_profile, extract_requirements  # noqa: E402
from ui import components as ui  # noqa: E402

S = get_settings()
OPTIONS = model_options(S)
SAMPLE_CV = Path(__file__).parent / "data" / "sample" / "sample_cv.txt"
MODE_JD, MODE_SEARCH = "My job descriptions", "Search live jobs"


# ---------- cached models and calls (protect free-tier quotas) ----------
# Cache keys include the settings each object depends on, so editing Secrets takes effect.
@st.cache_resource(show_spinner="Loading the model (the offline model takes a few minutes the first time)...")
def _provider(key: str, api_key: str):
    return build_provider(key, S)


def get_provider(key: str):
    return _provider(key, S.gemini_api_key)


@st.cache_resource(show_spinner="Loading the embedding model...")
def _embedder(mode: str, api_key: str):
    return build_embedder(S)


def get_embedder():
    return _embedder(S.embeddings, S.gemini_api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_profile(cv_hash: str, cv_text: str, model_key: str):
    return extract_profile(get_provider(model_key), cv_text)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_search(source: str, query: str, location: str, limit: int):
    return get_job_source(S).search(query, location, limit)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_requirements(job_ids: tuple[str, ...], model_key: str, _jobs):
    return extract_requirements(get_provider(model_key), _jobs, S.max_jd_chars, S.local_max_jd_chars)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_embed(texts: tuple[str, ...], embedder_name: str):
    return get_embedder().embed(list(texts))


def model_chain(choice: str) -> list[str]:
    if choice != AUTO:
        return [choice]
    # Skip models whose daily quota already ran out in this session (keep at least one).
    spent = st.session_state.get("spent_models", set())
    return [k for k in auto_chain(S) if k not in spent] or auto_chain(S)[-1:]


def remember_if_spent(key: str, e: Exception) -> None:
    if "daily quota" in str(e):
        st.session_state.setdefault("spent_models", set()).add(key)


def label(key: str) -> str:
    return OPTIONS.get(key, key)


def handle_explain(result) -> None:
    errs = st.session_state.setdefault("explain_errors", {})
    try:
        text, _ = with_fallback(
            model_chain(st.session_state["model_choice"]),
            lambda k: explain_match(get_provider(k), st.session_state["profile"], result),
            remember_if_spent)
        st.session_state.setdefault("explanations", {})[result.job.id] = text
        errs.pop(result.job.id, None)
    except Exception as e:
        errs[result.job.id] = str(e)


# ---------- layout ----------
ui.inject_css()
ui.app_header()

with st.sidebar:
    st.subheader("Your search")
    upload = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    cv_paste = "" if upload is not None else st.text_area(
        "Or paste your CV", height=150, placeholder="Paste the text of your CV here.")
    if upload is None and not cv_paste.strip():
        st.caption("No CV yet? We'll use a fictional sample CV so you can try the app.")
    mode = st.radio("Match against", [MODE_JD, MODE_SEARCH], horizontal=True,
                    help="Use job descriptions you already have, or search current openings online.")
    query = location = ""
    limit = S.default_num_jobs
    jd_files, jd_text = [], ""
    if mode == MODE_JD:
        jd_files = st.file_uploader("Job descriptions (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                                    accept_multiple_files=True)
        jd_text = st.text_area("Or paste a job description", height=150,
                               placeholder="Paste the full posting. Put the job title on the first line.")
    else:
        query = st.text_input("Role you're looking for", placeholder="e.g. Data analyst")
        location = st.text_input("Location", placeholder="e.g. Boston, MA or Remote")
        limit = st.slider("Number of jobs", 3, 10, S.default_num_jobs)
    st.selectbox("AI model", list(OPTIONS), format_func=label, key="model_choice",
                 help="Each Gemini model has its own free daily quota. Auto tries them in order "
                      "and uses the offline model (if installed) as a last resort.")
    run = st.button("Analyze my match" if mode == MODE_JD else "Find matching jobs",
                    type="primary", width="stretch")
    st.caption("Your CV is processed in memory and not saved by this app. With a Gemini model, "
               "its text is sent to Google's Gemini API for analysis.")

    with st.expander("Connection check"):
        key = S.gemini_api_key
        st.caption(f"Gemini key: {'found, ends in ' + key[-4:] if key else 'missing'}  \n"
                   f"Embeddings: {S.embeddings} (Gemini: {S.gemini_embed_model})  \n"
                   f"Job source: {S.job_source}")
        if st.button("Test selected model", width="stretch"):
            for k in model_chain(st.session_state["model_choice"]):
                try:
                    t = time.time()
                    get_provider(k).generate_text("Reply with OK.", what="testing the connection")
                    st.success(f"{label(k)}: OK in {time.time() - t:.1f}s")
                except Exception as e:
                    st.error(f"{label(k)}: {e}")
    if mode == MODE_SEARCH and S.job_source == "sample":
        st.info("Using sample job listings. Set JOB_SOURCE to \"jsearch\" in secrets for live jobs.")

if run:
    if mode == MODE_SEARCH and not query.strip():
        st.sidebar.error("Enter the role you're looking for.")
        st.stop()
    if mode == MODE_JD and not jd_files and not jd_text.strip():
        st.sidebar.error("Upload or paste at least one job description.")
        st.stop()
    for k in ("explanations", "explain_errors"):
        st.session_state.pop(k, None)
    choice = st.session_state["model_choice"]
    with st.status("Matching your CV to jobs", expanded=True) as status:
        def notify(k, e):
            remember_if_spent(k, e)
            st.write(f"{label(k)} unavailable: {e} Trying the next model.")
        try:
            st.write("Reading your CV")
            if upload is not None:
                cv_text = extract_text(upload.getvalue(), upload.name, S.max_cv_chars)
            elif cv_paste.strip():
                cv_text = clean_text(cv_paste, S.max_cv_chars)
            else:
                cv_text = SAMPLE_CV.read_text(encoding="utf-8")
            cv_hash = hashlib.sha256(cv_text.encode()).hexdigest()

            st.write("Extracting your skills")
            profile, used_p = with_fallback(model_chain(choice), lambda k: cached_profile(cv_hash, cv_text, k), notify)

            if mode == MODE_JD:
                st.write("Reading your job descriptions")
                items = [(f.name.rsplit(".", 1)[0],
                          extract_text(f.getvalue(), f.name, S.max_jd_chars * 2, what="job description"))
                         for f in jd_files]
                items.append(("Pasted job description", jd_text))
                raw_jobs = jobs_from_descriptions(items)
            else:
                st.write(f"Searching jobs for \"{query}\"")
                raw_jobs = cached_search(S.job_source, query.strip(), location.strip(), limit)
            if not raw_jobs:
                status.update(label="No jobs found", state="error")
                st.warning("No openings matched this search. Try a broader role or location.")
                st.stop()

            st.write("Reading the job descriptions")
            ids = tuple(j.id for j in raw_jobs)
            jobs, used_j = with_fallback(model_chain(choice), lambda k: cached_requirements(ids, k, raw_jobs), notify)

            st.write("Scoring each job")
            emb = get_embedder()
            results = score_jobs(profile, cv_text, jobs, emb, S,
                                 embed_fn=lambda t: cached_embed(tuple(t), emb.name))
            used = sorted({label(used_p), label(used_j)})
            st.session_state.update(profile=profile, results=results,
                                    models_used=", ".join(used), embedder_used=emb.name)
            done = "Analyzed 1 job" if len(results) == 1 else f"Ranked {len(results)} jobs"
            status.update(label=done, state="complete", expanded=False)
        except Exception as e:  # show a clear next step instead of a stack trace
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            st.stop()

results = st.session_state.get("results")
if not results:
    ui.hero()
else:
    st.markdown("## Your matches")
    st.caption(f"Analyzed with {st.session_state['models_used']}; "
               f"similarity by {st.session_state['embedder_used']}.")
    ui.profile_panel(st.session_state["profile"])
    for i, r in enumerate(results):
        ui.match_card(r, i, handle_explain)
