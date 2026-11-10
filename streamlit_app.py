"""SEAblings: CV-to-job matching. This file wires UI to core/ and holds no business logic."""
import hashlib
import time
import logging
import os
from dataclasses import replace

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
from core.advisor import match_tools  # noqa: E402
from core.embeddings import EmbeddingQuotaError, embed_with_wait, memo_embed  # noqa: E402
from core.job_sources import get_job_source, jobs_from_descriptions, name_from_requirements  # noqa: E402
from core.input_checks import cv_problem, posting_problem  # noqa: E402
from core.knowledge import SkillKB  # noqa: E402
from core.llm import (AUTO, LLMError, auto_chain, build_embedder, build_provider,  # noqa: E402
                      local_available, model_options, with_fallback)
from core.experience import relevant_roles_by_roles  # noqa: E402
from core.matcher import score_jobs  # noqa: E402
from core.requirements import retrieve_evidence  # noqa: E402
from core.samples import SAMPLE_CVS, sample_cv, sample_jobs  # noqa: E402
from core.skill_extractor import (explain_match, extract_profile, extract_requirements,  # noqa: E402
                                  judge_experience, label_requirements)
from core.validation import looks_like_role  # noqa: E402
from ui import components as ui  # noqa: E402

S = get_settings()
OPTIONS = model_options(S)
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
    return build_embedder(replace(S, embeddings=mode))


def get_embedder():
    # "local" once Gemini's daily embedding quota ran out in this session (see run below)
    return _embedder("local" if st.session_state.get("embed_local") else S.embeddings, S.gemini_api_key)


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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_judgment(cv_hash: str, job_ids: tuple[str, ...], model_key: str, _cv_text, _jobs):
    return judge_experience(get_provider(model_key), _cv_text, _jobs, S.max_jd_chars, S.local_max_jd_chars)


@st.cache_resource(show_spinner="Loading the skills knowledge base...")
def get_kb(embedder_name: str, _emb) -> SkillKB:
    # Saved vectors are used when they match this embedder; otherwise the entries are embedded once.
    return SkillKB.build(embedder_name, lambda t: embed_with_wait(_emb.embed, t))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_labels(cv_hash: str, job_ids: tuple[str, ...], model_key: str, embedder_name: str,
                  _jobs, _retrieved, _cv_text):
    return label_requirements(get_provider(model_key), _jobs, _retrieved, _cv_text)


def search_embed(emb, on_wait=None):
    """Embed function for one search: cached across reruns, each text embedded once, and waits
    out Gemini's per-minute limit."""
    return memo_embed(lambda t: embed_with_wait(lambda x: cached_embed(tuple(x), emb.name), t, on_wait))


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
        # Tool calling: the model calls compute_match_score and search_cv (core/advisor.py).
        tools = match_tools(result, st.session_state.get("cv_text", ""), search_embed(get_embedder()), S)
        (text, calls), _ = with_fallback(
            model_chain(st.session_state["model_choice"]),
            lambda k: explain_match(get_provider(k), st.session_state["profile"], result, tools),
            remember_if_spent)
        if not text.strip():
            raise LLMError("The model didn't finish its advice. Try again.")
        st.session_state.setdefault("explanations", {})[result.job.id] = text
        st.session_state.setdefault("tool_calls", {})[result.job.id] = calls
        errs.pop(result.job.id, None)
    except Exception as e:
        errs[result.job.id] = str(e)


# Everything one run produces, including the CV text kept for the advice tools.
RUN_KEYS = ("profile", "results", "cv_text", "models_used", "embedder_used",
            "explanations", "explain_errors", "tool_calls")


def new_search() -> None:
    """Clear the page, keeping the CV. A new version in their keys empties the job inputs."""
    for k in RUN_KEYS:
        st.session_state.pop(k, None)
    st.session_state["jobs_version"] = st.session_state.get("jobs_version", 0) + 1


# ---------- layout ----------
ui.inject_css()
ui.app_header()

version = st.session_state.get("jobs_version", 0)

with st.sidebar:
    with st.container(border=True, key="cv_section"):
        st.subheader("1. Your CV", anchor=False)
        upload = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        cv_paste = "" if upload is not None else st.text_area(
            "Or paste your CV", height=150, placeholder="Paste the text of your CV here.")
        if upload is None and not cv_paste.strip():
            st.selectbox("No CV yet? Try a fictional sample CV", list(SAMPLE_CVS), key="sample_cv")
    with st.container(border=True, key="jobs_section"):
        st.subheader("2. Jobs to match", anchor=False)
        mode = st.radio("Match against", [MODE_JD, MODE_SEARCH], horizontal=True,
                        help="Use job descriptions you already have, or search current openings online.")
        query = location = ""
        limit = S.default_num_jobs
        jd_files, jd_text = [], ""
        if mode == MODE_JD:
            jd_files = st.file_uploader("Job descriptions (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                                        accept_multiple_files=True, key=f"jd_files_{version}")
            jd_text = st.text_area("Or paste a job description", height=150, key=f"jd_text_{version}",
                                   placeholder="Paste the full posting. Put the job title on the first line.")
            if not jd_files and not jd_text.strip():
                st.caption(f"No job descriptions yet? We'll use {len(sample_jobs())} fictional sample postings.")
        else:
            query = st.text_input("Role you're looking for", placeholder="e.g. Data analyst",
                                  key=f"query_{version}")
            location = st.text_input("Location (optional)", placeholder="e.g. Boston, MA or Remote",
                                     key=f"location_{version}",
                                     help="If left empty, we search for this role across the US.")
            limit = st.slider("Number of jobs", 3, 10, S.default_num_jobs, key=f"limit_{version}")
            if S.job_source == "sample":
                st.info("Using sample job listings. Set JOB_SOURCE to \"jsearch\" in secrets for live jobs.")
    with st.container(border=True, key="model_section"):
        st.subheader("3. AI model", anchor=False)
        st.selectbox("Choose a model", list(OPTIONS), format_func=label, key="model_choice",
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

if run:
    if mode == MODE_SEARCH and not query.strip():
        st.sidebar.error("Enter the role you're looking for.")
        st.stop()
    if mode == MODE_SEARCH and not looks_like_role(query):
        st.sidebar.error("Enter a job role, such as Data analyst.")
        st.stop()
    for k in ("explanations", "explain_errors", "tool_calls"):
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
                cv_text = sample_cv(st.session_state.get("sample_cv", ""))
            cv_hash = hashlib.sha256(cv_text.encode()).hexdigest()

            st.write("Extracting your skills")
            profile, used_p = with_fallback(model_chain(choice), lambda k: cached_profile(cv_hash, cv_text, k), notify)
            if problem := cv_problem(profile):
                status.update(label="This doesn't look like a CV", state="error")
                st.error(f"{problem} Upload or paste your CV, or clear the CV box to try a sample CV.")
                st.stop()

            if mode == MODE_JD and not jd_files and not jd_text.strip():
                st.write("Using the fictional sample job postings")
                raw_jobs = sample_jobs()
            elif mode == MODE_JD:
                st.write("Reading your job descriptions")
                items = [(f.name.rsplit(".", 1)[0],
                          extract_text(f.getvalue(), f.name, S.max_jd_file_chars, what="job description"))
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
            jobs = name_from_requirements(jobs)
            skipped = [(j, why) for j in jobs if (why := posting_problem(j))]
            for j, why in skipped:  # never score a news item or an empty posting as a job
                st.write(f"Skipped \"{j.title}\": {why}.")
            jobs = [j for j in jobs if not posting_problem(j)]
            if not jobs:
                status.update(label="No job requirements found", state="error")
                st.warning("No job requirements were found in what you gave, so there is nothing to "
                           "match. Paste the full job posting, including its requirements.")
                st.stop()
            ids = tuple(j.id for j in jobs)

            relevant, background, used_e = None, None, used_j
            if S.experience_fit == "judge":
                st.write("Judging your relevant experience")
                try:
                    judged, used_e = with_fallback(
                        model_chain(choice), lambda k: cached_judgment(cv_hash, ids, k, cv_text, jobs), notify)
                    relevant = {jid: x.relevant_roles for jid, x in judged.items()}
                    background = {jid: (x.field_match, x.work_match) for jid, x in judged.items()}
                except LLMError as e:  # the rest of the match is ready: score with total years instead
                    st.write(f"Couldn't judge your relevant experience ({e}) Experience fit uses "
                             "all your years instead.")

            def embedding_steps(emb):
                """Everything that needs embeddings, so it can run again on the local model."""
                embed = search_embed(emb, lambda s: st.write(
                    f"Gemini's per-minute limit is reached. Waiting {s} seconds, then trying again…"))
                roles = (relevant_roles_by_roles(profile.roles, jobs, embed, emb.role_threshold)
                         if S.experience_fit == "roles" else relevant)
                checks, used_r = None, used_j
                if S.requirement_labels == "llm":
                    # Retrieval: the CV lines and knowledge-base families closest to each requirement
                    # go to the LLM, which labels it met / partial / missing with a quote.
                    st.write("Checking each requirement against your CV")
                    try:
                        retrieved = retrieve_evidence(jobs, cv_text, embed, get_kb(emb.name, emb))
                        checks, used_r = with_fallback(
                            model_chain(choice),
                            lambda k: cached_labels(cv_hash, ids, k, emb.name, jobs, retrieved, cv_text), notify)
                    except EmbeddingQuotaError:
                        raise
                    except LLMError as e:  # the rest of the match is ready: match skills by name instead
                        st.write(f"Couldn't check each requirement ({e}) Skills are matched by name and "
                                 "meaning instead.")
                st.write("Scoring each job")
                return score_jobs(profile, cv_text, jobs, emb, S, embed_fn=embed, relevant_roles=roles,
                                  background=background, requirement_checks=checks), used_r

            emb = get_embedder()
            try:
                results, used_r = embedding_steps(emb)
            except EmbeddingQuotaError as e:
                if not local_available():
                    raise LLMError(f"{e} Try again after it resets, or set EMBEDDINGS = \"local\" in "
                                   "Secrets with llama-cpp-python installed.") from e
                st.write(f"{e} Switching to the offline embedding model for this session (it loads in "
                         "about a minute the first time).")
                st.session_state["embed_local"] = True
                emb = get_embedder()
                results, used_r = embedding_steps(emb)
            used = sorted({label(used_p), label(used_j), label(used_e), label(used_r)})
            # The CV text stays in this session's memory only, for the advice tools (search_cv).
            st.session_state.update(profile=profile, results=results, cv_text=cv_text,
                                    models_used=", ".join(used), embedder_used=emb.name)
            done = "Analyzed 1 job" if len(results) == 1 else f"Ranked {len(results)} jobs"
            if skipped:
                done += f" (skipped {len(skipped)} that isn't a job posting)"
            status.update(label=done, state="complete", expanded=False)
        except Exception as e:  # show a clear next step instead of a stack trace
            status.update(label="Something went wrong", state="error")
            st.error(str(e))
            st.stop()

results = st.session_state.get("results")
if not results:
    ui.hero()
else:
    title, button = st.columns([5, 1], vertical_alignment="center")
    title.markdown("## Your matches")
    button.button("New search", on_click=new_search, width="stretch")
    st.caption(f"Analyzed with {st.session_state['models_used']}; "
               f"similarity by {st.session_state['embedder_used']}.")
    ui.profile_panel(st.session_state["profile"])
    for i, r in enumerate(results):
        # results are sorted best first: the top 3 show their breakdown open
        ui.match_card(r, i, handle_explain, st.session_state["profile"], show_breakdown=i < 3)
