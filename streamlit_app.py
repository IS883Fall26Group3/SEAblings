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
from core.validation import looks_like_job_description, looks_like_resume, looks_like_role  # noqa: E402
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


def request_advice(job_id: str) -> None:
    """Button callback: only marks the job. The advice is written after every card is drawn, so the
    whole page can be locked while it runs (see the results section)."""
    st.session_state["advice_pending"] = job_id


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


OK = "Looks good."
NOT_CV = "This doesn't look like a resume. Please upload or paste your resume."
NOT_JD = "This doesn't look like a job posting. Please upload or paste a job description."
NOT_ROLE = "Enter a job role, such as Data analyst."


class RunStopped(Exception):
    """Ends a run early with a message. kind is "error" or "warning"; retry offers Try again, which
    only helps when something failed, not when an input is wrong."""

    def __init__(self, kind: str, message: str, retry: bool = False):
        super().__init__(message)
        self.kind, self.retry = kind, retry


def stop_if_invalid(checks: list[tuple[str, bool, str]]) -> None:
    """checks: (input name, passed, problem). Lists every input on its own line if any failed."""
    if all(ok for _, ok, _ in checks):
        return
    raise RunStopped("warning", "  \n".join(f"**{name}:** {OK if ok else problem}"
                                            for name, ok, problem in checks))


def read_input(read) -> tuple[str, str]:
    """(text, problem): a file or paste we can't read (scanned PDF, too little text, wrong type)
    becomes a problem shown with the other input checks, not an error with Try again."""
    try:
        return read(), ""
    except ValueError as e:
        return "", str(e)


def job_count(n: int) -> str:
    return "1 job" if n == 1 else f"{n} jobs"


class StageLog:
    """Status box narration: a tick for each finished stage, then the one in progress."""

    def __init__(self, status, total: int):
        self.status, self.total, self.number = status, total, 0
        self.lines, self.notes, self.current = [], [], ""
        self.box = status.empty()

    def start(self, text: str) -> None:
        self.number += 1
        self.current = text
        self.status.update(label=f"Step {self.number} of {self.total}: {text}", expanded=True)
        self.draw()

    def done(self, text: str) -> None:
        self.lines.append(f"✓ {text}")
        self.current = ""
        self.draw()

    def note(self, text: str) -> None:
        self.notes.append(text)
        self.lines.append(f"⚠️ {text}")
        self.draw()

    def draw(self) -> None:
        current = [f"⏳ {self.current}"] if self.current else []
        self.box.markdown("  \n".join(self.lines + current))


# Everything one run produces, including the CV text kept for the advice tools. Cleared when a run
# starts, so old results can't come back after a failure.
RUN_KEYS = ("profile", "jobs", "results", "cv_text", "models_used", "embedder_used", "run_summary",
            "notice", "notes", "input_error", "explanations", "explain_errors", "tool_calls",
            "advice_pending")


def start_run() -> None:
    for k in RUN_KEYS:
        st.session_state.pop(k, None)
    st.session_state["running"] = True


def show_notice() -> None:
    kind, message, _ = st.session_state["notice"]
    (st.error if kind == "error" else st.warning)(message)
    for note in st.session_state.get("notes", []):
        st.caption(note)


def show_progress() -> None:
    """What's finished so far, during a run and after one that stopped early."""
    if profile := st.session_state.get("profile"):
        ui.profile_panel(profile, expanded=True)
    if jobs := st.session_state.get("jobs"):
        ui.job_previews(jobs)


def publish(live, **finished) -> None:
    """Save a finished stage right away (a rerun erases anything only drawn) and show it."""
    st.session_state.update(finished)
    with live.container():
        show_progress()


def new_search() -> None:
    """Clear the page, keeping the CV. A new version in their keys empties the job inputs."""
    for k in RUN_KEYS:
        st.session_state.pop(k, None)
    st.session_state["jobs_version"] = st.session_state.get("jobs_version", 0) + 1


# ---------- layout ----------
ui.inject_css()

version = st.session_state.get("jobs_version", 0)
notice = st.session_state.get("notice")

# Sidebar: the AI model and what the app can't do. The inputs live on the main page.
with st.sidebar:
    st.markdown('<p class="sea-side-brand">🌊 SEAblings · AI CV-to-job matcher</p>'
                '<p class="sea-side-team">IS883 project (Team 3)</p>', unsafe_allow_html=True)
    with st.container(border=True, key="model_section"):
        st.subheader("AI model", anchor=False)
        st.selectbox("Choose a model", list(OPTIONS), format_func=label, key="model_choice",
                     help="Each Gemini model has its own free daily quota. Auto tries them in order "
                          "and uses the offline model (if installed) as a last resort.")
        with st.expander("Connection check"):
            key = S.gemini_api_key
            st.caption(f"Gemini key: {'found, ends in ' + key[-4:] if key else 'missing'}  \n"
                       f"Embeddings: {S.embeddings} (Gemini: {S.gemini_embed_model})  \n"
                       f"Job source: {S.job_source}")
            if st.button("Test selected model", width="stretch"):
                for k in model_chain(st.session_state["model_choice"]):
                    try:
                        t = time.time()
                        with st.spinner(f"Testing {label(k)}..."):
                            get_provider(k).generate_text("Reply with OK.", what="testing the connection")
                        st.success(f"{label(k)}: OK in {time.time() - t:.1f}s")
                    except Exception as e:
                        st.error(f"{label(k)}: {e}")
    ui.limitations(S)

# Main page: a short intro on the landing page, then the inputs.
if not st.session_state.get("results") and not st.session_state.get("running") \
        and not (notice and notice[2]):
    ui.hero()  # the banner carries the app name, so no header bar above it
else:
    ui.app_header()

# Why the last run stopped, right under the header so it's seen first.
if notice:
    show_notice()
    if notice[2]:  # something failed, so trying again can help
        retry, new, _ = st.columns([1, 1, 3])
        retry.button("Try again", type="primary", on_click=start_run, width="stretch")
        new.button("New search", on_click=new_search, width="stretch")
if input_error := st.session_state.get("input_error"):
    st.error(input_error)

with st.container(key="inputs"):
    cv_col, jobs_col = st.columns(2, gap="medium")
    with cv_col, st.container(border=True, key="cv_section"):
        st.subheader("1. Your CV", anchor=False)
        upload = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        cv_paste = "" if upload is not None else st.text_area(
            "Or paste your CV", height=150, placeholder="Paste the text of your CV here.")
        if upload is None and not cv_paste.strip():
            st.selectbox("No CV yet? Try a sample CV", list(SAMPLE_CVS), key="sample_cv")
    with jobs_col, st.container(border=True, key="jobs_section"):
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
                st.caption(f"No job descriptions yet? We'll use {len(sample_jobs())} sample postings.")
        else:
            query = st.text_input("Role you're looking for", placeholder="e.g. Data analyst",
                                  key=f"query_{version}")
            location = st.text_input("Location (optional)", placeholder="e.g. Boston, MA or Remote",
                                     key=f"location_{version}",
                                     help="If left empty, we search for this role across the US.")
            limit = st.slider("Number of jobs", 3, 10, S.default_num_jobs, key=f"limit_{version}")
            if S.job_source == "sample":
                st.info("Using sample job listings. Set JOB_SOURCE to \"jsearch\" in secrets for live jobs.")
    st.button("Analyze my match" if mode == MODE_JD else "Find matching jobs",
              type="primary", width="stretch", on_click=start_run)
    st.caption("Your CV is processed in memory and not saved by this app. With a Gemini model, "
               "its text is sent to Google's Gemini API for analysis.")

# Separate slots: the results slot stays empty during a run, which blanks the last page at once.
progress_slot = st.empty()
results_slot = st.empty()

if st.session_state.get("running"):
    if mode == MODE_SEARCH and not query.strip():
        st.session_state.update(input_error="Enter the role you're looking for.", running=False)
        st.rerun()

    ui.grey_out_inputs()
    main = progress_slot.container()
    choice = st.session_state["model_choice"]
    status = main.status("Getting started", expanded=True)
    live = main.empty()
    log = StageLog(status, 4 + (mode == MODE_SEARCH) + (S.experience_fit == "judge")
                   + (S.requirement_labels == "llm"))

    def notify(k, e):
        remember_if_spent(k, e)
        log.note(f"{label(k)} unavailable: {e} Trying the next model.")
    try:
        # Defense 1: simple rules on what the user gave, before any model call, so a wrong input
        # costs no quota. The sample CVs and postings are ours, so they are trusted.
        sample_postings = mode == MODE_JD and not jd_files and not jd_text.strip()
        reading = "Reading your CV" if mode == MODE_SEARCH else "Reading your CV and job descriptions"
        log.start(reading)
        cv_error = ""  # a file or paste we couldn't read (e.g. a scanned PDF) is shown with the checks
        if upload is not None:
            cv_text, cv_error = read_input(lambda: extract_text(upload.getvalue(), upload.name, S.max_cv_chars))
        elif cv_paste.strip():
            cv_text, cv_error = read_input(lambda: clean_text(cv_paste, S.max_cv_chars))
        else:
            cv_text = sample_cv(st.session_state.get("sample_cv", ""))
        cv_hash = hashlib.sha256(cv_text.encode()).hexdigest()
        st.session_state["cv_text"] = cv_text  # in memory only; the advice tools (search_cv) need it
        user_cv = upload is not None or bool(cv_paste.strip())
        checks = [("CV", not cv_error and (not user_cv or looks_like_resume(cv_text)), cv_error or NOT_CV)]
        if sample_postings:
            raw_jobs = sample_jobs()
        elif mode == MODE_JD:
            items = []
            for f in jd_files:
                text, error = read_input(
                    lambda: extract_text(f.getvalue(), f.name, S.max_jd_file_chars, what="job description"))
                if error:
                    checks.append((f"Job description ({f.name})", False, error))
                else:
                    items.append((f.name, text))
            if jd_text.strip():
                items.append(("Pasted job description", jd_text))
            checks += [(name if name == "Pasted job description" else f"Job description ({name})",
                        looks_like_job_description(text), NOT_JD) for name, text in items]
            raw_jobs = jobs_from_descriptions([(n.rsplit(".", 1)[0], t) for n, t in items])
        else:
            checks.append(("Role", looks_like_role(query), NOT_ROLE))
        stop_if_invalid(checks)
        log.done("Read your CV and the sample job postings" if sample_postings
                 else reading.replace("Reading", "Read"))

        log.start("Extracting your skills")
        profile, used_p = with_fallback(model_chain(choice), lambda k: cached_profile(cv_hash, cv_text, k), notify)
        # Defense 2: the model says what kind of document it read, in the same extraction call.
        if problem := cv_problem(profile):
            raise RunStopped("error", f"{problem} Upload or paste your CV, or clear the CV box to try a sample CV.")
        publish(live, profile=profile)
        log.done(f"Found {len(profile.skills)} skills and {profile.years_experience:g} years of experience")

        if mode == MODE_SEARCH:
            log.start(f"Searching jobs for \"{query}\"")
            raw_jobs = cached_search(S.job_source, query.strip(), location.strip(), limit)
            publish(live, jobs=raw_jobs)
            log.done(f"Found {job_count(len(raw_jobs))}")
        if not raw_jobs:
            raise RunStopped("warning", "No openings matched this search. Try a broader role or location.")

        log.start("Reading the job requirements")
        ids = tuple(j.id for j in raw_jobs)
        jobs, used_j = with_fallback(model_chain(choice), lambda k: cached_requirements(ids, k, raw_jobs), notify)
        jobs = name_from_requirements(jobs)
        skipped = [(j, why) for j in jobs if (why := posting_problem(j))]
        for j, why in skipped:  # never score a news item or an empty posting as a job
            log.note(f"Skipped \"{j.title}\": {why}.")
        jobs = [j for j in jobs if not posting_problem(j)]
        if not jobs:
            raise RunStopped("warning", "No job requirements were found in what you gave, so there is nothing "
                                        "to match. Paste the full job posting, including its requirements.")
        publish(live, jobs=jobs)
        log.done(f"Read the requirements for {job_count(len(jobs))}")
        ids = tuple(j.id for j in jobs)

        relevant, background, used_e = None, None, used_j
        if S.experience_fit == "judge":
            log.start("Judging your relevant experience")
            try:
                judged, used_e = with_fallback(
                    model_chain(choice), lambda k: cached_judgment(cv_hash, ids, k, cv_text, jobs), notify)
                relevant = {jid: x.relevant_roles for jid, x in judged.items()}
                background = {jid: (x.field_match, x.work_match) for jid, x in judged.items()}
                log.done("Judged your relevant experience")
            except LLMError as e:  # the rest of the match is ready: score with total years instead
                log.note(f"Couldn't judge your relevant experience ({e}) Experience fit uses "
                         "all your years instead.")

        def embedding_steps(emb):
            """Everything that needs embeddings, so it can run again on the local model."""
            embed = search_embed(emb, lambda s: log.note(
                f"Gemini's per-minute limit is reached. Waiting {s} seconds, then trying again…"))
            roles = (relevant_roles_by_roles(profile.roles, jobs, embed, emb.role_threshold)
                     if S.experience_fit == "roles" else relevant)
            checks, used_r = None, used_j
            if S.requirement_labels == "llm":
                # Retrieval: the CV lines and knowledge-base families closest to each requirement
                # go to the LLM, which labels it met / partial / missing with a quote.
                log.start("Checking each requirement against your CV")
                try:
                    retrieved = retrieve_evidence(jobs, cv_text, embed, get_kb(emb.name, emb))
                    checks, used_r = with_fallback(
                        model_chain(choice),
                        lambda k: cached_labels(cv_hash, ids, k, emb.name, jobs, retrieved, cv_text), notify)
                    log.done("Checked each requirement against your CV")
                except EmbeddingQuotaError:
                    raise
                except LLMError as e:  # the rest of the match is ready: match skills by name instead
                    log.note(f"Couldn't check each requirement ({e}) Skills are matched by name and "
                             "meaning instead.")
            log.start("Scoring each job")
            return score_jobs(profile, cv_text, jobs, emb, S, embed_fn=embed, relevant_roles=roles,
                              background=background, requirement_checks=checks), used_r

        emb = get_embedder()
        steps_so_far = log.number
        try:
            results, used_r = embedding_steps(emb)
        except EmbeddingQuotaError as e:
            if not local_available():
                raise LLMError(f"{e} Try again after it resets, or set EMBEDDINGS = \"local\" in "
                               "Secrets with llama-cpp-python installed.") from e
            log.note(f"{e} Switching to the offline embedding model for this session (it loads in "
                     "about a minute the first time).")
            st.session_state["embed_local"] = True
            emb = get_embedder()
            log.number = steps_so_far  # the same steps again, on the local model
            results, used_r = embedding_steps(emb)
        used = sorted({label(used_p), label(used_j), label(used_e), label(used_r)})
        summary = "Analyzed 1 job" if len(results) == 1 else f"Ranked {len(results)} jobs"
        if skipped:
            summary += f" (skipped {len(skipped)} that isn't a job posting)"
        st.session_state.update(results=results, models_used=", ".join(used), embedder_used=emb.name,
                                run_summary=summary, notes=log.notes)
    except RunStopped as e:
        st.session_state.update(notice=(e.kind, str(e), e.retry), notes=log.notes)
    except Exception as e:  # show a clear next step instead of a stack trace
        st.session_state.update(notice=("error", str(e), True), notes=log.notes)
    st.session_state["running"] = False
    st.rerun()  # redraw the finished page; what the run finished is in session_state

with results_slot.container(), st.container(key="results"):
    if results := st.session_state.get("results"):
        title, button = st.columns([5, 1], vertical_alignment="center")
        title.markdown("## Your matches")
        button.button("New search", on_click=new_search, width="stretch")
        st.caption(f"{st.session_state['run_summary']}. Analyzed with {st.session_state['models_used']}; "
                   f"similarity by {st.session_state['embedder_used']}.")
        for note in st.session_state.get("notes", []):
            st.caption(note)
        ui.profile_panel(st.session_state["profile"])
        pending = st.session_state.get("advice_pending")
        if pending:
            ui.grey_out_results()
        slot = pending_result = None
        for i, r in enumerate(results):
            # results are sorted best first: the top 3 show their breakdown open
            card_slot = ui.match_card(r, i, request_advice, st.session_state["profile"],
                                      show_breakdown=i < 3, pending=r.job.id == pending)
            if card_slot is not None:
                slot, pending_result = card_slot, r
        if pending:
            st.session_state.pop("advice_pending")
            if pending_result is not None:
                with slot.container(), st.spinner("Writing advice for this role..."):
                    handle_explain(pending_result)
            st.rerun()
    elif notice and notice[2]:
        show_progress()  # what finished before the failure (the notice is at the top)
