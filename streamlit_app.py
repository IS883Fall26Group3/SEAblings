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
from core.llm import (AUTO, LLMError, auto_chain, build_embedder, build_provider,  # noqa: E402
                      model_options, with_fallback)
from core.experience import relevant_years_by_roles, role_years  # noqa: E402
from core.matcher import score_jobs  # noqa: E402
from core.skill_extractor import (explain_match, extract_profile, extract_requirements,  # noqa: E402
                                  judge_experience)
from core.validation import looks_like_job_description, looks_like_resume  # noqa: E402
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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_judgment(cv_hash: str, job_ids: tuple[str, ...], model_key: str, _cv_text, _jobs):
    return judge_experience(get_provider(model_key), _cv_text, _jobs, S.local_max_jd_chars)


def model_chain(choice: str) -> list[str]:
    if choice != AUTO:
        return [choice]
    # Skip models whose daily quota already ran out in this session (keep at least one).
    spent = st.session_state.get("spent_models", set())
    return [k for k in auto_chain(S) if k not in spent] or auto_chain(S)[-1:]


def remember_if_spent(key: str, e: Exception) -> None:
    if "daily quota" in str(e):
        st.session_state.setdefault("spent_models", set()).add(key)


OK = "Looks good."
NOT_CV = "This doesn't look like a resume. Please upload or paste your resume."
NOT_JD = "This doesn't look like a job posting. Please upload or paste a job description."


class RunStopped(Exception):
    """Ends a run early with a message: a "warning" for an input problem, otherwise an "error"."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def stop_if_invalid(checks: list[tuple[str, bool, str]]) -> None:
    """checks: (input name, passed, problem). Lists every input on its own line if any failed."""
    if all(ok for _, ok, _ in checks):
        return
    raise RunStopped("warning", "  \n".join(f"**{name}:** {OK if ok else problem}"
                                            for name, ok, problem in checks))


def scoring_failed(e: Exception) -> str:
    """The sidebar's models don't affect embeddings, so drop any advice to pick another one."""
    msg = str(e)
    if "daily quota" in msg:
        return "Scoring failed. " + msg.split(" Pick another model")[0]
    # Messages naming a secret are setup problems, which trying again won't fix.
    if isinstance(e, LLMError) and e.try_other_model and "GEMINI_" not in msg:
        return "Scoring failed, try again."
    return f"Scoring failed. {msg}"


def label(key: str) -> str:
    return OPTIONS.get(key, key)


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


# Everything one run produces. Cleared when a run starts, so old results can't come back after a failure.
RUN_KEYS = ("profile", "jobs", "results", "models_used", "embedder_used", "run_summary",
            "notice", "notes", "input_error", "explanations", "explain_errors")


def start_run() -> None:
    for k in RUN_KEYS:
        st.session_state.pop(k, None)
    st.session_state["running"] = True


def new_search() -> None:
    """Clear the page, keeping the CV. A new version in their keys empties the job inputs."""
    for k in RUN_KEYS:
        st.session_state.pop(k, None)
    st.session_state["jobs_version"] = st.session_state.get("jobs_version", 0) + 1


def show_notice() -> None:
    kind, message = st.session_state["notice"]
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

version = st.session_state.get("jobs_version", 0)

with st.sidebar:
    with st.container(border=True, key="cv_section"):
        st.subheader("1. Your CV", anchor=False)
        upload = st.file_uploader("CV (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        cv_paste = "" if upload is not None else st.text_area(
            "Or paste your CV", height=150, placeholder="Paste the text of your CV here.")
        if upload is None and not cv_paste.strip():
            st.caption("No CV yet? We'll use a fictional sample CV so you can try the app.")
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
        else:
            query = st.text_input("Role you're looking for", placeholder="e.g. Data analyst",
                                  key=f"query_{version}")
            location = st.text_input("Location", placeholder="e.g. Boston, MA or Remote",
                                     key=f"location_{version}")
            limit = st.slider("Number of jobs", 3, 10, S.default_num_jobs, key=f"limit_{version}")
            if S.job_source == "sample":
                st.info("Using sample job listings. Set JOB_SOURCE to \"jsearch\" in secrets for live jobs.")
    with st.container(border=True, key="model_section"):
        st.subheader("3. AI model", anchor=False)
        st.selectbox("Choose a model", list(OPTIONS), format_func=label, key="model_choice",
                     help="Each Gemini model has its own free daily quota. Auto tries them in order "
                          "and uses the offline model (if installed) as a last resort.")
    st.button("Analyze my match" if mode == MODE_JD else "Find matching jobs",
              type="primary", width="stretch", on_click=start_run)
    if input_error := st.session_state.get("input_error"):
        st.error(input_error)
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

# Separate slots: the results slot stays empty during a run, which blanks the last page at once.
progress_slot = st.empty()
results_slot = st.empty()

if st.session_state.get("running"):
    missing = ("Enter the role you're looking for." if mode == MODE_SEARCH and not query.strip()
               else "Upload or paste at least one job description."
               if mode == MODE_JD and not jd_files and not jd_text.strip() else "")
    if missing:
        st.session_state.update(input_error=missing, running=False)
        st.rerun()

    ui.grey_out_sidebar()
    main = progress_slot.container()
    choice = st.session_state["model_choice"]
    status = main.status("Getting started", expanded=True)
    live = main.empty()
    log = StageLog(status, 4 + (mode == MODE_SEARCH) + (S.experience_fit == "judge"))

    def notify(k, e):
        remember_if_spent(k, e)
        log.note(f"{label(k)} unavailable: {e} Trying the next model.")
    try:
        log.start("Reading your CV" if mode == MODE_SEARCH else "Reading your CV and job descriptions")
        if upload is not None:
            cv_text = extract_text(upload.getvalue(), upload.name, S.max_cv_chars)
        elif cv_paste.strip():
            cv_text = clean_text(cv_paste, S.max_cv_chars)
        else:
            cv_text = SAMPLE_CV.read_text(encoding="utf-8")
        cv_hash = hashlib.sha256(cv_text.encode()).hexdigest()
        # The bundled sample CV is trusted; only check what the user gave us.
        user_cv = upload is not None or bool(cv_paste.strip())
        checks = [("CV", not user_cv or looks_like_resume(cv_text), NOT_CV)]

        # Check the job descriptions too before any model call, so a wrong file costs no quota.
        if mode == MODE_JD:
            items = [(f.name, extract_text(f.getvalue(), f.name, S.max_jd_chars * 2, what="job description"))
                     for f in jd_files]
            if jd_text.strip():
                items.append(("Pasted job description", jd_text))
            checks += [(name if name == "Pasted job description" else f"Job description ({name})",
                        looks_like_job_description(text), NOT_JD) for name, text in items]
            raw_jobs = jobs_from_descriptions([(n.rsplit(".", 1)[0], t) for n, t in items])
        stop_if_invalid(checks)
        log.done("Read your CV" if mode == MODE_SEARCH else "Read your CV and job descriptions")

        log.start("Extracting your skills")
        profile, used_p = with_fallback(model_chain(choice), lambda k: cached_profile(cv_hash, cv_text, k), notify)
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
        publish(live, jobs=jobs)
        log.done(f"Read the requirements for {job_count(len(jobs))}")

        relevant, used_e = None, used_j
        if S.experience_fit == "judge":
            log.start("Judging your relevant experience")
            judged, used_e = with_fallback(
                model_chain(choice), lambda k: cached_judgment(cv_hash, ids, k, cv_text, jobs), notify)
            relevant = {jid: role_years(x.relevant_roles) for jid, x in judged.items()}
            log.done("Judged your relevant experience")

        log.start("Scoring each job")
        try:
            emb = get_embedder()
            embed = lambda t: cached_embed(tuple(t), emb.name)  # noqa: E731
            if S.experience_fit == "roles":
                relevant = relevant_years_by_roles(profile.roles, jobs, embed, emb.role_threshold)
            results = score_jobs(profile, cv_text, jobs, emb, S, embed_fn=embed, relevant_years=relevant)
        except Exception as e:
            raise RunStopped("error", scoring_failed(e)) from e
        used = sorted({label(used_p), label(used_j), label(used_e)})
        st.session_state.update(results=results, models_used=", ".join(used), embedder_used=emb.name,
                                run_summary="Analyzed 1 job" if len(results) == 1 else f"Ranked {len(results)} jobs")
    except RunStopped as e:
        st.session_state.update(notice=(e.kind, str(e)), notes=log.notes)
    except Exception as e:  # show a clear next step instead of a stack trace
        st.session_state.update(notice=("error", str(e)), notes=log.notes)
    st.session_state["running"] = False
    st.rerun()  # redraw the finished page; what the run finished is in session_state

notice = st.session_state.get("notice")
with results_slot.container():
    if results := st.session_state.get("results"):
        title, button = st.columns([5, 1], vertical_alignment="center")
        title.markdown("## Your matches")
        button.button("New search", on_click=new_search, width="stretch")
        st.caption(f"{st.session_state['run_summary']}. Analyzed with {st.session_state['models_used']}; "
                   f"similarity by {st.session_state['embedder_used']}.")
        ui.profile_panel(st.session_state["profile"])
        for i, r in enumerate(results):
            ui.match_card(r, i, handle_explain)
    elif notice and notice[0] == "error":
        show_notice()
        retry, new, _ = st.columns([1, 1, 3])
        retry.button("Try again", type="primary", on_click=start_run, width="stretch")
        new.button("New search", on_click=new_search, width="stretch")
        show_progress()
    else:
        if notice:
            show_notice()
        ui.hero()
