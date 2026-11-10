"""Reusable UI pieces. Keep presentation here so streamlit_app.py stays readable."""
from html import escape
from pathlib import Path

import streamlit as st

from core import score_notes as notes
from core.schemas import CandidateProfile, Job, MatchResult

CSS = Path(__file__).with_name("styles.css")


def inject_css() -> None:
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def grey_out_inputs() -> None:
    """Clicking an input would stop the run partway. Uses CSS because disabled=True drops text
    typed just before the click (Streamlit 1.66)."""
    st.markdown('<style>.st-key-inputs, [data-testid="stSidebar"] '
                '{ opacity: .5; pointer-events: none; }</style>', unsafe_allow_html=True)


def grey_out_results() -> None:
    """While advice is written, a click anywhere would stop it partway, so lock the whole page."""
    st.markdown('<style>.st-key-inputs, [data-testid="stSidebar"] { opacity: .5; pointer-events: none; }'
                '.st-key-results { pointer-events: none; }'
                '.st-key-results [data-testid="stButton"], .st-key-results [data-testid="stLinkButton"], '
                '.st-key-results [data-testid="stExpander"] { opacity: .5; }</style>',
                unsafe_allow_html=True)


def limitations(s) -> None:
    """What the app can't do, so users read the score with the right expectations."""
    search = (f"Live search covers US job listings only, {s.jsearch_num_jobs} jobs per search."
              if s.job_source == "jsearch" else "Job search uses sample listings in this setup, not live openings.")
    items = [
        "The score is based on your CV text only.",
        "Only English CVs, job descriptions and roles are supported.",
        "Scanned PDFs (images of text) can't be read.",
        f"The model reads up to {s.max_cv_chars:,} characters of your CV and {s.max_jd_chars:,} of each posting.",
        search,
        "If a Gemini model reaches its daily limit, we automatically switch to another model.",
        "AI advice can be wrong. Check it before you change your CV.",
    ]
    with st.container(key="limits_section"):
        st.subheader("Limitations", anchor=False)
        st.markdown("\n".join(f"- {i}" for i in items))


def app_header() -> None:
    """Brand bar shown on every screen, so the app's purpose is clear at a glance."""
    st.markdown(
        """<div class="sea-header">
        <div class="sea-brand"><span class="sea-logo" aria-hidden="true">🌊</span>
          <span class="sea-name">AI CV-to-job matcher</span>
          <span class="sea-tag">SEAblings</span></div>
        <div class="sea-team">IS883 · Fall 2026 · Group 3</div>
        </div>""",
        unsafe_allow_html=True,
    )


STEPS = [
    ("1", "Upload your CV", "PDF, DOCX or TXT, or paste it"),
    ("2", "Add the jobs you want", "paste postings or search live openings"),
    ("3", "See how well you fit", "a score, your gaps and advice per role"),
]

# Landing banner drawing: a CV and a job posting, with a check for the match.
SCENE = (
    '<svg class="sea-scene" viewBox="138 32 180 140" aria-hidden="true">'
    '<g transform="rotate(8 268 110)">'
    '<rect x="228" y="58" width="82" height="104" rx="8" fill="#CFE3DE" opacity=".55"/>'
    '<rect x="242" y="76" width="44" height="7" rx="3" fill="#0E5E6F" opacity=".5"/>'
    '<rect x="242" y="92" width="54" height="5" rx="2.5" fill="#0E5E6F" opacity=".35"/>'
    '<rect x="242" y="104" width="48" height="5" rx="2.5" fill="#0E5E6F" opacity=".35"/>'
    '<rect x="242" y="116" width="52" height="5" rx="2.5" fill="#0E5E6F" opacity=".35"/></g>'
    '<g transform="rotate(-6 190 110)">'
    '<rect x="146" y="50" width="88" height="112" rx="8" fill="#F2F6F5" opacity=".9"/>'
    '<circle cx="168" cy="74" r="10" fill="#0E5E6F" opacity=".55"/>'
    '<rect x="184" y="68" width="36" height="6" rx="3" fill="#0E5E6F" opacity=".6"/>'
    '<rect x="184" y="79" width="26" height="4" rx="2" fill="#0E5E6F" opacity=".35"/>'
    '<rect x="160" y="96" width="60" height="5" rx="2.5" fill="#0E5E6F" opacity=".3"/>'
    '<rect x="160" y="108" width="52" height="5" rx="2.5" fill="#0E5E6F" opacity=".3"/>'
    '<rect x="160" y="120" width="58" height="5" rx="2.5" fill="#0E5E6F" opacity=".3"/></g>'
    '<circle cx="232" cy="56" r="17" fill="#D9A21B"/>'
    '<path d="M224 56l6 6 11-12" fill="none" stroke="#12262B" stroke-width="3.5" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)


def hero() -> None:
    """Landing banner: what the app does in one line, with the inputs right below it."""
    steps = "".join(
        f'<span class="sea-pill"><span class="sea-pill-n">{n}</span>'
        f'<span><b>{escape(t)}</b>: {escape(d)}</span></span>' for n, t, d in STEPS)
    st.markdown(
        f'<div class="sea-banner">{SCENE}<div class="sea-banner-text">'
        '<h1>See how well your CV fits the job</h1>'
        '<p class="sea-lede">Upload your CV and the jobs you\'re considering. See how well you match '
        'each one, which requirements you meet, and what to work on next.</p>'
        f'<div class="sea-pills">{steps}</div></div></div>',
        unsafe_allow_html=True,
    )


def chips(skills: list[str], kind: str) -> str:
    return "".join(f'<span class="sea-chip {kind}">{escape(s)}</span>' for s in skills)


def score_ring(score: float) -> str:
    color = "var(--deep)" if score >= 75 else "var(--buoy)" if score >= 50 else "var(--mist)"
    return (f'<div class="sea-ring" style="--pct:{score:.0f}; --c:{color}" '
            f'role="img" aria-label="Match score {score:.0f} out of 100">'
            f'<span>{score:.0f}</span></div>')


def profile_panel(profile: CandidateProfile, expanded: bool = False) -> None:
    with st.expander("What we read from your CV", expanded=expanded):
        c1, c2 = st.columns(2)
        c1.metric("Experience", f"{profile.years_experience:g} years")
        c2.metric("Level", profile.seniority.title())
        st.write(profile.summary)
        st.markdown(chips(profile.skills, "have"), unsafe_allow_html=True)


def job_heading(job: Job) -> str:
    meta = ", ".join(x for x in (job.company, job.location, job.employment_type.title()) if x)
    via = "" if not job.source else f" (via {escape(job.source)})" if meta else escape(job.source)
    return (f'<p class="sea-job-title">{escape(job.title)}</p>'
            f'<p class="sea-meta">{escape(meta)}{via}</p>')


def job_previews(jobs: list[Job]) -> None:
    """The jobs before they are scored, with their requirements once those are extracted."""
    st.markdown("#### The jobs we're looking at")
    for job in jobs:
        with st.container(border=True):
            st.markdown(job_heading(job), unsafe_allow_html=True)
            req = job.requirements
            if not req:
                continue
            if req.role_summary:
                st.write(req.role_summary)
            if req.min_years_experience:
                st.caption(f"Asks for at least {req.min_years_experience:g} years of experience")
            if req.required_skills:
                st.markdown("**Required**  " + chips(req.required_skills, "gap"), unsafe_allow_html=True)


def _md(text: str) -> str:
    """Escape CV and posting text (skill names, role titles) so it shows as written in st.markdown:
    '$' would start a formula and '|' would break the labels table."""
    for ch in "\\`*_[]|$#":
        text = text.replace(ch, "\\" + ch)
    return text


def note_markdown(note: notes.Note) -> str:
    parts = [f"**{_md(note.label)}.** {_md(note.text)}"]
    if note.bullets:
        parts.append("\n".join(f"- {_md(b)}" for b in note.bullets))
    for g in note.groups:
        items = ("  \n" + ", ".join(_md(i) for i in g.items) if g.inline
                 else "\n" + "\n".join(f"- {_md(i)}" for i in g.items))
        parts.append(f"**{_md(g.title)}**{items}")
    if note.table:
        head = ["", "Your CV", "This job"] + (["Match"] if len(note.table[0]) > 3 else [])
        parts.append("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
                     + "\n".join("| " + " | ".join(_md(c) for c in row) + " |" for row in note.table))
    if note.after:
        parts.append(_md(note.after))
    return "\n\n".join(parts)


def score_breakdown(r: MatchResult, profile: CandidateProfile) -> None:
    """The three scores, each with the reason for it (core/score_notes.py) and a how-it's-measured tooltip."""
    rows = [("Skill coverage", r.skill_coverage, notes.SKILL_HELP, notes.skill_note(r, profile)),
            ("Profile similarity", r.semantic_similarity, notes.SIMILARITY_HELP,
             notes.similarity_note(r, profile)),
            ("Experience fit", r.seniority_fit,
             notes.EXPERIENCE_HELP if r.relevant_only else notes.TOTAL_HELP, notes.experience_note(r))]
    for name, value, help_text, note in rows:
        num, why = st.columns([1, 3])
        num.metric(name, f"{value:.0%}", help=help_text)
        why.markdown(note_markdown(note))


TOOL_NAMES = {"compute_match_score": "computed the match score",
              "search_cv": "searched your CV"}


def tool_label(call) -> str:
    """'searched your CV for "Tableau"', so the user sees which tools the advice relied on."""
    text = TOOL_NAMES.get(call.name, call.name)
    if call.args.get("by") == "app":
        text += " (done by the app)"
    phrase = call.args.get("phrase")
    return f'{text} for "{phrase}"' if phrase else text


def match_card(r: MatchResult, idx: int, on_explain, profile: CandidateProfile,
               show_breakdown: bool = True, pending: bool = False):
    """One job's card. on_explain(job_id) runs when its advice button is clicked. When pending (advice
    asked for, not written yet), returns an empty slot under the card for the app to show a spinner in."""
    job = r.job
    with st.container(border=True):
        left, right = st.columns([1, 6], vertical_alignment="center")
        left.markdown(score_ring(r.score), unsafe_allow_html=True)
        right.markdown(job_heading(job), unsafe_allow_html=True)

        if r.matched_skills:
            right.markdown("**You have**  " + chips(r.matched_skills, "have"), unsafe_allow_html=True)
        if r.partial_skills:
            right.markdown("**Partly**  " + chips(r.partial_skills, "part"), unsafe_allow_html=True)
        if r.missing_skills:
            right.markdown("**To build**  " + chips(r.missing_skills, "gap"), unsafe_allow_html=True)

        # The breakdown needs no LLM call, only the advice does. Shown open on the top matches;
        # folded on the rest so a long list stays easy to scroll.
        if show_breakdown:
            score_breakdown(r, profile)
        else:
            with st.expander("Score breakdown"):
                score_breakdown(r, profile)

        explanation = st.session_state.get("explanations", {}).get(job.id)
        error = st.session_state.get("explain_errors", {}).get(job.id)
        if explanation:
            st.markdown("**Advice for this role**\n\n" + explanation)
            calls = st.session_state.get("tool_calls", {}).get(job.id) or []
            if calls:
                st.caption("How the AI checked this: " + ", ".join(tool_label(c) for c in calls))
        elif error:
            st.warning(error)
        advice, posting = st.columns(2)
        slot = advice.empty() if pending else None
        if not explanation and not pending:
            advice.button("Get advice for this role", key=f"explain_{idx}", on_click=on_explain, args=(job.id,))
        if job.url:
            posting.link_button("Open job posting", job.url)
    return slot
