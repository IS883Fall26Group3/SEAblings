"""Reusable UI pieces. Keep presentation here so streamlit_app.py stays readable."""
from html import escape
from pathlib import Path

import streamlit as st

from core import score_notes as notes
from core.schemas import CandidateProfile, MatchResult

CSS = Path(__file__).with_name("styles.css")


def inject_css() -> None:
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


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
    ("1", "Upload your CV", "PDF, DOCX or TXT. AI reads your skills, experience and level."),
    ("2", "Add the jobs you want", "Upload or paste job descriptions, or search live "
                                   "openings for a role and location."),
    ("3", "See how well you fit", "A match score for each job, the skills you have, "
                                  "the gaps to close, and advice for the role."),
]


def hero() -> None:
    steps = "".join(
        f'<div class="sea-step"><span class="sea-step-n">{n}</span>'
        f'<p class="sea-step-t">{t}</p><p class="sea-step-d">{d}</p></div>'
        for n, t, d in STEPS)
    st.markdown(
        f"""<div class="sea-hero">
        <h1>See how well your CV fits the job</h1>
        <p>Stop guessing whether you're a fit. Upload your CV and the job descriptions you're
        considering, and see how well you match each one, plus what to learn next.</p>
        </div>
        <div class="sea-steps">{steps}</div>
        <div class="sea-score-note"><b>How the score works:</b>
        50% skill coverage + 30% profile similarity + 20% experience fit.
        The same CV and job always get the same score.</div>""",
        unsafe_allow_html=True,
    )


def chips(skills: list[str], kind: str) -> str:
    return "".join(f'<span class="sea-chip {kind}">{escape(s)}</span>' for s in skills)


def score_ring(score: float) -> str:
    color = "var(--deep)" if score >= 75 else "var(--buoy)" if score >= 50 else "var(--mist)"
    return (f'<div class="sea-ring" style="--pct:{score:.0f}; --c:{color}" '
            f'role="img" aria-label="Match score {score:.0f} out of 100">'
            f'<span>{score:.0f}</span></div>')


def profile_panel(profile: CandidateProfile) -> None:
    with st.expander("What we read from your CV", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Experience", f"{profile.years_experience:g} years")
        c2.metric("Level", profile.seniority.title())
        st.write(profile.summary)
        st.markdown(chips(profile.skills, "have"), unsafe_allow_html=True)


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
    if note.table:
        parts.append("| | Your CV | This job |\n|---|---|---|\n"
                     + "\n".join(f"| {_md(a)} | {_md(b)} | {_md(c)} |" for a, b, c in note.table))
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


def match_card(r: MatchResult, idx: int, on_explain, profile: CandidateProfile,
               show_breakdown: bool = True) -> None:
    job = r.job
    with st.container(border=True):
        left, right = st.columns([1, 6], vertical_alignment="center")
        left.markdown(score_ring(r.score), unsafe_allow_html=True)
        meta = ", ".join(x for x in (job.company, job.location, job.employment_type.title()) if x)
        via = "" if not job.source else f" (via {escape(job.source)})" if meta else escape(job.source)
        right.markdown(
            f'<p class="sea-job-title">{escape(job.title)}</p>'
            f'<p class="sea-meta">{escape(meta)}{via}</p>', unsafe_allow_html=True)

        if r.matched_skills:
            right.markdown("**You have**  " + chips(r.matched_skills, "have"), unsafe_allow_html=True)
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
        elif error:
            st.warning(error)
        advice, posting = st.columns(2)
        if not explanation and advice.button("Get advice for this role", key=f"explain_{idx}"):
            on_explain(r)
            st.rerun()
        if job.url:
            posting.link_button("Open job posting", job.url)
