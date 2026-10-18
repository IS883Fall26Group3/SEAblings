"""Reusable UI pieces. Keep presentation here so streamlit_app.py stays readable."""
from html import escape
from pathlib import Path

import streamlit as st

from core.schemas import CandidateProfile, MatchResult

CSS = Path(__file__).with_name("styles.css")


def inject_css() -> None:
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def app_header() -> None:
    """Brand bar shown on every screen, so the app's purpose is clear at a glance."""
    st.markdown(
        """<div class="sea-header">
        <div class="sea-brand"><span class="sea-logo" aria-hidden="true">🌊</span>
          <span class="sea-name">SEAblings</span>
          <span class="sea-tag">AI CV-to-job matcher</span></div>
        <div class="sea-team">IS883 · Fall 2026 · Group 3</div>
        </div>""",
        unsafe_allow_html=True,
    )


STEPS = [
    ("1", "Upload your CV", "PDF, DOCX or TXT. AI reads your skills, experience and level."),
    ("2", "Search live openings", "Current postings for your target role and location, "
                                  "from Google for Jobs (incl. Indeed and LinkedIn)."),
    ("3", "See your best fits", "Jobs ranked by match score, with the skills you have, "
                                "the gaps to close, and advice for each role."),
]


def hero() -> None:
    steps = "".join(
        f'<div class="sea-step"><span class="sea-step-n">{n}</span>'
        f'<p class="sea-step-t">{t}</p><p class="sea-step-d">{d}</p></div>'
        for n, t, d in STEPS)
    st.markdown(
        f"""<div class="sea-hero">
        <h1>Find the jobs your CV is ready for</h1>
        <p>Stop guessing which postings fit. Upload your CV, name the role and place you want,
        and get today's openings ranked by how well you match, plus what to learn next.</p>
        </div>
        <div class="sea-steps">{steps}</div>
        <div class="sea-score-note"><b>How the score works:</b>
        50% skill coverage + 30% profile similarity + 20% experience fit.
        The same CV and job always get the same score.</div>
        <p class="sea-start">👈 Start in the sidebar (tap » on a phone): add your CV (or use our sample),
        enter a role, and click <b>Find matching jobs</b>.</p>""",
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


def match_card(r: MatchResult, idx: int, on_explain) -> None:
    job = r.job
    with st.container(border=True):
        left, right = st.columns([1, 6], vertical_alignment="center")
        left.markdown(score_ring(r.score), unsafe_allow_html=True)
        meta = ", ".join(x for x in (job.company, job.location, job.employment_type.title()) if x)
        via = f" (via {escape(job.source)})" if job.source else ""
        right.markdown(
            f'<p class="sea-job-title">{escape(job.title)}</p>'
            f'<p class="sea-meta">{escape(meta)}{via}</p>', unsafe_allow_html=True)

        if r.matched_skills:
            right.markdown("**You have**  " + chips(r.matched_skills, "have"), unsafe_allow_html=True)
        if r.missing_skills:
            right.markdown("**To build**  " + chips(r.missing_skills, "gap"), unsafe_allow_html=True)

        with st.expander("Score breakdown and advice"):
            a, b, c = st.columns(3)
            a.metric("Skill coverage", f"{r.skill_coverage:.0%}")
            b.metric("Profile similarity", f"{r.semantic_similarity:.0%}")
            c.metric("Experience fit", f"{r.seniority_fit:.0%}")
            explanation = st.session_state.get("explanations", {}).get(job.id)
            error = st.session_state.get("explain_errors", {}).get(job.id)
            if explanation:
                st.markdown(explanation)
            else:
                if error:
                    st.warning(error)
                if st.button("Get advice for this role", key=f"explain_{idx}"):
                    on_explain(r)
                    st.rerun()
            if job.url:
                st.link_button("Open job posting", job.url)
