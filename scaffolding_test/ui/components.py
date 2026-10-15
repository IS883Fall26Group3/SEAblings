"""Reusable UI pieces. Keep presentation here so streamlit_app.py stays readable."""
from html import escape
from pathlib import Path

import streamlit as st

from core.schemas import CandidateProfile, MatchResult

CSS = Path(__file__).with_name("styles.css")


def inject_css() -> None:
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """<div class="sea-hero">
        <h1>Find roles that fit your CV</h1>
        <p>Upload your CV, tell us the role and location you want, and see current openings
        ranked by how well your skills match, with the gaps to close for each one.</p>
        </div>""",
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
