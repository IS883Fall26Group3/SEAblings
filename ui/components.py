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


def limitations(s) -> None:
    """What the app can't do, so users read the score with the right expectations."""
    search = ("Live search covers US job listings only, up to 10 jobs per search."
              if s.job_source == "jsearch" else
              "Job search uses fictional sample listings in this setup, not live openings.")
    items = [
        "The score is based on your CV text only. A skill your CV doesn't mention counts as not found.",
        "Scanned PDFs (images of text) can't be read. Upload a text-based PDF, DOCX or TXT.",
        f"The AI reads up to {s.max_cv_chars:,} characters of your CV and {s.max_jd_chars:,} of each posting.",
        search,
        "The free Gemini models have daily limits. Auto switches to another model when one runs out; "
        "the offline model is slower and less accurate.",
        "AI advice can be wrong. Check it before you change your CV.",
    ]
    with st.container(border=True, key="limits_section"):
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


FEATURES = [
    ("📄", "CV and job match"),
    ("✅", "Requirement-by-requirement check"),
    ("📊", "Score out of 100", "50% skill coverage + 30% profile similarity + 20% experience fit. "
                               "The same CV and job always get the same score."),
    ("💬", "Advice for each role"),
]

# Sea scene for the landing banner: moon, clouds, a CV and a job posting floating on the waves.
SCENE = (
    '<svg class="sea-scene" viewBox="0 0 420 210" aria-hidden="true">'
    '<circle cx="372" cy="46" r="15" fill="#F4E3A1"/>'
    '<ellipse cx="318" cy="52" rx="34" ry="7" fill="#fff" opacity=".18"/>'
    '<ellipse cx="350" cy="62" rx="26" ry="6" fill="#fff" opacity=".14"/>'
    '<ellipse cx="120" cy="40" rx="30" ry="6" fill="#fff" opacity=".14"/>'
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
    '<path d="M0 150 Q35 136 70 150 T140 150 T210 150 T280 150 T350 150 T420 150 V210 H0Z" '
    'fill="#fff" opacity=".12"/>'
    '<path d="M0 166 Q35 152 70 166 T140 166 T210 166 T280 166 T350 166 T420 166 V210 H0Z" '
    'fill="#CFE3DE" opacity=".22"/>'
    '<path d="M0 184 Q35 172 70 184 T140 184 T210 184 T280 184 T350 184 T420 184 V210 H0Z" '
    'fill="#CFE3DE" opacity=".35"/>'
    '<g transform="translate(372 150)"><rect x="-7" y="-20" width="14" height="22" rx="4" fill="#D9A21B"/>'
    '<rect x="-7" y="-12" width="14" height="5" fill="#F2F6F5"/>'
    '<line x1="0" y1="-20" x2="0" y2="-30" stroke="#F2F6F5" stroke-width="2"/></g>'
    '</svg>'
)


def hero() -> None:
    """Landing banner: what the app does in one line, with the inputs right below it."""
    pills = "".join(
        f'<span class="sea-pill"{f" title={chr(34)}{escape(f[2])}{chr(34)}" if len(f) > 2 else ""}>'
        f'<span aria-hidden="true">{f[0]}</span> {escape(f[1])}</span>' for f in FEATURES)
    st.markdown(
        f'<div class="sea-banner">{SCENE}<div class="sea-banner-text">'
        '<p class="sea-kicker">🌊 SEAblings · AI CV-to-job matcher</p>'
        '<h1>See how well your CV fits the job</h1>'
        '<p class="sea-lede">Upload your CV and the jobs you\'re considering. See how well you match '
        'each one, which requirements you meet, and what to work on next.</p>'
        f'<div class="sea-pills">{pills}</div></div></div>',
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
               show_breakdown: bool = True) -> None:
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
        if not explanation and advice.button("Get advice for this role", key=f"explain_{idx}"):
            with advice, st.spinner("Writing advice for this role..."):
                on_explain(r)
            st.rerun()
        if job.url:
            posting.link_button("Open job posting", job.url)
