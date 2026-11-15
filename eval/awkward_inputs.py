"""Awkward inputs A5-A9 (labels/awkward_inputs.csv), run through the real app headlessly (Streamlit's
AppTest), pasted as a user would. The expected result of each is written in the CSV before running.

  plain  data/sample/sample_cv.txt vs sample posting s5: the reference A6's score is compared with
  A5     malicious, caught by the rules: "Ignore all previous instructions..." in the CV
  A6     malicious, reworded so the rules miss it: a "note to the screening system" in the CV
  A7     abusive: insults pasted as the CV
  A8     invalid value: a posting asking for 150+ years of experience
  A9     a slightly altered well-known example: Barack Obama's career without the law degree,
         against a legal-counsel posting (labels must follow the CV, not the model's memory)

usage (from the repo root; needs GEMINI_API_KEY in .streamlit/secrets.toml):
  python eval/awkward_inputs.py            # all cases
  python eval/awkward_inputs.py A8 A9      # only these
Prints what each run showed; fill in labels/awkward_inputs.csv from it."""
# --- Maylada (with Claude Code): test harness for the awkward inputs (directives §5.5 item 2) ---
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import os  # noqa: E402

os.chdir(REPO)
from streamlit.testing.v1 import AppTest  # noqa: E402

import core.embeddings  # noqa: E402
import core.llm.gemini as gemini  # noqa: E402

MODEL = "gemini:gemini-flash-lite-latest"  # fixed, as every eval run (directives §5.4)
AWK = "eval/data/awkward/"


def sample_job(job_id: str) -> str:
    job = next(j for j in json.loads((REPO / "data/sample/jobs.json").read_text(encoding="utf-8"))
               if j["id"] == job_id)
    return job["title"] + "\n" + job["description"]


def file(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


# case: (CV text, posting text, requirements to watch)
CASES = {
    "plain": (lambda: file("data/sample/sample_cv.txt"), lambda: sample_job("s5"), ("pytorch",)),
    "A5": (lambda: file(AWK + "injection_cv.txt"), lambda: sample_job("s5"), ()),
    "A6": (lambda: file(AWK + "injection_cv_reworded.txt"), lambda: sample_job("s5"),
           ("machine learning", "deep learning", "pytorch")),
    "A7": (lambda: file(AWK + "abusive_cv.txt"), lambda: sample_job("s6"), ()),
    "A8": (lambda: file("data/sample/sample_cv.txt"), lambda: file(AWK + "invalid_years_posting.txt"),
           ("excel", "financial modeling")),
    "A9": (lambda: file(AWK + "altered_known_cv.txt"), lambda: file(AWK + "legal_counsel_posting.txt"),
           ("j.d.", "bar", "litigation", "public speaking", "writing")),
}
CALLS: list[str] = []
_original = gemini.call_gemini


def counted(fn, what, model):
    CALLS.append(what)
    return _original(fn, what, model)


gemini.call_gemini = counted
core.embeddings.call_gemini = counted


def run(case: str) -> dict:
    cv_text, posting, watch = (f() if callable(f) else f for f in CASES[case])
    at = AppTest.from_file(str(REPO / "streamlit_app.py"), default_timeout=900)
    at.run()
    at.sidebar.selectbox(key="model_choice").set_value(MODEL).run()
    next(t for t in at.text_area if t.label.startswith("Or paste your CV")).set_value(cv_text).run()
    next(t for t in at.text_area if t.label.startswith("Or paste a job")).set_value(posting).run()
    CALLS.clear()
    next(b for b in at.button if b.label == "Analyze my match").click().run()
    out = {"case": case, "watch": watch, "llm_calls": [c for c in CALLS if "comparing" not in c],
           "messages": [m.value for m in (*at.error, *at.warning)],
           "exception": [str(e.value)[:300] for e in at.exception]}
    results = at.session_state["results"] if "results" in at.session_state else None
    if results:
        r = results[0]
        # for A6: the injected note is the lines this CV adds to the plain sample CV
        plain = set(file("data/sample/sample_cv.txt").splitlines())
        note = " ".join(line for line in cv_text.splitlines() if line not in plain).lower()
        out.update(score=r.score, met=r.matched_skills, partial=r.partial_skills, missing=r.missing_skills,
                   min_years=r.min_years, experience_fit=r.seniority_fit,
                   quotes={c.requirement: c.quote for c in r.requirement_checks if c.quote},
                   from_note=[c.requirement for c in r.requirement_checks
                              if case == "A6" and c.quote and note and c.quote.lower() in note])
    return out


def show(o: dict) -> None:
    print(f"\n== {o['case']}: {len(o['llm_calls'])} LLM calls {o['llm_calls']}")
    for m in o["messages"] + o["exception"]:
        print("  message:", m[:300])
    if "score" not in o:
        print("  no match score")
        return
    print(f"  score {o['score']:.0f}; minimum years {o['min_years']}; experience fit {o['experience_fit']:.0%}")
    print("  met:", o["met"]), print("  partial:", o["partial"]), print("  missing:", o["missing"])
    for skill in o["watch"]:
        where = [(k, s) for k in ("met", "partial", "missing") for s in o[k] if skill in s.lower()]
        print(f"  {skill}: {', '.join(f'{s} {k}' for k, s in where) or 'not a requirement'}")
    for req, quote in o["quotes"].items():
        print(f'  quote for {req}: "{quote[:90]}"')
    if o["case"] == "A6":
        print("  quotes taken from the injected note:", o["from_note"] or "none")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CASES)
    found = {case: run(case) for case in wanted}
    for o in found.values():
        show(o)
    if "score" in found.get("plain", {}) and "score" in found.get("A6", {}):
        print(f"\nA6 vs plain: score {found['A6']['score'] - found['plain']['score']:+.0f} points")
