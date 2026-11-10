"""Efficiency: Gemini calls, tokens and seconds for one search in the real app, then one advice
request (tool calling). Drives streamlit_app.py headlessly (Streamlit's AppTest) and records every
Gemini call the app makes, without changing the app: it wraps core.llm.gemini.call_gemini.

usage (from the repo root; needs GEMINI_API_KEY in .streamlit/secrets.toml):
  python eval/measure_cost.py one     # the default sample CV vs 1 pasted sample posting (s6)
  python eval/measure_cost.py ten     # a sample CV vs all 10 sample postings, then advice on the top job
Run each in its own process, so no cached result from an earlier run hides a call.
Prints a table and appends rows to results/cost_per_search.csv (no CV text)."""
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import os  # noqa: E402

os.chdir(REPO)
from streamlit.testing.v1 import AppTest  # noqa: E402

import core.embeddings  # noqa: E402
import core.llm.gemini as gemini  # noqa: E402

MODEL = "gemini:gemini-flash-lite-latest"
OUT = REPO / "eval" / "results" / "cost_per_search.csv"
CALLS: list[dict] = []
PHASE = ["search"]
_original = gemini.call_gemini


def counted(fn, what, model):
    start = time.time()
    result = _original(fn, what, model)
    u = getattr(result, "usage_metadata", None)
    texts = len(getattr(result, "embeddings", None) or [])
    CALLS.append({"phase": PHASE[0], "step": what, "model": model, "seconds": round(time.time() - start, 2),
                  "input_tokens": getattr(u, "prompt_token_count", None) or 0,
                  "output_tokens": (getattr(u, "candidates_token_count", None) or 0)
                  + (getattr(u, "thoughts_token_count", None) or 0),
                  "texts_embedded": texts})
    return result


gemini.call_gemini = counted          # used by the Gemini chat models
core.embeddings.call_gemini = counted  # and by the Gemini embedder


def run(case: str) -> None:
    at = AppTest.from_file(str(REPO / "streamlit_app.py"), default_timeout=900)
    at.run()
    at.sidebar.selectbox(key="model_choice").set_value(MODEL).run()
    if case == "one":
        job = json.loads((REPO / "data/sample/jobs.json").read_text(encoding="utf-8"))[5]
        jd = next(t for t in at.text_area if t.label.startswith("Or paste a job"))
        jd.set_value(job["title"] + "\n" + job["description"]).run()
    else:
        at.selectbox(key="sample_cv").set_value("Mei Chen: product analytics, fintech").run()
    t = time.time()
    next(b for b in at.button if b.label in ("Analyze my match", "Find matching jobs")).click().run()
    wall = {"search": time.time() - t}
    if at.exception or at.error:
        raise SystemExit(f"app error: {[e.value for e in at.error]} {[str(e.value)[:300] for e in at.exception]}")
    n_jobs = len(at.session_state["results"])
    if case == "ten":
        PHASE[0] = "advice"
        t = time.time()
        next(b for b in at.button if b.label == "Get advice for this role").click().run()
        wall["advice"] = time.time() - t
        ss = at.session_state
        text = next(iter(ss["explanations"].values()), "") if "explanations" in ss else ""
        tools = [c.name for v in (ss["tool_calls"].values() if "tool_calls" in ss else []) for c in v]
        errors = list(ss["explain_errors"].values()) if "explain_errors" in ss else []
        print(f"advice: {len(text.split())} words; tools called: {tools}; errors: {errors}")

    print(f"\ncase {case}: 1 CV vs {n_jobs} job(s), model {MODEL}")
    print(f"{'phase':7} {'step':45} {'sec':>6} {'in tok':>7} {'out tok':>7} {'texts':>6}")
    for c in CALLS:
        print(f"{c['phase']:7} {c['step'][:45]:45} {c['seconds']:6.1f} {c['input_tokens']:7} "
              f"{c['output_tokens']:7} {c['texts_embedded']:6}")
    for phase, secs in wall.items():
        cs = [c for c in CALLS if c["phase"] == phase]
        llm = [c for c in cs if not c["texts_embedded"]]
        emb = [c for c in cs if c["texts_embedded"]]
        print(f"{phase}: {len(llm)} LLM calls, {sum(c['input_tokens'] for c in llm)} input + "
              f"{sum(c['output_tokens'] for c in llm)} output tokens; {len(emb)} embedding calls for "
              f"{sum(c['texts_embedded'] for c in emb)} texts (each text counts as 1 request of the "
              f"1,000 a day); {secs:.0f}s wall time, {sum(c['seconds'] for c in cs):.0f}s waiting on Gemini")
    new = not OUT.exists()
    with OUT.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["case", "jobs", "phase", "step", "model", "seconds", "input_tokens",
                               "output_tokens", "texts_embedded"])
        if new:
            w.writeheader()
        w.writerows({"case": case, "jobs": n_jobs, **c} for c in CALLS)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "one")
