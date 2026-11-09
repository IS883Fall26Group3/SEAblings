"""Success metric: requirement classification accuracy (met / partial / missing) on the 16 test pairs.

Compares three labelers with the team's answer key (labels/requirements/<name>.csv, role "main"):
  keyword  baseline: a requirement is met only if its words appear in the CV, otherwise missing
  app      the app itself: retrieval + LLM labels (core/requirements.py) on the frozen requirement
           lists, scored by core/matcher.py exactly as on a job card; the years row is "met" when
           experience fit is full, "partial" when at least half, as the card says. Repeated runs,
           because LLM labels vary a little between runs.
Also: agreement between two labelers on the double-labeled pairs, and the share of the app's quotes
found word for word in the CV (the deterministic check).

usage (from the repo root):
  python eval/requirement_accuracy.py           # answer key checks, labeler agreement, keyword baseline
  python eval/requirement_accuracy.py --app 3   # also the app's labels, 3 runs (needs GEMINI_API_KEY)
  python eval/requirement_accuracy.py --set practice --app 1   # the practice set (practice_set.py),
                                                               # where changes are tuned
Writes results/requirement_accuracy.csv or results/practice_accuracy.csv (labels only, no CV text).
The app's labels are cached per run and per VERSION; bump VERSION whenever the labeling code or prompt
changes, so labels from older code are never reused."""
import csv
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from requirement_set import HERE, LABELS, PAIRS, PEOPLE  # noqa: F401  (also loads secrets)
import practice_set

from core.config import get_settings  # noqa: E402
from core.knowledge import SkillKB  # noqa: E402
from core.llm import build_embedder, build_provider  # noqa: E402
from core.matcher import score_jobs  # noqa: E402
from core.requirements import retrieve_evidence  # noqa: E402
from core.schemas import (CandidateProfile, Job, JobRequirements, RequirementCheck, Role)  # noqa: E402
from core.skill_extractor import judge_experience, label_requirements  # noqa: E402
from extract import load_cvs, patient  # noqa: E402
from skillmatch import embed_cached  # noqa: E402

OUT = HERE / "results"
LABELS3 = ("met", "partial", "missing")
MODEL = "gemini-flash-lite-latest"
VERSION = "v2"  # v1: as first committed. v2: word-match retrieval, prompt aligned with labeling_guide.md

# name: (pairs as (pair, CV, job, kind), answer-key files, frozen requirement lists, cache prefix, output)
SETS = {
    "test": ([p[:4] for p in PAIRS], lambda: sorted(f for f in PEOPLE.glob("*.csv")
                                                   if not f.stem.endswith("_extraction")),
             LABELS / "requirements_frozen.json", "reqset", "requirement_accuracy.csv"),
    "practice": (practice_set.PRACTICE, lambda: [practice_set.KEY], practice_set.FROZEN, "practice",
                 "practice_accuracy.csv"),
}


# ---------- the answer key ----------
def load_key(files):
    """main[(pair, requirement)] = row with the label; second[(pair, requirement)] = second label."""
    main, second, problems = {}, {}, []
    for f in files:
        for i, row in enumerate(csv.DictReader(f.open(encoding="utf-8-sig")), start=2):
            label = row["label"].strip().lower()
            if label not in LABELS3:
                problems.append(f"{f.name} line {i}: label '{row['label']}'")
                continue
            key = (row["pair"], row["requirement"])
            (main if row["role"] == "main" else second)[key] = dict(row, label=label, labeler=f.stem)
    return main, second, problems


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa: agreement corrected for the agreement two random labelers would reach."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS3) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------- keyword baseline ----------
def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip() + " "


def keyword_label(requirement: str, cv_text: str) -> str:
    return "met" if _words(requirement) in _words(cv_text) else "missing"


# ---------- the app ----------
def frozen_jobs(path) -> dict[str, Job]:
    frozen = json.loads(path.read_text(encoding="utf-8"))["jobs"]
    raw = {j["id"]: j for j in json.loads((HERE / "data" / "jobs" / "jobs_jsearch.json").read_text(encoding="utf-8"))}
    return {jid: Job(**raw[jid], requirements=JobRequirements(
        job_id=jid, required_skills=fz["required_skills"], nice_to_have_skills=[],
        min_years_experience=fz["min_years_experience"])) for jid, fz in frozen.items()}


def cached(path, key, fn, label):
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if key not in done:
        out = patient(fn, label, 13.0)
        if out is None:
            raise SystemExit(f"{label} failed; run again to resume.")
        done[key] = out
        path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        print(label, "done", flush=True)
    return done[key]


def app_labels(runs: int, pairs, frozen_path, prefix: str, label_model: str = MODEL) -> list[dict]:
    """One dict per run: (pair, requirement) -> (label, quote verified or None)."""
    s = get_settings()
    emb = build_embedder(s)
    def embed(texts: list[str]) -> np.ndarray:  # cached on disk, paced for the free tier
        vec = embed_cached(emb, "gemini", texts)
        return np.stack([vec[t] for t in texts])

    kb = SkillKB.build(emb.name, embed)
    llm = build_provider(f"gemini:{MODEL}", s)
    labeler = build_provider(f"gemini:{label_model}", s)
    cvs, jobs = load_cvs(), frozen_jobs(frozen_path)
    profiles = json.loads((OUT / f"extracted_{MODEL}.json").read_text(encoding="utf-8"))["cvs"]
    by_cv = defaultdict(list)
    for pid, cv, jid, *_ in pairs:
        by_cv[cv].append((pid, jid))

    judge_file = OUT / f"{prefix}_judge_{MODEL}.json"  # judged once: it only feeds the years rows
    out = []
    for run in range(1, runs + 1):
        run_file = OUT / f"{prefix}_labels_{label_model}_{VERSION}_run{run}.json"
        labels = {}
        for cv, items in sorted(by_cv.items()):
            cv_jobs = [jobs[jid].model_copy(deep=True) for _, jid in items]
            judged = cached(judge_file, cv, lambda: {jid: x.model_dump() for jid, x in judge_experience(
                llm, cvs[cv], cv_jobs, s.max_jd_chars, s.local_max_jd_chars).items()}, f"judge {cv}")
            retrieved = retrieve_evidence(cv_jobs, cvs[cv], embed, kb)
            checks = cached(run_file, cv, lambda: {jid: [c.model_dump() for c in cs] for jid, cs in
                                                   label_requirements(labeler, cv_jobs, retrieved, cvs[cv]).items()},
                            f"run {run} labels {cv}")
            results = score_jobs(
                CandidateProfile(**profiles[cv]), cvs[cv], cv_jobs, emb, s, embed_fn=embed,
                relevant_roles={j: [Role(**r) for r in x["relevant_roles"]] for j, x in judged.items()},
                background={j: (x["field_match"], x["work_match"]) for j, x in judged.items()},
                requirement_checks={j: [RequirementCheck(**c) for c in cs] for j, cs in checks.items()})
            by_job = {r.job.id: r for r in results}
            for pid, jid in items:
                r = by_job[jid]
                quotes = {c.requirement: c.quote_verified for c in r.requirement_checks
                          if c.status in ("met", "partial")}
                for skill in r.job.requirements.required_skills:
                    lab = ("met" if skill in r.matched_skills else
                           "partial" if skill in r.partial_skills else "missing")
                    labels[(pid, skill)] = (lab, quotes.get(skill))
                mn = r.job.requirements.min_years_experience
                if mn:
                    pct = round(100 * r.seniority_fit)
                    labels[(pid, f"{mn:g}+ years of relevant experience")] = (
                        "met" if pct == 100 else "partial" if pct >= 50 else "missing", None)
        out.append(labels)
    return out


# ---------- reporting ----------
def report(name: str, key: dict, pred: dict[tuple, str]) -> dict:
    rows = [(k, v["label"], pred[k]) for k, v in key.items() if k in pred]
    acc = sum(t == p for _, t, p in rows) / len(rows)
    recall = {c: (sum(t == p == c for _, t, p in rows) / n) if (n := sum(t == c for _, t, _ in rows)) else None
              for c in LABELS3}
    bal = np.mean([x for x in recall.values() if x is not None])
    parts = {}
    for field in ("type", "kind"):
        for val in sorted({key[k][field] for k, _, _ in rows}):
            sub = [(t, p) for k, t, p in rows if key[k][field] == val]
            parts[f"{val}"] = (sum(t == p for t, p in sub) / len(sub), len(sub))
    print(f"\n{name}: accuracy {acc:.1%} ({sum(t == p for _, t, p in rows)}/{len(rows)}), "
          f"balanced accuracy {bal:.1%}")
    print("  share of each team label the labeler got right: "
          + ", ".join(f"{c} {r:.0%}" for c, r in recall.items() if r is not None))
    print("  by part: " + ", ".join(f"{k} {a:.0%} (n={n})" for k, (a, n) in parts.items()))
    confusion = Counter((t, p) for _, t, p in rows)
    print("  team label (rows) x this labeler (columns):  " + "  ".join(f"{c:>7}" for c in LABELS3))
    for t in LABELS3:
        print(f"  {t:>44}  " + "  ".join(f"{confusion[(t, p)]:>7}" for p in LABELS3))
    return {"accuracy": acc, "balanced": bal}


def main() -> None:
    name = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "test"
    pairs, key_files, frozen_path, prefix, out_name = SETS[name]
    print(f"set: {name}, labeling code version {VERSION}")
    main_key, second, problems = load_key(key_files())
    expected = sum(1 for _ in main_key) + len(problems)
    print(f"answer key: {len(main_key)} labeled requirements in {len({k[0] for k in main_key})} pairs "
          f"({dict(Counter(v['label'] for v in main_key.values()))})")
    for p in problems:
        print("  not counted:", p)
    if not main_key or expected == 0:
        raise SystemExit("No labels yet.")

    both = [k for k in second if k in main_key]
    if both:
        a = [main_key[k]["label"] for k in both]
        b = [second[k]["label"] for k in both]
        agree = sum(x == y for x, y in zip(a, b)) / len(both)
        print(f"\nlabeler agreement on {len({k[0] for k in both})} double-labeled pairs ({len(both)} "
              f"requirements): {agree:.0%} the same label, Cohen's kappa {kappa(a, b):.2f}")
        for k in both:
            if main_key[k]["label"] != second[k]["label"]:
                print(f"  {k[0]} {k[1]}: {main_key[k]['labeler']} {main_key[k]['label']}, "
                      f"{second[k]['labeler']} {second[k]['label']}")

    cvs = load_cvs()
    cv_of = {p[0]: p[1] for p in pairs}
    keyword = {k: keyword_label(k[1], cvs[cv_of[k[0]]]) for k in main_key}
    report("keyword baseline", main_key, keyword)

    runs = int(sys.argv[sys.argv.index("--app") + 1]) if "--app" in sys.argv else 0
    label_model = sys.argv[sys.argv.index("--label-model") + 1] if "--label-model" in sys.argv else MODEL
    print("labels by", label_model)
    app_runs = app_labels(runs, pairs, frozen_path, prefix, label_model) if runs else []
    scores = [report(f"app, run {i}", main_key, {k: v[0] for k, v in run.items()})
              for i, run in enumerate(app_runs, 1)]
    if scores:
        accs = [x["accuracy"] for x in scores]
        print(f"\napp over {len(accs)} runs: accuracy mean {np.mean(accs):.1%} "
              f"(min {min(accs):.1%}, max {max(accs):.1%}); target 75%")
        q = [v[1] for run in app_runs for v in run.values() if v[1] is not None]
        print(f"quotes found word for word in the CV: {sum(q)}/{len(q)} ({sum(q) / len(q):.0%}) of the app's "
              "met / partial labels that came with a quote check")

    with (OUT / out_name).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "kind", "cv", "job_id", "type", "requirement", "team", "keyword"]
                   + [f"app_run{i}" for i in range(1, len(app_runs) + 1)])
        for k, v in main_key.items():
            w.writerow([k[0], v["kind"], v["cv"], v["job_id"], v["type"], k[1], v["label"], keyword[k]]
                       + [run.get(k, ("", None))[0] for run in app_runs])


if __name__ == "__main__":
    main()
