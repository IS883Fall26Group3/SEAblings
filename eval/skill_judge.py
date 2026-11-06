"""Can an LLM decide skill matches better than the embedding threshold?

Asks the LLM to label every pair in labels/skill_pairs.csv (one call, blind to the reviewers'
labels, Claude's drafts and the cosines), with the same rule the reviewers used, then compares
with the reviewers' labels:
  - the app today: same standard name, else cosine >= skill_threshold
  - the LLM alone on every pair
  - hybrid: the app's rule outside a borderline cosine band, the LLM inside it
Each model runs twice, to see how often its answers change between runs.
Answers are cached in results/skill_judge_<model>.json.

usage (from the repo root): python eval/skill_judge.py [gemini model ...]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from pydantic import BaseModel

from skillmatch import HERE, OUT, build, embed_cached

sys.path.insert(0, str(HERE.parent))
from core.config import get_settings  # noqa: E402
from core.llm.gemini import GeminiProvider  # noqa: E402
from core.skill_names import skill_key  # noqa: E402

BANDS = [(0.88, 0.95), (0.90, 0.95), (0.90, 0.94)]
RUNS = 2
TIMEOUT_S = 180  # the app allows 60s; gemini-flash-latest thinks longer than that on 116 pairs

PROMPT = """You are a recruiter screening CVs. For each numbered pair, a job requires JOB SKILL and
a candidate's CV lists CV SKILL. Answer "yes" if you would accept the CV skill as covering the job
skill: the same skill, an alias or abbreviation, or a specific instance of it (PostgreSQL covers
SQL). Answer "no" if it is only related, broader, or an alternative tool (Tableau does not cover
Power BI; HTML does not cover CSS).
The job title is context only.

{pairs}"""


class Answer(BaseModel):
    id: int
    covered: str  # "yes" or "no"


class Answers(BaseModel):
    items: list[Answer]


def ask(llm, df: pd.DataFrame) -> list[bool]:
    pairs = "\n".join(f"{i}. JOB SKILL: {r.job_skill} | CV SKILL: {r.cv_skill} | job: {r.example_job}"
                      for i, r in enumerate(df.itertuples()))
    got = llm.generate_json(PROMPT.format(pairs=pairs), Answers, what="labeling skill pairs",
                            temperature=0.0)
    by_id = {a.id: a.covered.strip().lower() == "yes" for a in got.items}
    missing = [i for i in range(len(df)) if i not in by_id]
    if missing:
        print(f"  {len(missing)} pairs unanswered (counted as no): {missing[:10]}")
    return [by_id.get(i, False) for i in range(len(df))]


def score(name: str, says: np.ndarray, y: np.ndarray):
    tp, fp, fn = (says & y).sum(), (says & ~y).sum(), (~says & y).sum()
    p, r = tp / max(says.sum(), 1), tp / max(y.sum(), 1)
    f05 = 1.25 * p * r / max(0.25 * p + r, 1e-9)
    agree = (says == y).mean()
    print(f"  {name:34} agrees {agree:.0%}  right {tp:2}  loose {fp:2}  missed {fn:2}  "
          f"precision {p:.2f}  recall {r:.2f}  F0.5 {f05:.2f}")


def main(models: list[str]):
    s = get_settings()
    emb = build("gemini")
    df = pd.read_csv(HERE / "labels" / "skill_pairs.csv", dtype=str, keep_default_na=False)
    r1, r2 = df.reviewer_1.str.strip().str.lower(), df.reviewer_2.str.strip().str.lower()
    df = df[(r1 != "") & (r1 == r2)].reset_index(drop=True)
    y = (df.reviewer_1.str.strip().str.lower() == "yes").values
    vec = embed_cached(emb, "gemini", sorted(set(df.job_skill) | set(df.cv_skill)))
    c = np.array([float(vec[a] @ vec[b]) for a, b in zip(df.job_skill, df.cv_skill)])
    same = np.array([skill_key(a) == skill_key(b) for a, b in zip(df.job_skill, df.cv_skill)])
    app = same | (c >= emb.skill_threshold)
    print(f"{len(df)} pairs where the reviewers agree, {y.sum()} yes")
    print("app today (standard name, else cosine >= %.3f):" % emb.skill_threshold)
    score("app", app, y)

    for model in models:
        cache = OUT / f"skill_judge_{model}.json"
        runs = json.loads(cache.read_text()) if cache.exists() else []
        llm = GeminiProvider(s.gemini_api_key, model, TIMEOUT_S)
        while len(runs) < RUNS:
            start = time.time()
            runs.append(ask(llm, df))
            cache.write_text(json.dumps(runs))
            print(f"{model} run {len(runs)}: {time.time() - start:.0f}s", flush=True)
            time.sleep(13)  # free tier: about 5 requests a minute
        runs = [np.array(r) for r in runs]
        flips = (runs[0] != runs[1])
        print(f"\n{model}: answers that changed between the two runs: {flips.sum()} of {len(df)}")
        for k, says in enumerate(runs, 1):
            score(f"LLM alone, run {k}", says, y)
            for lo, hi in BANDS:
                band = ~same & (c >= lo) & (c < hi)
                score(f"hybrid, LLM on {lo:.2f}-{hi:.2f} ({band.sum()} pairs)", np.where(band, says, app), y)
        wrong = np.where(runs[0] != y)[0]
        print("  LLM run 1 disagrees with the reviewers on:")
        for i in wrong:
            print(f"    {df.job_skill[i]} | {df.cv_skill[i]}  cosine {c[i]:.3f}  reviewers {df.reviewer_1[i]}  "
                  f"LLM {'yes' if runs[0][i] else 'no'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["gemini-flash-lite-latest", "gemini-flash-latest"])
