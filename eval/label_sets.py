"""Build (or refresh) the two labeling sheets in labels/.

cv_job.csv      CV-job pairs to rate 1-5. Per CV: the 6 most similar jobs by document cosine
                (where the app's top 5 comes from) plus 10 spread over the remaining ranks.
                Chosen from documents only, so the sheet stays fixed if extraction is rerun.
skill_pairs.csv job skill vs the closest CV skill, the exact decision match_skills() makes,
                sampled across cosine bins with most pairs near the threshold.

Rerunning keeps every label already entered (claude_draft, claude_note, reviewer_1, reviewer_2, notes).
Cosines are not shown in the sheets so they don't anchor the labelers.

usage (from the repo root): python eval/label_sets.py [extracted_<model>.json ...]
"""
import json
import sys

import numpy as np
import pandas as pd

from extract import load_cvs, load_jobs
from skillmatch import HERE, OUT, build, embed_cached

sys.path.insert(0, str(HERE.parent))
from core.config import get_settings  # noqa: E402

LABELS = HERE / "labels"
LABEL_COLS = ["claude_draft", "claude_note", "reviewer_1", "reviewer_2", "notes"]
TOP, SPREAD = 6, 10
BINS = [(0.0, 0.70, 12), (0.70, 0.75, 14), (0.75, 0.80, 20), (0.80, 0.85, 26),
        (0.85, 0.90, 26), (0.90, 0.95, 20), (0.95, 1.01, 12)]  # (low, high, how many)


def norm(s: str) -> str:
    return s.strip().lower()


def merge_labels(new: pd.DataFrame, path, keys: list[str]) -> pd.DataFrame:
    for c in LABEL_COLS:
        new[c] = ""
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = new.drop(columns=LABEL_COLS).merge(old[keys + LABEL_COLS], on=keys, how="left").fillna("")
    return new


def doc_cosines(emb, s) -> pd.DataFrame:
    cvs, jobs = load_cvs(), load_jobs()
    texts = [t[:s.max_jd_chars] for t in cvs.values()] + [j.description[:s.max_jd_chars] for j in jobs]
    vec = embed_cached(emb, "local", texts)
    rows = [(cv, j.id, j.title, j.company, j.url, float(vec[ct[:s.max_jd_chars]] @ vec[j.description[:s.max_jd_chars]]))
            for cv, ct in cvs.items() for j in jobs]
    return pd.DataFrame(rows, columns=["cv", "job_id", "job_title", "company", "url", "cosine"])


def cv_job_sheet(docs: pd.DataFrame) -> pd.DataFrame:
    picks = []
    for cv, g in docs.groupby("cv"):
        g = g.sort_values("cosine", ascending=False).reset_index(drop=True)
        rest = np.linspace(TOP, len(g) - 1, SPREAD).round().astype(int)
        picks.append(g.loc[list(range(TOP)) + list(rest)])
    sheet = pd.concat(picks)[["cv", "job_id", "job_title", "company", "url"]]
    return sheet.sample(frac=1, random_state=0)  # shuffled, so rank order doesn't leak


def skill_sheet(emb, extracted: list[dict]) -> pd.DataFrame:
    titles = {j.id: j.title for j in load_jobs()}
    cands = {}
    for ex in extracted:
        cv_skills = {cv: sorted({norm(x) for x in p["skills"] if x.strip()}) for cv, p in ex["cvs"].items()}
        job_skills = {jid: sorted({norm(x) for x in r["required_skills"] if x.strip()}) for jid, r in ex["jobs"].items()}
        vec = embed_cached(emb, "local", sorted({x for v in [*cv_skills.values(), *job_skills.values()] for x in v}))
        for cv, cs in cv_skills.items():
            if not cs:
                continue
            M = np.stack([vec[x] for x in cs])
            for jid, js in job_skills.items():
                for x in js:
                    if x in cs:
                        continue  # exact matches never reach the threshold
                    sims = M @ vec[x]
                    i = int(np.argmax(sims))
                    cands.setdefault((x, cs[i]), (float(sims[i]), titles.get(jid, jid)))
    df = pd.DataFrame([(a, b, c, t) for (a, b), (c, t) in cands.items()],
                      columns=["job_skill", "cv_skill", "cosine", "example_job"])
    print(f"{len(df)} distinct (job skill, closest CV skill) pairs")
    parts = []
    for lo, hi, n in BINS:
        g = df[(df.cosine >= lo) & (df.cosine < hi)]
        parts.append(g.sample(min(n, len(g)), random_state=0))
        print(f"  cosine {lo:.2f}-{hi:.2f}: {len(g):5d} available, {min(n, len(g))} sampled")
    return pd.concat(parts).drop(columns="cosine").sample(frac=1, random_state=0)


def main(files: list[str]):
    s = get_settings()
    emb = build("local")
    LABELS.mkdir(exist_ok=True)

    docs = doc_cosines(emb, s)
    docs.to_csv(OUT / "doc_cosines_local.csv", index=False)
    path = LABELS / "cv_job.csv"
    sheet = merge_labels(cv_job_sheet(docs), path, ["cv", "job_id"])
    sheet.to_csv(path, index=False)
    print(f"{path.name}: {len(sheet)} pairs ({sheet.cv.nunique()} CVs)")

    extracted = [json.loads((OUT / f).read_text(encoding="utf-8")) for f in files]
    if extracted:
        path = LABELS / "skill_pairs.csv"
        sheet = merge_labels(skill_sheet(emb, extracted), path, ["job_skill", "cv_skill"])
        sheet.to_csv(path, index=False)
        print(f"{path.name}: {len(sheet)} pairs")


if __name__ == "__main__":
    main(sys.argv[1:])
