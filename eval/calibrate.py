"""Step 4: calibrate skill_threshold, cos_low/cos_high and the score weights against labels.

Reproduces core/matcher.py scoring for every labeled CV-job pair (checked against score_jobs),
sweeps a grid of settings, and scores each by how well it ranks jobs *within each CV*:
mean per-CV Spearman with the labels, and NDCG@5 (quality of the top 5 the user sees).
Leave-one-CV-out: tune on 10 CVs, test on the 11th, so the gain over the current settings
is measured on CVs the tuning never saw.

usage (from the repo root):
  python eval/calibrate.py local|gemini extracted_<model>.json [label column]
label column: reviewer_mean (default: mean of reviewer_1 and reviewer_2), reviewer_1,
reviewer_2 or claude_draft. Reads labels/cv_job_reviewed.csv, else labels/cv_job.csv.
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from extract import load_cvs, load_jobs
from skillmatch import HERE, OUT, build, embed_cached

sys.path.insert(0, str(HERE.parent))
from core.config import get_settings  # noqa: E402
from core.matcher import score_jobs  # noqa: E402
from core.schemas import CandidateProfile, JobRequirements  # noqa: E402

THRESHOLDS = np.round(np.arange(0.70, 0.976, 0.025), 3)
WEIGHTS = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=2) if w[0] + w[1] <= 1.0001]


def norm(s: str) -> str:
    return s.strip().lower()


def load_labels(column: str) -> pd.DataFrame:
    path = HERE / "labels" / "cv_job_reviewed.csv"
    path = path if path.exists() else HERE / "labels" / "cv_job.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    if column == "reviewer_mean":
        df["label"] = df[["reviewer_1", "reviewer_2"]].mean(axis=1)
    else:
        df["label"] = df[column]
    print(f"labels: {path.name}, column {column}, {df.label.notna().sum()} pairs")
    return df.dropna(subset=["label"])


def features(emb, kind: str, s, ex: dict, labels: pd.DataFrame) -> pd.DataFrame:
    """Per pair: document cosine, per required skill (exact match?, best CV-skill cosine), years."""
    cvs, jobs = load_cvs(), {j.id: j for j in load_jobs()}
    docs = {f"cv:{c}": t[:emb.max_chars] for c, t in cvs.items()}
    docs |= {f"job:{j}": jobs[j].description[:emb.max_chars] for j in labels.job_id.unique()}
    skills = {norm(x) for p in ex["cvs"].values() for x in p["skills"] if x.strip()}
    skills |= {norm(x) for r in ex["jobs"].values() for x in r["required_skills"] if x.strip()}
    vec = embed_cached(emb, kind, sorted(set(docs.values()) | skills))

    rows = []
    for r in labels.itertuples():
        prof, req = ex["cvs"].get(r.cv), ex["jobs"].get(r.job_id)
        if prof is None:
            continue
        cv_sk = [norm(x) for x in prof["skills"] if x.strip()]
        need = [norm(x) for x in (req["required_skills"] if req else []) if x.strip()]
        M = np.stack([vec[x] for x in cv_sk]) if cv_sk else None
        exact = np.array([x in set(cv_sk) for x in need], dtype=bool)
        best = np.array([float((M @ vec[x]).max()) if M is not None else -1.0 for x in need])
        rows.append(dict(cv=r.cv, job_id=r.job_id, label=r.label, exact=exact, best=best,
                         doc_cos=float(vec[docs[f"cv:{r.cv}"]] @ vec[docs[f"job:{r.job_id}"]]),
                         years=prof["years_experience"], min_years=req["min_years_experience"] if req else None))
    return pd.DataFrame(rows)


def components(f: pd.DataFrame, th: float):
    cov = np.array([((e | (b >= th)).mean() if len(e) else 0.5) for e, b in zip(f.exact, f.best)])
    sen = np.array([1.0 if not m else min(1.0, y / m) for y, m in zip(f.years, f.min_years)])
    return cov, sen


def per_cv(score: np.ndarray, f: pd.DataFrame, cvs) -> np.ndarray:
    """(2, len(cvs)): Spearman and NDCG@5 for each CV's ranking (NaN if its labels are all equal)."""
    out = np.full((2, len(cvs)), np.nan)
    for i, cv in enumerate(cvs):
        m = (f.cv == cv).values
        y, x = f.label.values[m], score[m]
        if np.ptp(y) == 0:
            continue
        out[0, i] = spearmanr(x, y)[0] if np.ptp(x) > 0 else 0.0
        out[1, i] = ndcg_score([y - 1], [x], k=5)  # ties in the score are averaged, not lucky
    return out


def evaluate(score: np.ndarray, f: pd.DataFrame, cvs) -> tuple[float, float]:
    sp, nd = per_cv(score, f, cvs)
    return float(np.nanmean(sp)), float(np.nanmean(nd))


def sweep(f: pd.DataFrame, cvs, cos_grid) -> tuple[pd.DataFrame, np.ndarray]:
    """Every setting scored once per CV, so any train/test split is just a column average."""
    rows, metrics = [], []
    for th in THRESHOLDS:
        cov, sen = components(f, th)
        for lo, hi in cos_grid:
            sem = np.clip((f.doc_cos.values - lo) / (hi - lo), 0, 1)
            for wc, ws in WEIGHTS:
                rows.append((th, lo, hi, wc, ws, round(1 - wc - ws, 1)))
                metrics.append(per_cv(wc * cov + ws * sem + (1 - wc - ws) * sen, f, cvs))
    grid = pd.DataFrame(rows, columns=["skill_threshold", "cos_low", "cos_high", "w_skill", "w_semantic",
                                       "w_seniority"])
    return grid, np.stack(metrics)  # (settings, 2, CVs)


def best(grid: pd.DataFrame, metrics: np.ndarray, cols) -> int:
    sp, nd = np.nanmean(metrics[:, 0, cols], axis=1), np.nanmean(metrics[:, 1, cols], axis=1)
    return int(np.lexsort((-nd, -sp))[0])  # highest Spearman, ties by NDCG@5


def check_against_app(emb, kind, s, ex, f):
    """Our vectorized score must equal core.matcher.score_jobs at the app's own settings.
    score_jobs gets the cached vectors (embed_fn), so the check costs no extra API calls."""
    cv = f.cv.iloc[0]
    jobs = [j for j in load_jobs() if j.id in set(f[f.cv == cv].job_id)]
    for j in jobs:
        r = ex["jobs"].get(j.id)
        j.requirements = JobRequirements(**r) if r else None
    prof = CandidateProfile(**ex["cvs"][cv])
    def embed_fn(texts):
        vec = embed_cached(emb, kind, texts)
        return np.stack([vec[t] for t in texts])
    app = {m.job.id: m.score for m in score_jobs(prof, load_cvs()[cv], jobs, emb, s, embed_fn=embed_fn)}
    g = f[f.cv == cv]
    cov, sen = components(g, emb.skill_threshold)
    sem = np.clip((g.doc_cos.values - emb.cos_low) / (emb.cos_high - emb.cos_low), 0, 1)
    ours = 100 * (s.w_skill_coverage * cov + s.w_semantic * sem + s.w_seniority * sen)
    diff = max(abs(app[j] - o) for j, o in zip(g.job_id, ours))
    assert diff < 0.11, f"eval scoring differs from the app by {diff}"
    print(f"check: matches core.matcher.score_jobs on {len(g)} jobs of {cv} (max diff {diff:.2f})")


def skill_threshold_report(emb, kind: str):
    """labels/skill_pairs.csv: precision/recall of 'cosine >= threshold' against yes/no labels.
    Uses the reviewers' labels where both exist, otherwise Claude's draft (and says so)."""
    path = HERE / "labels" / "skill_pairs.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    reviewed = (df.reviewer_1.str.strip() != "") & (df.reviewer_2.str.strip() != "")
    col = "reviewers (pairs where both agree)" if reviewed.all() else "claude_draft (NOT reviewed yet)"
    if reviewed.all():
        df = df[df.reviewer_1.str.lower() == df.reviewer_2.str.lower()]
        df["label"] = df.reviewer_1
    else:
        df["label"] = df.claude_draft
    y = (df.label.str.lower() == "yes").values
    vec = embed_cached(emb, kind, sorted(set(df.job_skill) | set(df.cv_skill)))
    c = np.array([float(vec[a] @ vec[b]) for a, b in zip(df.job_skill, df.cv_skill)])
    print(f"\nskill_threshold vs labels/skill_pairs.csv ({len(df)} pairs, {y.sum()} yes; labels: {col})")
    print(f"  cosine of 'yes' pairs: median {np.median(c[y]):.3f}  min {c[y].min():.3f} | "
          f"'no' pairs: median {np.median(c[~y]):.3f}  max {c[~y].max():.3f}")
    for th in THRESHOLDS:
        at = c >= th
        tp = (at & y).sum()
        p, r = tp / max(at.sum(), 1), tp / y.sum()
        f1 = 2 * p * r / max(p + r, 1e-9)
        mark = "   <- app" if np.isclose(th, emb.skill_threshold) else ""
        print(f"  {th:.3f}: precision {p:.2f}  recall {r:.2f}  F1 {f1:.2f}  F0.5 {1.25 * p * r / max(.25 * p + r, 1e-9):.2f}{mark}")
    print("  'yes' pairs the app's threshold misses:",
          sorted(zip(np.round(c[y & (c < emb.skill_threshold)], 2), df.job_skill[y & (c < emb.skill_threshold)],
                     df.cv_skill[y & (c < emb.skill_threshold)]), reverse=True)[:12])
    print("  'no' pairs scoring highest:",
          sorted(zip(np.round(c[~y], 2), df.job_skill[~y], df.cv_skill[~y]), reverse=True)[:8])


def main(kind: str, extracted: str, column: str):
    s = get_settings()
    emb = build(kind)
    skill_threshold_report(emb, kind)
    ex = json.loads((OUT / extracted).read_text(encoding="utf-8"))
    f = features(emb, kind, s, ex, load_labels(column))
    check_against_app(emb, kind, s, ex, f)
    cvs = sorted(f.cv.unique())
    print(f"{emb.name} | skills from {ex['model']} | {len(f)} pairs, {len(cvs)} CVs")

    q = np.quantile(f.doc_cos, [0, .05, .25, .5, .75, .95, 1])
    print("document cosine: min %.3f p5 %.3f p25 %.3f median %.3f p75 %.3f p95 %.3f max %.3f" % tuple(q))
    lows = sorted({emb.cos_low, *np.round(np.quantile(f.doc_cos, [0, .05, .25]), 3)})
    highs = sorted({emb.cos_high, *np.round(np.quantile(f.doc_cos, [.75, .95, 1]), 3)})
    cos_grid = [(lo, hi) for lo in lows for hi in highs if hi > lo + 0.02]

    cur = dict(skill_threshold=emb.skill_threshold, cos_low=emb.cos_low, cos_high=emb.cos_high,
               w_skill=s.w_skill_coverage, w_semantic=s.w_semantic, w_seniority=s.w_seniority)
    print("\nEach component alone (mean per-CV Spearman / NDCG@5):")
    cov, sen = components(f, emb.skill_threshold)
    sem = f.doc_cos.values
    for name, x in [("skill coverage @ current threshold", cov), ("semantic (document cosine)", sem),
                    ("seniority fit", sen)]:
        print("  %-36s %6.3f / %.3f" % (name, *evaluate(x, f, cvs)))

    grid, metrics = sweep(f, cvs, cos_grid)
    allc = list(range(len(cvs)))
    grid["spearman"] = np.nanmean(metrics[:, 0, :], axis=1)
    grid["ndcg5"] = np.nanmean(metrics[:, 1, :], axis=1)
    grid.to_csv(OUT / f"calibration_grid_{kind}.csv", index=False)
    sem_c = np.clip((f.doc_cos.values - emb.cos_low) / (emb.cos_high - emb.cos_low), 0, 1)
    cur_cv = per_cv(s.w_skill_coverage * cov + s.w_semantic * sem_c + s.w_seniority * sen, f, cvs)
    print(f"\ncurrent settings {cur}: Spearman {np.nanmean(cur_cv[0]):.3f}  NDCG@5 {np.nanmean(cur_cv[1]):.3f}")
    print(f"\nbest 10 of {len(grid)} settings on all CVs (by Spearman, ties by NDCG@5):")
    print(grid.sort_values(["spearman", "ndcg5"], ascending=False).head(10).round(3).to_string(index=False))
    print("\nhow much each setting matters (best Spearman reachable with the others free):")
    for k in ["skill_threshold", "cos_low", "cos_high", "w_skill", "w_semantic"]:
        print(f"  {k:16s}", grid.groupby(k).spearman.max().round(3).to_dict())

    # Leave-one-CV-out: does tuning beat the current settings on CVs it never saw?
    loco = []
    for i, held in enumerate(cvs):
        g = grid.loc[best(grid, metrics, [c for c in allc if c != i])]
        loco.append((held, g.skill_threshold, g.cos_low, g.cos_high, g.w_skill, g.w_semantic,
                     *metrics[g.name, :, i], *cur_cv[:, i]))
    loco = pd.DataFrame(loco, columns=["held_out", "th", "lo", "hi", "w_skill", "w_sem",
                                       "tuned_spearman", "tuned_ndcg5", "current_spearman", "current_ndcg5"])
    loco.to_csv(OUT / f"calibration_loco_{kind}.csv", index=False)
    print("\nleave-one-CV-out (settings tuned on the other 10 CVs):")
    print(loco.round(3).to_string(index=False))
    m = loco[["tuned_spearman", "current_spearman", "tuned_ndcg5", "current_ndcg5"]].mean()
    print(f"\nheld-out mean: Spearman tuned {m.tuned_spearman:.3f} vs current {m.current_spearman:.3f} | "
          f"NDCG@5 tuned {m.tuned_ndcg5:.3f} vs current {m.current_ndcg5:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "reviewer_mean")
