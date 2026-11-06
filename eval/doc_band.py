"""Step 6 revisited: the "document" similarity band at the old (3,000) and the app's embedding length.
The band is the p5-p95 of CV-vs-posting cosines over the labeled pairs (spread only, never the
labels), as in similarity.py. Embeddings only, no LLM calls: needs data/resume and an API key.
The 3,000-char run is checked against results/similarity_gemini.csv (the original step 6).

usage (from the repo root):
  python eval/doc_band.py gemini
"""
import sys

import numpy as np
import pandas as pd

from calibrate import evaluate, load_labels
from extract import load_cvs, load_jobs
from skillmatch import OUT, build, embed_cached

OLD_CHARS = 3_000


def main(kind: str):
    emb = build(kind)
    labels = load_labels("reviewer_mean")
    cvs, jobs = load_cvs(), {j.id: j for j in load_jobs()}
    f = labels[["cv", "job_id", "label"]].reset_index(drop=True)
    cvl = sorted(f.cv.unique())
    old_path = OUT / f"similarity_{kind}.csv"
    old = pd.read_csv(old_path).set_index(["cv", "job_id"])["document"] if old_path.exists() else None

    out = f.copy()
    for n in sorted({OLD_CHARS, emb.max_chars}):
        docs = {c: t[:n] for c, t in cvs.items()} | {j: jobs[j].description[:n] for j in f.job_id.unique()}
        vec = embed_cached(emb, kind, sorted(set(docs.values())))
        cos = np.array([float(vec[docs[c]] @ vec[docs[j]]) for c, j in zip(f.cv, f.job_id)])
        out[f"document_{n}"] = cos
        lo, hi = np.percentile(cos, [5, 95])
        sp, nd = evaluate(cos, f, cvl)
        cut_cv = sum(len(t) > n for t in cvs.values())
        cut_job = sum(len(jobs[j].description) > n for j in f.job_id.unique())
        print(f"\n== {emb.name}, {n:,} chars: {cut_cv}/{len(cvs)} CVs and {cut_job}/{f.job_id.nunique()} "
              f"postings cut")
        print(f"   band (p5-p95): {lo:.3f}-{hi:.3f} | min {cos.min():.3f}, median {np.median(cos):.3f}, "
              f"max {cos.max():.3f}")
        print(f"   similarity alone vs labels: Spearman {sp:.3f}, NDCG@5 {nd:.3f}")
        app = np.clip((cos - emb.cos_low) / (emb.cos_high - emb.cos_low), 0, 1)
        print(f"   with the app's band {emb.cos_low}-{emb.cos_high}: {np.mean(app == 1):.0%} of pairs at 100%, "
              f"{np.mean(app == 0):.0%} at 0%")
        if n == OLD_CHARS and old is not None:
            o = np.array([old[(c, j)] for c, j in zip(f.cv, f.job_id)])
            print(f"   vs {old_path.name}: max difference {np.abs(o - cos).max():.4f}, "
                  f"correlation {np.corrcoef(o, cos)[0, 1]:.4f}")
    out.to_csv(OUT / f"doc_band_{kind}.csv", index=False)
    print(f"\nper-pair cosines: results/doc_band_{kind}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemini")
