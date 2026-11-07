"""Compare profile-similarity modes on the labeled CV-job pairs (core/similarity.py):
  document  whole CV vs whole posting (the original)
  summary   CV summary vs the job's role_summary (technique B, the app default)
  chunks    mean cosine of each job's 3 closest CV chunks (technique A)
Skills, experience fit (total years) and weights stay as in the app; only similarity changes.
Each mode's band (cosine range stretched to 0-1) is its p5-p95 over the labeled pairs: set from the
spread of cosines only, never from the labels.

usage (from the repo root):
  python eval/similarity.py gemini:gemini-flash-lite-latest extracted_gemini-flash-lite-latest.json gemini local
Needs results/profiles_roles_<model>.json (CV summaries) from eval/experience.py.
"""
import json
import sys

import numpy as np
import pandas as pd

from calibrate import components, evaluate, features, load_labels
from experience import cached_llm
from extract import load_cvs, load_jobs
from skillmatch import OUT, build, embed_cached

from core.config import get_settings
from core.llm import build_provider
from core.schemas import CandidateProfile, JobRequirements
from core.similarity import chunk_text, profile_cosines
from core.skill_extractor import extract_requirements

VARIANTS = [("document", "document", 3), ("summary", "summary", 3),
            ("chunks k=3", "chunks", 3), ("chunks k=1", "chunks", 1), ("chunks k=5", "chunks", 5)]


def main(model: str, extracted: str, kinds: list[str]):
    s = get_settings()
    tag = model.split(":", 1)[1]
    ex = json.loads((OUT / extracted).read_text(encoding="utf-8"))
    labels = load_labels("reviewer_mean")
    cvs, jobs = load_cvs(), {j.id: j for j in load_jobs()}
    prof_path = OUT / f"profiles_roles_{tag}.json"
    if not prof_path.exists():
        sys.exit(f"{prof_path.name} missing: run eval/experience.py first (it extracts the CV summaries)")
    profiles = {cv: CandidateProfile(**{k: v for k, v in p.items() if k != "_seconds"})
                for cv, p in json.loads(prof_path.read_text(encoding="utf-8")).items()}

    # role_summary for the labeled jobs, 10 per call; skills stay from `extracted` so only similarity changes
    labeled = sorted(labels.job_id.unique())
    batches = {f"b{i // 10:02d}": labeled[i:i + 10] for i in range(0, len(labeled), 10)}
    llm = build_provider(model, s)
    summ = cached_llm(OUT / f"role_summaries_{tag}.json", sorted(batches),
                      lambda b: {j.id: j.requirements.role_summary if j.requirements else ""
                                 for j in extract_requirements(llm, [jobs[x].model_copy() for x in batches[b]])},
                      "role summaries")
    role_summary = {jid: sm for b in summ.values() for jid, sm in b.items() if jid != "_seconds"}
    for jid, r in ex["jobs"].items():
        jobs[jid].requirements = JobRequirements(**r, role_summary=role_summary.get(jid, ""))
    print(f"role_summary filled for {sum(bool(v) for v in role_summary.values())} of {len(labeled)} jobs")

    for kind in kinds:
        emb = build(kind)

        def embed(texts):
            vec = embed_cached(emb, kind, texts)
            return np.stack([vec[t] for t in texts])

        f = features(emb, kind, s, ex, labels)
        cvl = sorted(f.cv.unique())
        cov, _ = components(f, emb.skill_threshold)
        sen = np.array([1.0 if pd.isna(m) or not m else min(1.0, y / m) for y, m in zip(f.years, f.min_years)])
        raw, used = {}, {}
        for name, mode, k in VARIANTS:
            vals = {}
            for cv in cvl:
                js = [jobs[j] for j in f.job_id[f.cv == cv]]
                for j, (c, m) in zip(js, profile_cosines(mode, profiles[cv], cvs[cv], js, embed, emb.max_chars, k)):
                    vals[(cv, j.id)] = (c, m)
            raw[name] = np.array([vals[(cv, j)][0] for cv, j in zip(f.cv, f.job_id)])
            used[name] = [vals[(cv, j)][1] for cv, j in zip(f.cv, f.job_id)]

        print(f"\n===== {emb.name} | skills from {ex['model']} | {len(f)} pairs, {len(cvl)} CVs")
        print(f"summary mode fell back to document on {used['summary'].count('document')} pairs | "
              f"CV chunks per CV: {np.mean([len(chunk_text(cvs[c][:4 * emb.max_chars])) for c in cvl]):.1f}")
        rows, bands = [], {}
        for name, _, _ in VARIANTS:
            lo, hi = np.quantile(raw[name], [.05, .95])
            bands[name] = (round(float(lo), 3), round(float(hi), 3))
            sem = np.clip((raw[name] - lo) / (hi - lo), 0, 1)
            full = s.w_skill_coverage * cov + s.w_semantic * sem + s.w_seniority * sen
            rows.append((name, *bands[name], *evaluate(raw[name], f, cvl), *evaluate(full, f, cvl)))
        app_sem = np.clip((raw["document"] - emb.cos_low) / (emb.cos_high - emb.cos_low), 0, 1)
        app_full = evaluate(s.w_skill_coverage * cov + s.w_semantic * app_sem + s.w_seniority * sen, f, cvl)
        res = pd.DataFrame(rows, columns=["similarity", "band_lo", "band_hi", "alone_spearman", "alone_ndcg5",
                                          "full_spearman", "full_ndcg5"])
        print(res.round(3).to_string(index=False))
        print(f"document with the app's current band {emb.cos_low}-{emb.cos_high}: full Spearman "
              f"{app_full[0]:.3f}  NDCG@5 {app_full[1]:.3f}")
        print("agreement between modes (Spearman over pairs): " + ", ".join(
            f"{a} vs {b} {pd.Series(raw[a]).corr(pd.Series(raw[b]), method='spearman'):.2f}"
            for a, b in [("document", "summary"), ("document", "chunks k=3"), ("summary", "chunks k=3")]))

        out = f[["cv", "job_id", "label"]].copy()
        for name in raw:
            out[name] = raw[name]
        out.to_csv(OUT / f"similarity_{kind}.csv", index=False)
        print("top job per CV by each mode (label in brackets):")
        for cv in cvl:
            g = out[out.cv == cv]
            print(f"  {cv}: " + " | ".join(
                f"{n} {g.loc[g[n].idxmax(), 'job_id']} [{g.loc[g[n].idxmax(), 'label']:.1f}]"
                for n in ["document", "summary", "chunks k=3"]) + f" | best label {g.label.max():.1f}")
        print(f"bands for core/embeddings.py ({kind}): summary {bands['summary']}, chunks {bands['chunks k=3']}")

    sa = profiles["SA"].summary
    print(f"\nexample summaries\n  SA CV: {sa}\n  j012: {jobs['j012'].requirements.role_summary}"
          f"\n  j031: {jobs['j031'].requirements.role_summary}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["gemini"])
