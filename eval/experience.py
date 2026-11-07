"""Compare three experience fits on the labeled CV-job pairs:
  total  years_experience / min years (the current app)
  roles  years in CV roles whose embedding is close to the job (technique A, core/experience.py)
  judge  years in CV roles the LLM judges relevant (technique B, skill_extractor.judge_experience)
Skills, profile similarity and weights stay as in the app, so only the experience fit changes.

usage (from the repo root):
  python eval/experience.py gemini|local gemini:gemini-flash-lite-latest extracted_gemini-flash-lite-latest.json
LLM results are cached in results/ (profiles_roles_<model>.json, judge_<model>.json).
"""
import json
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibrate import components, evaluate, features, load_labels, per_cv
from extract import load_cvs, load_jobs, patient
from skillmatch import OUT, build, embed_cached

from core.config import get_settings
from core.experience import experience_fit, job_text, role_text, role_years
from core.llm import build_provider
from core.schemas import CandidateProfile, JobRequirements, RelevantExperience
from core.skill_extractor import extract_profile, judge_experience

TODAY = date(2026, 10, 31)  # fixed, so "present" roles give the same years on every rerun
ROLE_THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.025), 3)


def cached_llm(path, keys, fn, label):
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k in keys:
        if k in done:
            continue
        start = time.time()
        out = patient(lambda: fn(k), f"{label} {k}", 13.0)
        if out is None:
            continue
        done[k] = out | {"_seconds": round(time.time() - start, 1)}
        path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{label} {k}: {time.time() - start:.0f}s", flush=True)
    return done


def main(kind: str, model: str, extracted: str):
    s = get_settings()
    emb = build(kind)
    llm = build_provider(model, s)
    tag = model.split(":", 1)[1]
    ex = json.loads((OUT / extracted).read_text(encoding="utf-8"))
    labels = load_labels("reviewer_mean")
    cvs, jobs = load_cvs(), {j.id: j for j in load_jobs()}
    for jid, r in ex["jobs"].items():
        jobs[jid].requirements = JobRequirements(**r)

    # 1. CV profiles with roles (new extraction prompt), one call per CV
    profiles = cached_llm(OUT / f"profiles_roles_{tag}.json", sorted(cvs),
                          lambda cv: extract_profile(llm, cvs[cv]).model_dump(), "roles")
    # 2. LLM judge, one batched call per CV over its labeled jobs
    pairs = labels.groupby("cv").job_id.apply(list).to_dict()
    judged = cached_llm(OUT / f"judge_{tag}.json", sorted(pairs),
                        lambda cv: {jid: x.model_dump() for jid, x in judge_experience(
                            llm, cvs[cv], [jobs[j] for j in pairs[cv]], s.max_jd_chars,
                            s.local_max_jd_chars).items()}, "judge")
    missing = [cv for cv in cvs if cv not in profiles or cv not in judged]
    if missing:
        sys.exit(f"LLM results missing for {missing}: rerun after the quota resets")

    # 3. Features: skills and document similarity as in the app; experience three ways
    f = features(emb, kind, s, ex, labels)
    roles = {cv: CandidateProfile(**{k: v for k, v in p.items() if k != "_seconds"}).roles
             for cv, p in profiles.items()}
    texts = sorted({role_text(r) for rs in roles.values() for r in rs} | {job_text(jobs[j]) for j in f.job_id})
    vec = embed_cached(emb, kind, texts)
    f["years_total"] = [profiles[cv]["years_experience"] for cv in f.cv]
    f["sims"] = [np.array([vec[role_text(r)] @ vec[job_text(jobs[j])] for r in roles[cv]])
                 for cv, j in zip(f.cv, f.job_id)]
    f["years_judge"] = [role_years([r for r in (RelevantExperience(**judged[cv].get(j, {})).relevant_roles)], TODAY)
                        if j in judged[cv] else np.nan for cv, j in zip(f.cv, f.job_id)]

    def roles_years(th):
        return np.array([role_years([r for r, x in zip(roles[cv], sm) if x >= th], TODAY)
                         for cv, sm in zip(f.cv, f.sims)])

    def fit(years):  # as score_jobs: relevant years capped at total years; NaN minimum means none stated
        return np.array([experience_fit(min(y, t), None if pd.isna(m) else m)
                         for y, t, m in zip(years, f.years_total, f.min_years)])

    cvl = sorted(f.cv.unique())
    cov, _ = components(f, emb.skill_threshold)
    sem = np.clip((f.doc_cos.values - emb.cos_low) / (emb.cos_high - emb.cos_low), 0, 1)

    def full(exp):  # the app's score with this experience fit
        return s.w_skill_coverage * cov + s.w_semantic * sem + s.w_seniority * exp

    total_fit = np.array([1.0 if pd.isna(m) or not m else min(1.0, y / m) for y, m in zip(f.years_total, f.min_years)])
    # a job the judge left out falls back to total years, as in the app (score_jobs)
    judge_fit = np.where(f.years_judge.isna(), total_fit, fit(np.nan_to_num(f.years_judge.values)))
    print(f"\n{emb.name} | roles+judge by {tag} | {len(f)} pairs, {len(cvl)} CVs | "
          f"judge covered {f.years_judge.notna().sum()} pairs")
    print(f"roles per CV: {np.mean([len(r) for r in roles.values()]):.1f} | "
          f"role-job cosine: p10 {np.quantile(np.concatenate(f.sims.values), .1):.3f} "
          f"median {np.median(np.concatenate(f.sims.values)):.3f} p90 {np.quantile(np.concatenate(f.sims.values), .9):.3f}")

    print("\nroles threshold sweep (experience fit alone | full score), mean per-CV Spearman / NDCG@5:")
    sweep = {}
    for th in ROLE_THRESHOLDS:
        x = fit(roles_years(th))
        sweep[th] = (per_cv(x, f, cvl), per_cv(full(x), f, cvl))
        a, b = sweep[th]
        print(f"  {th:.3f}: alone {np.nanmean(a[0]):6.3f} / {np.nanmean(a[1]):.3f} | "
              f"full {np.nanmean(b[0]):6.3f} / {np.nanmean(b[1]):.3f} | mean relevant years "
              f"{roles_years(th).mean():.1f}")

    # Leave-one-CV-out for the one tuned setting (the roles threshold, chosen on the full score)
    loco, chosen = [], []
    for i, held in enumerate(cvl):
        train = [k for k in range(len(cvl)) if k != i]
        th = max(ROLE_THRESHOLDS, key=lambda t: (np.nanmean(sweep[t][1][0][train]), np.nanmean(sweep[t][1][1][train])))
        chosen.append(th)
        loco.append(sweep[th][1][:, i])
    loco = np.array(loco).T
    roles_th = float(pd.Series(chosen).mode()[0])
    roles_fit = fit(roles_years(roles_th))

    rows = []
    for name, x in [("total years (current)", total_fit), (f"roles @ {roles_th}", roles_fit), ("judge", judge_fit)]:
        alone, comb = evaluate(x, f, cvl), evaluate(full(x), f, cvl)
        rows.append((name, *alone, *comb))
    res = pd.DataFrame(rows, columns=["experience fit", "alone_spearman", "alone_ndcg5", "full_spearman", "full_ndcg5"])
    print("\n" + res.round(3).to_string(index=False))
    # Variant: jobs that state no minimum keep fit 1.0 (as the current app), so relevant years only
    # matter when the job asks for a number of years
    no_min = f.min_years.isna().values | (f.min_years.fillna(0).values == 0)
    print(f"\nvariant, no stated minimum -> fit 1.0 ({no_min.sum()} of {len(f)} pairs have no minimum):")
    for name, x in [(f"roles @ {roles_th}", roles_fit), ("judge", judge_fit)]:
        x = np.where(no_min, 1.0, x)
        alone, comb = evaluate(x, f, cvl), evaluate(full(x), f, cvl)
        print(f"  {name:22s} alone {alone[0]:6.3f} / {alone[1]:.3f} | full {comb[0]:6.3f} / {comb[1]:.3f}")
    print(f"roles, threshold chosen on the other 10 CVs (picked {sorted(set(chosen))}): held-out full score "
          f"Spearman {np.nanmean(loco[0]):.3f}  NDCG@5 {np.nanmean(loco[1]):.3f}")

    yr, yj = roles_years(roles_th), np.nan_to_num(f.years_judge.values)
    print(f"\nrelevant years, mean: total {np.mean(f.years_total):.1f} | roles {yr.mean():.1f} | judge {yj.mean():.1f}")
    print(f"agreement roles vs judge (Spearman over pairs): {spearmanr(yr, yj)[0]:.3f} | "
          f"both zero {np.mean((yr == 0) & (yj == 0)):.0%}, only roles zero {np.mean((yr == 0) & (yj > 0)):.0%}, "
          f"only judge zero {np.mean((yr > 0) & (yj == 0)):.0%}")
    print(f"cost per CV: roles 0 LLM calls, {np.mean([len(r) for r in roles.values()]):.0f} role texts + 1 per job "
          f"embedded | judge 1 LLM call, {np.mean([judged[cv]['_seconds'] for cv in cvl]):.0f}s for 16 jobs")

    out = f[["cv", "job_id", "label", "min_years", "years_total", "years_judge"]].copy()
    out["years_roles"] = yr
    out["fit_total"], out["fit_roles"], out["fit_judge"] = total_fit, roles_fit, judge_fit
    out.to_csv(OUT / f"experience_{kind}.csv", index=False)

    print("\nexamples (highest-rated job per CV):")
    for cv in cvl:
        g = out[out.cv == cv].sort_values("label", ascending=False).iloc[0]
        sims = f.sims[(f.cv == cv) & (f.job_id == g.job_id)].iloc[0]
        jr = [r.title for r, x in zip(roles[cv], sims) if x >= roles_th]
        jj = [r.title for r in RelevantExperience(**judged[cv].get(g.job_id, {})).relevant_roles]
        print(f"  {cv} / {g.job_id} {jobs[g.job_id].title[:45]} (label {g.label}, min {g.min_years}): "
              f"total {g.years_total:.1f}y | roles {g.years_roles:.1f}y {jr} | judge {g.years_judge:.1f}y {jj}")
        print(f"      judge: {RelevantExperience(**judged[cv].get(g.job_id, {})).reasoning[:200]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
