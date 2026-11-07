"""Old main vs the new branch, one change at a time, on the 176 labeled CV-job pairs.

Every step scores the pairs with the app's own core.matcher.score_jobs, so the numbers are what the
app would show. Steps are cumulative:
  A  old main: main's extraction (3,000 characters, main's prompts), 3,000-character embeddings,
     band 0.731-0.795, exact skill names, total years
  B  + whole postings: this branch's extraction (10,000 characters), 7,000-character embeddings,
     band 0.709-0.786
  C  + skill lookup table
  D  + judge: years in relevant roles
  E  + judge: field and type-of-work factor on profile similarity (= the new branch)
plus C + the factor alone, to separate the judge's two effects.

Needs results/extracted_main.json (main's eval/extract.py, run in a worktree of main) and
results/extracted_gemini-flash-lite-latest.json (this branch). The judge runs once per CV (11 calls)
with the extracted requirements attached, as in the app; cached in results/judge_full_<model>.json.

usage (from the repo root): python eval/pipeline.py [gemini model]
"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibrate import evaluate, load_labels, per_cv
from experience import cached_llm
from extract import load_cvs, load_jobs
from skillmatch import OUT, build, embed_cached

import core.matcher as matcher  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.llm import build_provider  # noqa: E402
from core.schemas import CandidateProfile, JobRequirements, Role  # noqa: E402
from core.skill_extractor import judge_experience  # noqa: E402
from core.skill_names import skill_key  # noqa: E402

OLD_BAND, OLD_CHARS = (0.731, 0.795), 3_000


def exact_name(s: str, aliases=None) -> str:
    return s.strip().lower()  # main compared skills by lowercase name only


def main(model: str):
    s = get_settings()
    emb = build("gemini")
    labels = load_labels("reviewer_mean").reset_index(drop=True)
    cvs, all_jobs = load_cvs(), {j.id: j for j in load_jobs()}
    pairs = labels.groupby("cv").job_id.apply(list).to_dict()
    ex_old = json.loads((OUT / "extracted_main.json").read_text(encoding="utf-8"))
    ex_new = json.loads((OUT / "extracted_gemini-flash-lite-latest.json").read_text(encoding="utf-8"))

    def jobs_for(cv, ex):
        out = []
        for jid in pairs[cv]:
            j = all_jobs[jid].model_copy()
            r = ex["jobs"].get(jid)
            j.requirements = JobRequirements(**r) if r else None
            out.append(j)
        return out

    llm = build_provider(f"gemini:{model}", s)
    judged = cached_llm(OUT / f"judge_full_{model}.json", sorted(pairs),
                        lambda cv: {jid: x.model_dump() for jid, x in judge_experience(
                            llm, cvs[cv], jobs_for(cv, ex_new), s.max_jd_chars,
                            s.local_max_jd_chars).items()}, "judge")

    def embed_fn(texts):
        vec = embed_cached(emb, "gemini", texts)
        return np.stack([vec[t] for t in texts])

    old_emb = SimpleNamespace(name=emb.name, skill_threshold=emb.skill_threshold, cos_low=OLD_BAND[0],
                              cos_high=OLD_BAND[1], max_chars=OLD_CHARS, sim_bands=emb.sim_bands)

    def run(ex, embedder, table: bool, years: bool, factor: bool) -> pd.DataFrame:
        matcher.skill_key = skill_key if table else exact_name
        rows = []
        for cv in sorted(pairs):
            prof = CandidateProfile(**ex["cvs"][cv])
            j = {jid: x for jid, x in judged.get(cv, {}).items() if isinstance(x, dict)}  # not _seconds
            roles = {jid: [Role(**r) for r in x["relevant_roles"]] for jid, x in j.items()} if years else None
            bg = {jid: (x["field_match"], x["work_match"]) for jid, x in j.items()} if factor else None
            for r in matcher.score_jobs(prof, cvs[cv], jobs_for(cv, ex), embedder, s, embed_fn=embed_fn,
                                        relevant_roles=roles, background=bg):
                rows.append(dict(cv=cv, job_id=r.job.id, score=r.score, skills=r.skill_coverage,
                                 similarity=r.semantic_similarity, experience=r.seniority_fit))
        matcher.skill_key = skill_key
        return labels[["cv", "job_id", "label"]].merge(pd.DataFrame(rows), on=["cv", "job_id"])

    steps = {
        "A  old main": run(ex_old, old_emb, False, False, False),
        "B  + whole postings, new band": run(ex_new, emb, False, False, False),
        "C  + skill lookup table": run(ex_new, emb, True, False, False),
        "D  + judge: relevant years": run(ex_new, emb, True, True, False),
        "E  + judge: field/work factor (new)": run(ex_new, emb, True, True, True),
        "C + factor only (no judge years)": run(ex_new, emb, True, False, True),
    }
    cvl = sorted(pairs)
    print(f"{len(labels)} labeled pairs, {len(cvl)} CVs; mean per-CV Spearman / NDCG@5 (higher is better)\n")
    print(f"{'step':38} {'total':>13} {'skills':>13} {'similarity':>13} {'experience':>13}"
          f" {'poor fits':>9} {'good fits':>9}")
    for name, f in steps.items():
        cells = []
        for col in ("score", "skills", "similarity", "experience"):
            sp, nd = evaluate(f[col].values, f, cvl)
            cells.append(f"{sp:.3f} / {nd:.3f}")
        poor = f.score[f.label <= 1.5].mean()
        good = f.score[f.label >= 3].mean()
        print(f"{name:38} " + " ".join(f"{c:>13}" for c in cells) + f" {poor:9.1f} {good:9.1f}")

    a, e = steps["A  old main"], steps["E  + judge: field/work factor (new)"]
    sa, se = per_cv(a.score.values, a, cvl)[0], per_cv(e.score.values, e, cvl)[0]
    print(f"\nper CV, old main -> new (Spearman): " + ", ".join(f"{c} {x:.2f}->{y:.2f}" for c, x, y in zip(cvl, sa, se)))
    print(f"new better on {(se > sa).sum()} of {len(cvl)} CVs")
    print(f"pairs: poor fits = labeled 1-1.5 ({(a.label <= 1.5).sum()}), good fits = labeled 3+ ({(a.label >= 3).sum()})")
    print(f"labels vs score across all 176 pairs: old {spearmanr(a.score, a.label)[0]:.3f}, "
          f"new {spearmanr(e.score, e.label)[0]:.3f}")
    print(f"mean score: old {a.score.mean():.1f}, new {e.score.mean():.1f}")
    rated = [x for cv in cvl for x in judged[cv].values() if isinstance(x, dict)]
    print(f"judge: {len(rated)} pairs rated; with at least one relevant role: {sum(bool(x['relevant_roles']) for x in rated)}")
    pd.concat({k: v for k, v in steps.items()}, names=["step"]).to_csv(OUT / "pipeline_gemini.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemini-flash-lite-latest")
