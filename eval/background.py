"""Step 7: does the judge's field / type-of-work rating make profile similarity agree with the labels?
The judge (skill_extractor.judge_experience, the app's default experience fit) also rates each job's
field and kind of work against the CV: same, related or different. Profile similarity becomes the
banded embedding score times a factor from that rating. Several factor sets are compared on the
labeled pairs; the embedding cosines at the app's length come from doc_band.py.

usage (from the repo root, after doc_band.py):
  python eval/background.py gemini:gemini-flash-lite-latest
One judge call per CV (11), cached in results/background_<model>.json.
"""
import sys

import numpy as np
import pandas as pd

from calibrate import evaluate, load_labels
from experience import cached_llm
from extract import load_cvs, load_jobs
from skillmatch import OUT, build

from core.config import get_settings
from core.llm import build_provider
from core.matcher import background_factor, rescale_cosine
from core.skill_extractor import judge_experience

VARIANTS = {  # name -> (value of "same", "related", "different" per dimension, how two combine)
    "embedding only (before)": None,
    "average, related 0.5": ({"same": 1, "related": 0.5, "different": 0}, "mean"),
    "average, related 0.75": ({"same": 1, "related": 0.75, "different": 0}, "mean"),
    "only 'different' lowers": ({"same": 1, "related": 1, "different": 0}, "mean"),
    "0 only when both different": ({"same": 1, "related": 1, "different": 0}, "max"),
    "any 'different' gives 0": ({"same": 1, "related": 0.75, "different": 0}, "min"),
}


def main(model: str):
    s = get_settings()
    tag = model.split(":", 1)[1]
    emb = build("gemini")
    llm = build_provider(model, s)
    labels = load_labels("reviewer_mean")
    cvs, jobs = load_cvs(), {j.id: j for j in load_jobs()}
    pairs = labels.groupby("cv").job_id.apply(list).to_dict()
    rated = cached_llm(OUT / f"background_{tag}.json", sorted(pairs),
                       lambda cv: {jid: x.model_dump() for jid, x in judge_experience(
                           llm, cvs[cv], [jobs[j] for j in pairs[cv]], s.max_jd_chars,
                           s.local_max_jd_chars).items()}, "judge")

    cos = pd.read_csv(OUT / "doc_band_gemini.csv").set_index(["cv", "job_id"])[f"document_{emb.max_chars}"]
    f = labels[["cv", "job_id", "label"]].reset_index(drop=True)
    f = f[[cv in rated and jid in rated[cv] for cv, jid in zip(f.cv, f.job_id)]].reset_index(drop=True)
    f["field"] = [rated[cv][jid].get("field_match", "") for cv, jid in zip(f.cv, f.job_id)]
    f["work"] = [rated[cv][jid].get("work_match", "") for cv, jid in zip(f.cv, f.job_id)]
    f["embedding"] = [rescale_cosine(cos[(cv, jid)], emb.cos_low, emb.cos_high) for cv, jid in zip(f.cv, f.job_id)]
    cvl = sorted(f.cv.unique())
    print(f"{len(f)} of {len(labels)} labeled pairs rated by {tag}")

    print("\nratings vs the team's labels (1 = wrong field ... 5 = strong fit):")
    print(f.groupby(["field", "work"]).label.agg(pairs="size", mean_label="mean").round(2).to_string())

    rows = []
    for name, spec in VARIANTS.items():
        factor = np.ones(len(f)) if spec is None else np.array(
            [background_factor(a, b, *spec) for a, b in zip(f.field, f.work)])
        sim = f.embedding.values * factor
        sp, nd = evaluate(sim, f, cvl)
        far = (f.label <= 1.5).values
        rows.append((name, sp, nd, sim[far].mean(), sim[~far].mean(), np.mean(sim[far] == 0)))
        f[name] = sim
    out = pd.DataFrame(rows, columns=["variant", "spearman", "ndcg5", "mean_sim_far", "mean_sim_rest",
                                      "far_at_0"])
    print("\nprofile similarity alone vs labels (far = pairs rated 1-1.5, wrong field or far below level):")
    print(out.round(3).to_string(index=False))
    f.to_csv(OUT / f"background_{tag}.csv", index=False)
    print(f"\nper-pair ratings and scores: results/background_{tag}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemini:gemini-flash-lite-latest")
