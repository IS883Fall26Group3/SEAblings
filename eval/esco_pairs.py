"""Step 2: a first skill_threshold from ESCO, with no hand labels.
Positives: a skill's preferred label vs one of its alternative labels (same concept).
Hard negatives: the most similar preferred label of a *different* concept (the near misses
the threshold must reject). Easy negatives: random pairs of different concepts.

usage (from the repo root): python eval/esco_pairs.py local|gemini
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from skillmatch import OUT, HERE, build, embed_cached

N_CONCEPTS = 600
MAX_WORDS = 4  # app skills are 1-4 words; long ESCO phrases behave differently
rng = np.random.default_rng(0)


def short(label: str) -> bool:
    return 0 < len(label.split()) <= MAX_WORDS


def load() -> pd.DataFrame:
    df = pd.read_csv(HERE / "data" / "esco" / "skills_en.csv", usecols=["conceptUri", "preferredLabel", "altLabels"])
    df["alts"] = df.altLabels.fillna("").map(lambda s: [a.strip() for a in s.split("\n") if short(a.strip())])
    return df[df.preferredLabel.map(short)].reset_index(drop=True)


def main(kind: str):
    df = load()
    emb = build(kind)
    pool = df.preferredLabel.tolist()
    sample = df[df.alts.str.len() > 0].sample(N_CONCEPTS, random_state=0)
    alts = [rng.choice(a) for a in sample.alts]
    print(f"{emb.name}: {len(pool)} short ESCO skills, {N_CONCEPTS} sampled concepts")
    vec = embed_cached(emb, kind, sorted(set(pool) | set(alts)), chunk=1000)

    P = np.stack([vec[p] for p in pool])
    rows = []
    for (idx, r), alt in zip(sample.iterrows(), alts):
        a = vec[r.preferredLabel]
        rows.append((r.preferredLabel, alt, "synonym", float(a @ vec[alt])))
        sims = P @ a
        sims[idx] = -1  # exclude the concept itself
        j = int(np.argmax(sims))
        rows.append((r.preferredLabel, pool[j], "hard_negative", float(sims[j])))
        k = int(rng.integers(len(pool)))
        if k != idx:
            rows.append((r.preferredLabel, pool[k], "random_negative", float(sims[k])))
    pairs = pd.DataFrame(rows, columns=["skill_a", "skill_b", "kind", "cosine"])
    pairs.to_csv(OUT / f"esco_pairs_{kind}.csv", index=False)

    print(f"\n=== {emb.name} ===")
    for k, g in pairs.groupby("kind"):
        print(f"{k:16s} n={len(g):4d}  cosine p10 {g.cosine.quantile(.1):.3f}  "
              f"median {g.cosine.median():.3f}  p90 {g.cosine.quantile(.9):.3f}")
    hard = pairs[pairs.kind != "random_negative"]
    y, c = (hard.kind == "synonym").values, hard.cosine.values
    p, r, t = precision_recall_curve(y, c)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    best = int(np.argmax(f1[:-1]))
    print(f"\nsynonyms vs hard negatives: AUC-PR {average_precision_score(y, c):.3f}")
    print(f"best-F1 cutoff {t[best]:.3f}: F1 {f1[best]:.3f}  P {p[best]:.3f}  R {r[best]:.3f}")
    for target in (0.8, 0.9, 0.95):
        ok = np.where(p[:-1] >= target)[0]
        if len(ok):
            i = ok[0]
            print(f"lowest cutoff with precision >= {target}: {t[i]:.3f} (recall {r[i]:.3f})")
    for th in (0.75, 0.8, 0.85, 0.9):
        at = c >= th
        print(f"cutoff {th}: precision {(at & y).sum() / max(at.sum(), 1):.3f}  recall {(at & y).sum() / y.sum():.3f}"
              + ("   <- app skill_threshold" if th == emb.skill_threshold else ""))
    print("\nhighest-cosine hard negatives (different ESCO concepts):")
    print(pairs[pairs.kind == "hard_negative"].nlargest(10, "cosine")[["skill_a", "skill_b", "cosine"]].to_string(index=False))
    print("\nlowest-cosine synonyms:")
    print(pairs[pairs.kind == "synonym"].nsmallest(10, "cosine")[["skill_a", "skill_b", "cosine"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
