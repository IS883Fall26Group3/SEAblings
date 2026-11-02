"""Step 1: how well does each app embedder separate related from unrelated skills on SkillMatch-1K?
Uses the app's own embedder classes (core/embeddings.py). Vectors are cached per embedder,
so a Gemini run that hits a quota can be resumed.

usage (from the repo root): python eval/skillmatch.py local|gemini
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import env  # noqa: E402,F401  (GEMINI_API_KEY from .streamlit/secrets.toml)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.config import get_settings  # noqa: E402
from core.embeddings import GeminiEmbedder, LocalEmbedder  # noqa: E402
from core.llm.base import LLMError  # noqa: E402

DATA = HERE / "data" / "skillmatch" / "test.csv"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def build(kind: str):
    s = get_settings()
    if kind == "local":
        return LocalEmbedder(s.local_embed_repo, s.local_embed_file, s.local_threads)
    return GeminiEmbedder(s.gemini_api_key, s.gemini_embed_model, s.llm_timeout_s)


def embed_cached(emb, kind: str, texts: list[str], chunk: int = 100) -> dict[str, np.ndarray]:
    """One vector cache per embedder, shared by all eval scripts (keys are the raw strings)."""
    cache = OUT / f"vectors_{kind}.npz"
    vec = dict(np.load(cache, allow_pickle=True).items()) if cache.exists() else {}
    todo = [t for t in texts if t not in vec]
    gemini = isinstance(emb, GeminiEmbedder)
    if gemini:  # free tier: 100 texts/minute, every text counts; stay under it instead of hitting it
        chunk = min(chunk, 50)
    for i in range(0, len(todo), chunk):
        if gemini and i:
            time.sleep(31)
        part = todo[i:i + chunk]
        for attempt in range(6):  # free-tier Gemini: 100 texts/minute, sometimes overloaded
            try:
                vectors = emb.embed(part)
                break
            except LLMError as e:
                if attempt == 5 or not ("rate-limited" in str(e) or "overloaded" in str(e)):
                    raise
                m = re.search(r"Wait (\d+)s", str(e))
                print(f"  {str(e)[:50]}... retry in {int(m.group(1)) + 3 if m else 60}s", flush=True)
                time.sleep(int(m.group(1)) + 3 if m else 60)
        for t, v in zip(part, vectors):
            vec[t] = v
        np.savez(cache, **vec)
        print(f"  embedded {len(vec)}/{len(texts)}", flush=True)
    return vec


def main(kind: str):
    df = pd.read_csv(DATA)
    df["related"] = df["related"].astype(str).str.lower() == "true"
    skills = sorted(set(df.skill_a) | set(df.skill_b))
    emb = build(kind)
    print(f"{emb.name}: {len(df)} pairs, {len(skills)} unique skills")
    vec = embed_cached(emb, kind, skills)
    df["cosine"] = [float(vec[a] @ vec[b]) for a, b in zip(df.skill_a, df.skill_b)]
    df.to_csv(OUT / f"pairs_{kind}.csv", index=False)

    y, c = df.related.values, df.cosine.values
    p, r, t = precision_recall_curve(y, c)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    best = int(np.argmax(f1[:-1]))
    at = c >= emb.skill_threshold
    tp = int((at & y).sum())
    pos, neg = c[y], c[~y]
    print(f"\n=== {emb.name} ===")
    print(f"AUC-PR {average_precision_score(y, c):.3f}   ROC-AUC {roc_auc_score(y, c):.3f}")
    print(f"cosine related:   mean {pos.mean():.3f}  p10 {np.percentile(pos, 10):.3f}  "
          f"median {np.median(pos):.3f}  p90 {np.percentile(pos, 90):.3f}")
    print(f"cosine unrelated: mean {neg.mean():.3f}  p10 {np.percentile(neg, 10):.3f}  "
          f"median {np.median(neg):.3f}  p90 {np.percentile(neg, 90):.3f}")
    print(f"best-F1 cutoff {t[best]:.3f}: F1 {f1[best]:.3f}  P {p[best]:.3f}  R {r[best]:.3f}")
    print(f"app skill_threshold {emb.skill_threshold}: flags {int(at.sum())} pairs, "
          f"P {tp / max(at.sum(), 1):.3f}  R {tp / y.sum():.3f}")
    print("\nhighest-cosine unrelated pairs:")
    print(df[~df.related].nlargest(8, "cosine")[["skill_a", "skill_b", "cosine"]].to_string(index=False))
    print("\nlowest-cosine related pairs:")
    print(df[df.related].nsmallest(8, "cosine")[["skill_a", "skill_b", "cosine"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
