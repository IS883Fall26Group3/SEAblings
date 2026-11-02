"""Extract CV profiles and job requirements with the app's own prompts and providers.
Results are cached in results/extracted_<model>.json and saved after every document,
so a run interrupted by a quota or timeout resumes where it stopped.

usage (from the repo root):
  python eval/extract.py local                      # offline Llama, no API key
  python eval/extract.py gemini:gemini-flash-latest # needs GEMINI_API_KEY
"""
import json
import re
import sys
import time
from pathlib import Path

import env  # noqa: E402,F401  (GEMINI_API_KEY from .streamlit/secrets.toml)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.config import get_settings  # noqa: E402
from core.llm import LLMError, build_provider  # noqa: E402
from core.schemas import Job  # noqa: E402
from core.skill_extractor import extract_profile, extract_requirements  # noqa: E402

DATA = HERE / "data"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def load_jobs() -> list[Job]:
    jobs = []
    for f in sorted((DATA / "jobs").glob("*.json")):
        jobs += [Job(**j) for j in json.loads(f.read_text(encoding="utf-8"))]
    return jobs


def load_cvs() -> dict[str, str]:
    return {f.stem.removesuffix("_Resume"): f.read_text(encoding="utf-8")
            for f in sorted((DATA / "resume").glob("*.txt"))}


def patient(fn, label: str, pace_s: float, tries: int = 6):
    """Free-tier Gemini allows ~5 requests a minute and is sometimes overloaded (503):
    space calls out and wait as long as the error says before retrying."""
    for attempt in range(1, tries + 1):
        try:
            result = fn()
            time.sleep(pace_s)
            return result
        except LLMError as e:
            if "daily quota" in str(e):  # every later call would fail too: stop, rerun after reset
                sys.exit(f"{label}: {e}")
            transient = "overloaded" in str(e) or "rate-limited" in str(e)  # not bad key etc.
            if attempt == tries or not transient:
                print(f"{label}: FAILED ({e})", flush=True)
                return None
            m = re.search(r"Wait (\d+)s", str(e))
            wait = int(m.group(1)) + 3 if m else 30
            print(f"{label}: {str(e)[:60]}... retry {attempt} in {wait}s", flush=True)
            time.sleep(wait)


def main(key: str):
    s = get_settings()
    if key == "local":
        key = f"local:{s.local_llm_repo}"
    llm = build_provider(key, s)
    pace = 13.0 if key.startswith("gemini") else 0.0  # stay under 5 requests/minute
    cache = OUT / f"extracted_{re.sub(r'[^A-Za-z0-9.-]+', '_', key.split(':', 1)[1])}.json"
    done = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {"cvs": {}, "jobs": {}}
    done["model"] = key

    def save():
        cache.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")

    for name, text in load_cvs().items():
        if name in done["cvs"]:
            continue
        start = time.time()
        prof = patient(lambda: extract_profile(llm, text), f"cv {name}", pace)
        if prof is None:
            continue  # a rerun retries only the failures
        done["cvs"][name] = prof.model_dump()
        save()
        print(f"cv {name}: {len(done['cvs'][name]['skills'])} skills, {time.time() - start:.0f}s", flush=True)

    todo = [j for j in load_jobs() if j.id not in done["jobs"]]
    step = 10 if llm.batch_jobs else 1
    for i in range(0, len(todo), step):
        start = time.time()
        part = patient(lambda: extract_requirements(llm, todo[i:i + step], s.max_jd_chars, s.local_max_jd_chars),
                       f"jobs {todo[i].id}..", pace)
        if part is None:
            continue
        for j in part:
            if j.requirements:
                done["jobs"][j.id] = j.requirements.model_dump()
        save()
        print(f"jobs {todo[i].id}..{todo[i + len(part) - 1].id}: {time.time() - start:.0f}s", flush=True)
    print(f"done: {len(done['cvs'])} CVs, {len(done['jobs'])} jobs -> {cache.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
