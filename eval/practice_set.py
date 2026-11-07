"""Practice set for tuning the requirement labels, kept apart from the test set.

Tuning prompts or retrieval on the 16 test pairs would make the test number look better than the
app really is. So changes are tried on these 8 pairs instead: the two CVs that are in no test pair
(AYP, VV), each with postings that are in no test pair either. The test set is then run once, after
the changes are done.

Builds, once:
  labels/practice_frozen.json   each practice posting's requirements, extracted once with the app's
                                own prompt (as for the test set)
  labels/practice_labels.csv    the practice answer key, labeled by the rules in labeling_guide.md.
                                Labeled by Claude (an AI assistant) before the app was run on these
                                pairs; used only for tuning, never reported as the success metric.

usage (from the repo root):
  python eval/practice_set.py        # needs GEMINI_API_KEY; never overwrites a filled-in sheet
Then: python eval/requirement_accuracy.py --set practice --app 1"""
from requirement_set import (LABELS, MODELS, Job, build_provider, extract_requirements,  # noqa: F401
                             get_settings, json, load_jobs, with_fallback, write_new, years_text)

# (pair id, CV, job id, kind). Earlier fit rating (labels/cv_job_reviewed.csv) in the comment.
PRACTICE = [
    ("D01", "AYP", "j061", "regular"),  # 3.0 Senior Consultant (Generalist), Oliver Wyman
    ("D02", "AYP", "j048", "regular"),  # 2.0 Strategy Consultant
    ("D03", "AYP", "j035", "regular"),  # 2.0 Lead Product Analyst, Plymouth Rock
    ("D04", "AYP", "j037", "regular"),  # 2.0 Junior ML Engineer / Data Scientist
    ("D05", "VV", "j003", "regular"),   # 2.0 BI Developer II (Power BI), Boston Beer
    ("D06", "VV", "j030", "regular"),   # 2.0 Data Analyst V
    ("D07", "VV", "j057", "regular"),   # 2.0 Senior Analyst, R&D Quality Data Analytics, Takeda
    ("D08", "VV", "j059", "regular"),   # 2.0 BI Developer / Report Writer, MWRA
]
FROZEN = LABELS / "practice_frozen.json"
KEY = LABELS / "practice_labels.csv"


def freeze() -> dict:
    if FROZEN.exists():
        print("kept", FROZEN.name, "(already frozen)")
        return json.loads(FROZEN.read_text(encoding="utf-8"))
    s, jobs = get_settings(), load_jobs()
    wanted = [jobs[jid].model_copy() for jid in dict.fromkeys(p[2] for p in PRACTICE)]
    done, model = with_fallback(MODELS, lambda k: extract_requirements(
        build_provider(k, s), wanted, s.max_jd_chars, s.local_max_jd_chars),
        lambda k, e: print(f"{k} failed: {e}"))
    frozen = {"model": model, "jobs": {}}
    for j in done:
        if j.requirements is None:
            raise SystemExit(f"No requirements came back for {j.id}; run again.")
        frozen["jobs"][j.id] = {"title": j.title, "company": j.company, "url": j.url,
                                "required_skills": j.requirements.required_skills,
                                "min_years_experience": j.requirements.min_years_experience}
    FROZEN.write_text(json.dumps(frozen, indent=1, ensure_ascii=False), encoding="utf-8")
    print("froze requirements for", len(frozen["jobs"]), "postings with", model)
    return frozen


def main() -> None:
    frozen = freeze()["jobs"]
    rows = []
    for pid, cv, jid, kind in PRACTICE:
        fz = frozen[jid]
        reqs = [("skill", s) for s in fz["required_skills"]]
        if fz["min_years_experience"]:
            reqs.append(("years", years_text(fz["min_years_experience"])))
        rows += [[pid, kind, cv, jid, fz["title"], k, req, "main", "", ""] for k, req in reqs]
    write_new(KEY, ["pair", "kind", "cv", "job_id", "job_title", "type", "requirement", "role",
                    "label", "note"], rows)


if __name__ == "__main__":
    main()
