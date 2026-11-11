"""Test set for the success metric: requirement classification accuracy (met / partial / missing).

Builds, once, before anyone runs the app's labels on these pairs:
  labels/requirements_frozen.json        each test posting's requirements, extracted ONCE with the
                                         app's own prompt and kept fixed, so labelers and the app
                                         label the same list
  labels/requirements/<name>.csv         the answer key: each labeler's rows (main pairs, and the one
                                         pair they label a second time to measure agreement)
  labels/requirements/<name>_extraction.csv  requirements the extraction missed or invented, for the
                                         postings that person checks
  labels/awkward_inputs.csv              4 awkward inputs with the expected behavior written now
  labels/requirement_postings.md         the 12 test postings in full, for reading while labeling
See eval/labeling_guide.md for how to label.

usage (from the repo root):
  python eval/requirement_set.py      # needs GEMINI_API_KEY; never overwrites a filled-in sheet

Why these pairs: of the 176 CV-job pairs rated earlier (labels/cv_job_reviewed.csv), 152 were poor
fits, whose requirements are almost all "missing", which would make accuracy easy. So the 14 regular
pairs are the good (rated 4+) and stretch (3+) fits, at most 3 per CV, spread over 9 CVs. The 2
confusing pairs were rated poor fits but share several required skills with the CV."""
import csv
import json
import sys
from pathlib import Path

import env  # noqa: F401  (GEMINI_API_KEY from .streamlit/secrets.toml)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core.config import get_settings  # noqa: E402
from core.llm import build_provider  # noqa: E402
from core.schemas import Job  # noqa: E402
from core.skill_extractor import extract_requirements  # noqa: E402

LABELS = HERE / "labels"
PEOPLE = LABELS / "requirements"  # one answer-key file per labeler
JOBS = HERE / "data" / "jobs" / "jobs_jsearch.json"
# One fixed model, no fallback: every compared run reads what this model froze, so a failure stops
# with an error instead of quietly switching models (course rule: comparisons use one fixed model).
MODEL = "gemini:gemini-flash-lite-latest"

# (pair id, CV, job id, kind, labeler, second labeler). Earlier fit rating in the comment.
# Nobody labels their own CV. The second labeler labels the same pair separately, to measure how
# often two people agree.
PAIRS = [
    ("P01", "AS", "j054", "regular", "Ardian", "Tigor"),    # 5.0 Financial Analyst, Dover
    ("P02", "AS", "j012", "regular", "Maylada", "Ardian"),  # 4.5 FP&A Analyst, HireVue
    ("P03", "AS", "j040", "regular", "Sheila", "Maylada"),  # 4.5 FP&A Analyst, 7Seventy (short posting)
    ("P04", "LB", "j062", "regular", "Tigor", "Sheila"),    # 4.0 Knowledge, Data and AI Strategy Lead
    ("P05", "LB", "j006", "regular", "Ardian", ""),         # 4.0 Strategy Consultant, Synthires
    ("P06", "SA", "j012", "regular", "Maylada", ""),        # 4.0 FP&A Analyst, HireVue
    ("P07", "SA", "j054", "regular", "Sheila", ""),         # 4.0 Financial Analyst, Dover
    ("P08", "ZW", "j024", "regular", "Tigor", ""),          # 4.0 Machine Learning Engineer, Spotter
    ("P09", "ZW", "j031", "regular", "Ardian", ""),         # 3.5 Sr. BI Data Engineer, Boston Beer
    ("P10", "SU", "j069", "regular", "Maylada", ""),        # 3.5 Product Manager III, ArcBest
    ("P11", "KF", "j072", "regular", "Sheila", ""),         # 3.5 Data Analyst (Life Sciences)
    ("P12", "CYL", "j022", "regular", "Tigor", ""),         # 3.0 Junior Product Analyst, JMD
    ("P13", "MS", "j019", "regular", "Ardian", ""),         # 3.0 Associate Consultant, ZS
    ("P14", "ADW", "j022", "regular", "Maylada", ""),       # 3.0 Junior Product Analyst, JMD
    ("P15", "SU", "j032", "confusing", "Sheila", ""),       # 1.0 Epic BI Developer (healthcare): shares Tableau, Power BI
    ("P16", "SA", "j031", "confusing", "Tigor", ""),        # 1.5 Sr. BI Data Engineer (brewery): shares SQL, Python
]

# Behavior checks: pass / fail against the expected behavior written here, before running.
# They use the fictional sample CV and files we wrote, so nothing personal is involved.
AWKWARD = [
    ("A1", "Empty CV: no file, nothing pasted", "data/sample/jobs.json: s6",
     "The sidebar says the fictional sample CV will be used; the app analyzes the sample CV and "
     "shows results for it, with no error."),
    ("A2", "A document that isn't a CV: eval/data/awkward/not_a_cv.txt (a recipe)", "jobs_jsearch.json: j012",
     "The app tells the user the file doesn't look like a CV and asks for one, instead of scoring it."),
    ("A3", "data/sample/sample_cv.txt", "Not a job posting: eval/data/awkward/not_a_posting.txt (company news)",
     "The app says it found no requirements in that text and doesn't give it a match score as if it "
     "were a job."),
    ("A4", "data/sample/sample_cv.txt", "A very short posting: eval/data/awkward/short_posting.txt",
     "Requirements SQL and Excel are both labeled met, each with a quote from the CV; no minimum "
     "years, so experience fit is full."),
]


def load_jobs() -> dict[str, Job]:
    return {j["id"]: Job(**j) for j in json.loads(JOBS.read_text(encoding="utf-8"))}


def freeze_requirements(jobs: dict[str, Job]) -> dict:
    path = LABELS / "requirements_frozen.json"
    if path.exists():
        print("kept", path.name, "(already frozen)")
        return json.loads(path.read_text(encoding="utf-8"))
    s = get_settings()
    wanted = [jobs[jid].model_copy() for jid in dict.fromkeys(p[2] for p in PAIRS)]
    done, model = extract_requirements(build_provider(MODEL, s), wanted, s.max_jd_chars,
                                       s.local_max_jd_chars), MODEL
    frozen = {"model": model, "jobs": {}}
    for j in done:
        if j.requirements is None:
            raise SystemExit(f"No requirements came back for {j.id}; run again.")
        r = j.requirements
        frozen["jobs"][j.id] = {"title": j.title, "company": j.company, "url": j.url,
                                "required_skills": r.required_skills,
                                "min_years_experience": r.min_years_experience}
    path.write_text(json.dumps(frozen, indent=1, ensure_ascii=False), encoding="utf-8")
    print("froze requirements for", len(frozen["jobs"]), "postings with", model)
    return frozen


def write_new(path: Path, header: list[str], rows: list[list]) -> None:
    if path.exists():
        print("kept", path.name, "(exists: it may already hold labels)")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: opens cleanly in Excel
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path.name, f"({len(rows)} rows)")


def years_text(y: float) -> str:
    return f"{y:g}+ years of relevant experience"


def main() -> None:
    jobs = load_jobs()
    frozen = freeze_requirements(jobs)["jobs"]

    # One file per person, so four people never edit the same file (no merge conflicts).
    people = sorted({p[4] for p in PAIRS} | {p[5] for p in PAIRS if p[5]})
    checker = {}  # each posting's requirement list is checked by the first person who labels it
    for p in PAIRS:
        checker.setdefault(p[2], p[4])
    PEOPLE.mkdir(exist_ok=True)
    for person in people:
        rows = []
        for pid, cv, jid, kind, who, who2 in PAIRS:
            role = "main" if who == person else "second" if who2 == person else ""
            if not role:
                continue
            fz = frozen[jid]
            reqs = [("skill", s) for s in fz["required_skills"]]
            if fz["min_years_experience"]:
                reqs.append(("years", years_text(fz["min_years_experience"])))
            rows += [[pid, kind, cv, jid, fz["title"], k, req, role, "", ""] for k, req in reqs]
        write_new(PEOPLE / f"{person}.csv", ["pair", "kind", "cv", "job_id", "job_title", "type",
                                            "requirement", "role", "label", "note"], rows)
        write_new(PEOPLE / f"{person}_extraction.csv",
                  ["job_id", "job_title", "url", "requirements", "min_years", "missed", "invented"],
                  [[jid, fz["title"], fz["url"], "; ".join(fz["required_skills"]),
                    fz["min_years_experience"] or "", "", ""]
                   for jid, fz in frozen.items() if checker[jid] == person])

    write_new(LABELS / "awkward_inputs.csv", ["id", "cv", "posting", "expected", "actual", "pass"],
              [[i, cv, post, exp, "", ""] for i, cv, post, exp in AWKWARD])

    md = ["# Test postings (requirement accuracy)", "",
          "The 12 postings in the test set, in full, for reading while you label. The requirement list "
          "to label is in `requirement_labels.csv`. Source: public postings collected for this project "
          "(`eval/data/jobs/jobs_jsearch.json`).", ""]
    for jid, fz in frozen.items():
        j = jobs[jid]
        md += [f"## {jid}: {j.title}", f"{j.company} · {j.location} · [original posting]({j.url})", "",
               "**Requirements to label:** " + "; ".join(fz["required_skills"])
               + (f"; {years_text(fz['min_years_experience'])}" if fz["min_years_experience"] else ""), "",
               j.description.strip(), ""]
    (LABELS / "requirement_postings.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote requirement_postings.md")


if __name__ == "__main__":
    main()
