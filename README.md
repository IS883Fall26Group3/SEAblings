# SEAblings: CV-to-job matching

IS883 Fall 2026, Group 3. Upload or paste a CV, then match it against job descriptions you already
have or search current openings by role (location optional). Jobs are ranked by how well your skills
match, with matched skills, gaps, and advice per role.

## How it works

1. **Read the inputs** (`core/cv_parser.py`): CV and any job descriptions as PDF, DOCX or TXT, or pasted, in
   memory only. With no CV, the app uses a fictional sample so you can try it.
2. **Check the inputs** (`core/validation.py`): simple rules, no LLM call, so a wrong input costs no quota.
   A CV scores points for section headings, date ranges, contact details and a degree; a job description for
   job-posting phrases; each must score enough and read more like its own type than the other. A role is
   rejected with fewer than 2 different letters or over 80 characters. Each input shows its result on its own line.
3. **Extract profile** (`core/skill_extractor.py`): Gemini returns skills, years of experience, level as JSON.
4. **Find jobs** (`core/job_sources/`): the job descriptions you gave, or a search by role: JSearch API
   (aggregates Google for Jobs, incl. Indeed listings) or offline sample data. With no location, it
   searches across the US.
5. **Extract job requirements**: one batched call for all jobs (one call per job on the offline model).
6. **Score** (`core/matcher.py`): `50% skill coverage + 30% profile similarity + 20% experience fit`,
   using embeddings (`core/embeddings.py`: Gemini, or local without a key). Deterministic, so the same inputs give the same score.
   Each card explains its three scores (`core/score_notes.py`): matched and close-matched skills, the
   field / type of work / level of the CV beside the job's, and the roles counted as experience. Built
   from what the matcher recorded, with no extra calls, so the text always agrees with the numbers.
7. **Advice** (on demand, `core/advice.py`): bullets for a selected job: strongest match, main gap, up to
   three "May not be a real gap" items, and the most useful next step. The LLM lists the posting's
   requirements (structured extraction), each one retrieves its 3 closest CV chunks (RAG), and the LLM rates
   it strong, weak or not shown, with the full CV to check before calling something a gap. Python then checks
   each quote against the CV, keeps "strong" for real work (not a summary or skills list alone), and adds a
   "May not be a real gap" bullet for a "To build" chip the CV contradicts. Job duties come before basic
   conditions such as a license or schedule, and years of experience are left to Experience fit. A next step
   must name the gap; a job rated different in both field and work gets a step toward closer roles instead.
   Two LLM calls per click; scores and chips don't change. CV chunks are up to 40 words of whole lines (wrapped
   lines rejoined), never mixing roles or sections; a role that has to split repeats its last line (about
   10 words of overlap).

## Using the app

The sidebar has three sections: **1. Your CV**, **2. Jobs to match** and **3. AI model**. During a run, numbered
steps show what's finished, and your profile and the jobs appear as soon as they're ready. If a later step fails,
that work stays on the page with **Try again** and **New search** buttons. **New search** clears the results and
job inputs but keeps your CV.

## Choosing a model

The sidebar's **AI model** menu offers each Gemini model listed in `GEMINI_MODELS`, an offline
Hugging Face model (when installed), and **Auto**. Gemini's free tier limits requests per day
*per model*, so Auto tries the models in order and moves on when one hits its quota or is
overloaded. The offline model is the last resort: no quota, but slower and less accurate.

**Offline model.** Llama 3.2 1B Instruct (4-bit GGUF, about 0.8 GB) runs inside the app with
`llama-cpp-python`, the same setup as the course's `llama_demo`. It downloads from Hugging Face the
first time it's used and again after each app restart. Output is constrained to the expected JSON
schema, which keeps a small model's extraction well-formed. It reads each job separately, so fewer
jobs means faster results.

Embeddings use Gemini (`gemini-embedding-001`) whenever `GEMINI_API_KEY` is set, and a small local
model (bge-small, GGUF, via `llama-cpp-python`) otherwise; set `EMBEDDINGS = "local"` to force the
local one. Gemini's free tier embeds about 100 texts and 30,000 tokens per minute, and one search
embeds the CV, each job, and every unique skill; at the limit the app waits (up to a minute, with a
message) and tries once more. For profile similarity the CV and each posting are embedded up to
7,000 characters with Gemini (3,000 with the local model); the LLM reads up to 10,000 characters of
each posting. Similarity thresholds and bands differ per embedding model and per length
(`core/embeddings.py`); see `eval/README.md` (step 6, revisited) for how they were checked.

**Experience fit** (`EXPERIENCE_FIT`, `core/experience.py`) can count *relevant* years instead of
all years: `"roles"` keeps CV roles whose embedding is close to the job; `"judge"` asks the LLM
which roles count (one extra call per search, guided steps, CV only). Either way the years are
added up in Python from the roles' dates. Default `"judge"` (best in `eval/README.md` step 5); if
the judge call fails, that search falls back to all years. `"total"` counts every year on the CV.

**Profile similarity** (`PROFILE_SIMILARITY`, `core/similarity.py`): `"document"` (default)
compares the whole CV with the whole posting; `"summary"` compares the CV summary with a 2-3
sentence `role_summary` written during requirement extraction (no extra call); `"chunks"` averages
each job's 3 closest CV chunks. Each mode has its own cosine band in `core/embeddings.py`; document
ranked best in `eval/README.md` (step 6).

## Run locally

```bash
pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your keys
streamlit run streamlit_app.py
pytest
```

Keys: Gemini from Google AI Studio; JSearch from RapidAPI (free BASIC plan is hard-capped monthly).
Set `JOB_SOURCE = "sample"` during development to save quota. Each teammate should use their own keys.

## Deploy (Streamlit Community Cloud)

Main file: `streamlit_app.py`. Paste the contents of your `secrets.toml` into App settings > Secrets.
No model weights are stored or downloaded, which keeps the app within free-tier memory limits.

## Project structure

```
streamlit_app.py        UI wiring only
ui/                     components and CSS
core/                   business logic, no Streamlit imports (testable)
  schemas.py            shared data contracts: change only by team agreement
  llm/                  model providers (Gemini, offline Hugging Face) and Auto fallback
  embeddings.py         Gemini or local embeddings, each with its own thresholds
  job_sources/          one class per provider, same interface
  validation.py         rule-based checks on the CV, job descriptions and role, before any LLM call
  advice.py             on-demand advice: CV chunking, retrieval, and checks on the LLM's ratings
prompts/                prompt templates (version these like code)
data/sample/            fictional CV and job listings for dev, tests, demo fallback
data/skill_aliases.csv  skill name lookup table (built by scripts/build_skill_aliases.py)
data/onet/              O*NET software names the lookup table is built from (source below)
scripts/                one-off build scripts
tests/                  pytest, no API keys needed (tests/data/ holds a dummy resume)
eval/                   embedding threshold calibration (see eval/README.md)
notebooks/              experiments only; the app never imports from here
docs/                   architecture diagram, screenshots, report assets
```

**Skill data source.** `data/onet/software_skills.csv` is from the O*NET 31.0 Database by the
U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license; O*NET® is a trademark of USDOL/ETA
([onetcenter.org](https://www.onetcenter.org/database.html)). `data/skill_aliases.csv` is derived from it
and a short hand-made list.

## Team ownership

| Area | Files | Owner |
|---|---|---|
| Job data | `core/job_sources/`, `data/sample/` | TBD |
| LLM and extraction | `core/llm/`, `core/embeddings.py`, `core/skill_extractor.py`, `core/advice.py`, `core/cv_parser.py`, `core/validation.py`, `prompts/` | TBD |
| Scoring and evaluation | `core/matcher.py`, `tests/`, evaluation notebook | TBD |
| UI and deployment | `streamlit_app.py`, `ui/`, `.streamlit/`, README | TBD |

## Workflow

- `main` is always deployable; Streamlit Cloud redeploys on every push to it.
- Work on `feature/<short-name>` branches; open a pull request with at least one reviewer.
- Track tasks in GitHub Issues and the Projects board; link each PR to its issue.

## Privacy

CVs are processed in memory and never written to disk or logs. The CV text stays in the session's
memory until the next run or "New search", for the advice button. Never commit real CVs or API keys.
Note that content sent to the Gemini API free tier may be used by Google to improve its products.

## [Oct 23, 2026] Comments from Professor
- Basic Flow: Input CV+JD --> Calculate score --> Give text advise [reasonable]
- Current Problems: score is inaccurate (do not have industry context)
- To do:
  1) Need to be more clear/be careful of using embedding
  2) Explore co-sine similarity
  3) Comparison about reasoning method e.g., one-shot, chain of thoughts
  4) Add error handling and unhappy cases (need to show some explanation for users what to do next)
  5) Need to define specific how to calculate and steps for success metric
