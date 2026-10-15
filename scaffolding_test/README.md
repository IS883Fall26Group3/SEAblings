# SEAblings: CV-to-job matching

IS883 Fall 2026, Group 3. Upload a CV, enter a target role and location, and get current job
openings ranked by how well your skills match, with matched skills, gaps, and advice per role.

## How it works

1. **Parse CV** (`core/cv_parser.py`): PDF/DOCX/TXT to text, in memory only.
2. **Extract profile** (`core/skill_extractor.py`): Gemini returns skills, years of experience, level as JSON.
3. **Search jobs** (`core/job_sources/`): JSearch API (aggregates Google for Jobs, incl. Indeed listings),
   or offline sample data.
4. **Extract job requirements**: one batched Gemini call for all jobs.
5. **Score** (`core/matcher.py`): `50% skill coverage + 30% profile similarity + 20% experience fit`,
   using Gemini embeddings. Deterministic, so the same inputs give the same score.
6. **Explain** (on demand): Gemini writes short advice for a selected job.

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
  job_sources/          one class per provider, same interface
prompts/                prompt templates (version these like code)
data/sample/            fictional CV and job listings for dev, tests, demo fallback
tests/                  pytest, no API keys needed
notebooks/              experiments only; the app never imports from here
docs/                   architecture diagram, screenshots, report assets
```

## Team ownership

| Area | Files | Owner |
|---|---|---|
| Job data | `core/job_sources/`, `data/sample/` | TBD |
| LLM and extraction | `core/llm_client.py`, `core/skill_extractor.py`, `core/cv_parser.py`, `prompts/` | TBD |
| Scoring and evaluation | `core/matcher.py`, `tests/`, evaluation notebook | TBD |
| UI and deployment | `streamlit_app.py`, `ui/`, `.streamlit/`, README | TBD |

## Workflow

- `main` is always deployable; Streamlit Cloud redeploys on every push to it.
- Work on `feature/<short-name>` branches; open a pull request with at least one reviewer.
- Track tasks in GitHub Issues and the Projects board; link each PR to its issue.

## Privacy

CVs are processed in memory and never written to disk or logs. Never commit real CVs or API keys.
Note that content sent to the Gemini API free tier may be used by Google to improve its products.
