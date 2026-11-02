# SEAblings: CV-to-job matching

IS883 Fall 2026, Group 3. Upload a CV, enter a target role and location, and get current job
openings ranked by how well your skills match, with matched skills, gaps, and advice per role.

## How it works

1. **Parse CV** (`core/cv_parser.py`): PDF/DOCX/TXT to text, in memory only.
2. **Extract profile** (`core/skill_extractor.py`): Gemini returns skills, years of experience, level as JSON.
3. **Search jobs** (`core/job_sources/`): JSearch API (aggregates Google for Jobs, incl. Indeed listings),
   or offline sample data.
4. **Extract job requirements**: one batched call for all jobs (one call per job on the offline model).
5. **Score** (`core/matcher.py`): `50% skill coverage + 30% profile similarity + 20% experience fit`,
   using embeddings (`core/embeddings.py`: local if installed, otherwise Gemini). Deterministic, so the same inputs give the same score.
6. **Explain** (on demand): Gemini writes short advice for a selected job.

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

Embeddings use a small local model (bge-small, GGUF) whenever `llama-cpp-python` is installed, so
scoring never uses Gemini quota; if it can't load, the app falls back to Gemini embeddings.
Similarity thresholds differ per embedding model (`core/embeddings.py`), so re-check them with the
evaluation set.

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
| LLM and extraction | `core/llm/`, `core/embeddings.py`, `core/skill_extractor.py`, `core/cv_parser.py`, `prompts/` | TBD |
| Scoring and evaluation | `core/matcher.py`, `tests/`, evaluation notebook | TBD |
| UI and deployment | `streamlit_app.py`, `ui/`, `.streamlit/`, README | TBD |

## Workflow

- `main` is always deployable; Streamlit Cloud redeploys on every push to it.
- Work on `feature/<short-name>` branches; open a pull request with at least one reviewer.
- Track tasks in GitHub Issues and the Projects board; link each PR to its issue.

## Privacy

CVs are processed in memory and never written to disk or logs. Never commit real CVs or API keys.
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
