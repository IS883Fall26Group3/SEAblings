# Embedding evaluation

Calibrates the thresholds in `core/embeddings.py` (`skill_threshold`, `cos_low`, `cos_high`).
Each embedder has its own values, so run every step once per embedder.

```bash
pip install -r requirements-dev.txt
python eval/skillmatch.py local|gemini     # step 1
python eval/esco_pairs.py local|gemini     # step 2
python eval/extract.py gemini:gemini-flash-lite-latest   # step 3: skills from data/jobs, data/resume
python eval/label_sets.py extracted_gemini-flash-lite-latest.json   # step 3: build/refresh labels/
python eval/calibrate.py local extracted_gemini-flash-lite-latest.json   # step 4
python eval/doc_band.py gemini   # step 6 revisited: document band at the app's embedding length
```

`GEMINI_API_KEY` is read from the environment or, failing that, from `.streamlit/secrets.toml`
(`env.py`). Vectors are cached in `results/vectors_<embedder>.npz` and extractions in
`results/extracted_<model>.json`, so a run that hits a quota resumes where it stopped.

Free-tier limits met while running this (October 2026): 5 requests/minute and a small daily cap
for each flash model (overloaded "503" attempts and their retries count too), and 100 texts per
minute for `gemini-embedding-001`, where every text in a batch counts. `extract.py` spaces calls
13 s apart and stops at a daily cap; embeddings wait and retry. Embeddings are also capped at
30,000 tokens a minute (met in November 2026, once CVs and postings were embedded at 7,000
characters: 50 of them in one batch is far over it), so `embed_cached` limits each Gemini batch to
50 texts and 40,000 characters (about 10,000 tokens), 31 s apart.

## Plan

1. **SkillMatch-1K** (`skillmatch.py`): how well each embedder separates related from unrelated
   skills. A sanity check on the model, not a threshold: SkillMatch "related" means listed as
   alternatives in job ads (HTML/CSS), which is looser than our "the CV skill covers the job skill".
2. **ESCO** (`esco_pairs.py`): synonyms from `altLabels` vs the most similar *different* ESCO
   skill, for a first `skill_threshold`.
3. **Own labeled pairs** (`extract.py`, `label_sets.py`): 80 job descriptions (`data/jobs/`) and
   11 CVs (`data/resume/`, not in git). Two sheets in `labels/`, drafted by Claude, checked by the
   team (see Labeling guide).
4. **Calibration** (`calibrate.py`): reproduces `core/matcher.py` scoring (checked against
   `score_jobs`), sweeps `skill_threshold`, `cos_low`/`cos_high` and the weights (12,672
   settings), and scores each by mean per-CV Spearman and NDCG@5. Leave-one-CV-out: tune on 10
   CVs, test on the 11th, compare with the current settings on the same CV.

5. **Experience fit** (`experience.py`): total years vs relevant years by embedding ("roles")
   vs relevant years by LLM judge ("judge"); everything else in the score unchanged.
   `python eval/experience.py gemini gemini:gemini-flash-lite-latest extracted_gemini-flash-lite-latest.json`

6. **Profile similarity** (`similarity.py`): whole document ("document", the app default) vs
   summaries ("summary") vs best CV chunks per job ("chunks"); everything else unchanged.
   `python eval/similarity.py gemini:gemini-flash-lite-latest extracted_gemini-flash-lite-latest.json gemini local`
   Revisited (`doc_band.py`): the document band re-measured once the app embedded 7,000
   characters instead of 3,000; embeddings only, no LLM calls.

7. **Background rating** (`background.py`): profile similarity scaled by the judge's same / related
   / different rating of the field and kind of work. `python eval/background.py gemini:gemini-flash-lite-latest`

## Labeling guide

**`labels/cv_job.csv`**: 176 pairs, 16 jobs per CV (the 6 most similar by document cosine plus 10
spread over the rest), shuffled. Read the CV in `data/resume/` and the posting (`url`, or the
description in `data/jobs/`), then rate:

| Score | Meaning |
|---|---|
| 5 | Strong fit: meets nearly all requirements, would likely get an interview |
| 4 | Good fit: worth applying, one or two gaps |
| 3 | Stretch: transferable experience, missing a key requirement |
| 2 | Weak: some overlap, unlikely to be considered |
| 1 | Wrong field or far below the level asked |

Ignore work authorization, sponsorship and security clearance: the app doesn't model them, and
counting them would make almost every pair a 1.

**`labels/skill_pairs.csv`** (built after Gemini extraction): `yes` if a recruiter screening for
`job_skill` would accept `cv_skill` as covering it (same skill, alias or abbreviation, or a
specific instance: PostgreSQL covers SQL). `no` if it is only related, broader, or an alternative
tool (Tableau does not cover Power BI; HTML does not cover CSS).

**Who fills what.** `claude_draft` and `claude_note` are Claude's drafts. `reviewer_1` checks the
drafts and writes their own score (copy the draft if they agree). `reviewer_2` should label
**without looking** at the Claude or reviewer_1 columns (hide them), so the agreement between
reviewers is an honest ceiling. Disagreements by more than 1 point go in `notes` and get settled
together.

## Results

### Step 1: SkillMatch-1K (2,000 pairs, 2,648 unique skills)

| Embedder | AUC-PR | ROC-AUC | Best F1 (cutoff) | Median cosine related / unrelated | At app `skill_threshold` |
|---|---|---|---|---|---|
| bge-small-en-v1.5 (local) | 0.856 | 0.823 | 0.750 (0.575) | 0.642 / 0.534 | 0.85: 21 pairs flagged, precision 1.00, recall 0.02 |
| gemini-embedding-001 | not run yet | | | | |

Reference from the paper: Sentence-BERT 0.876 AUC-PR off the shelf, 0.969 fine-tuned.

- bge-small cosines sit in a narrow band (unrelated pairs still around 0.53), which is why the
  thresholds are high and sensitive to small changes.
- Some "unrelated" pairs are mislabeled (test plans/cases vs software testing), because negatives
  are random pairs that never appeared together in job ads. This caps any model's score.
- Per-pair cosines: `results/pairs_local.csv`.

### Step 2: ESCO synonyms vs near misses (600 concepts, labels of 1-4 words)

| Embedder | Synonym median | Hard-negative median | AUC-PR | Best F1 (cutoff) | At 0.85 |
|---|---|---|---|---|---|
| bge-small (local) | 0.918 | 0.850 | 0.794 | 0.691 (0.868) | precision 0.61, recall 0.77 |

- The current 0.85 sits right at the best-F1 cutoff, so it is a reasonable starting point.
- The "hard negatives" are noisy: ESCO keeps a skill and its knowledge form as separate concepts
  ("search engine optimisation" vs "conduct search engine optimisation", "lawn care" vs
  "provide lawn care"). The app should match these, so true precision is higher than shown.
  Real near misses do appear too ("shunt inbound loads" vs "shunt outbound loads", 0.96).
- Per-pair cosines: `results/esco_pairs_local.csv`. ESCO can't set the final threshold; step 3 does.

### Step 3: early findings (from Claude's draft labels, to be confirmed by the team's labels)

- **Document cosines are compressed for bge-small:** CV-job cosines run 0.59-0.72 (p5-p95, all
  880 pairs) against an app range of `cos_low` 0.55 to `cos_high` 0.90, so the semantic score only
  moves between about 0.1 and 0.5. A band near 0.59-0.72 would use the full scale (but would also
  give semantic similarity more weight, so it must be tuned together with the weights).
- **Document similarity barely tracks fit:** Spearman 0.21 between cosine and draft relevance
  (per CV from -0.14 to 0.80). Likely cause: `LocalEmbedder` reads only the first 512 tokens
  (about 2,000 characters), which for most CVs is the education section and the first job.
- **One hub job:** j062 (Knowledge, Data, and AI Strategy Lead Consultant) is in the top 7 by
  cosine for 10 of 11 CVs.
- **Offline extraction fails on CVs:** Llama-3.2-1B loops (repeats "Financial modeling" until the
  512-token cap), copies the prompt's example skills ("Python", "SQL", "Financial modeling") into
  a CV that has neither, and estimated 2.5 years for about 8. Tested on one CV (ADW), which failed
  at both 512 and 2,048 tokens; other CVs and job extraction weren't tried. Step 3 skills use Gemini.
- **Gemini flash-lite extraction** (flash hit its daily cap; flash-lite is the app's Auto
  fallback) worked on all 11 CVs and 80 jobs, with realistic years. Two quirks: languages are
  listed as skills ("English", "Chinese"), and "Financial Modeling" appears for AS and LB, whose CVs
  don't mention it. It is an example in `prompts/extract_profile.md`, so the example may leak.
- **Few skill pairs ever reach the threshold:** of 2,611 distinct (job skill, closest CV skill)
  pairs, 59 have a bge-small cosine of 0.85 or more.

### Step 4: calibration, local embedder (bge-small), skills from gemini-flash-lite

Labels: `cv_job_reviewed.csv`, mean of reviewer_1 and reviewer_2. **Caveat:** reviewer_1 equals
Claude's draft on all 176 pairs, `notes` repeats `claude_note` on every row, and reviewer_2 is
never more than 1 point from the draft. So these labels are mostly Claude's judgment; the
numbers below should be rerun if the reviews are redone independently.

**`skill_threshold`** (`labels/skill_pairs.csv`, 116 pairs, Claude's drafts, not reviewed yet):

| Threshold | Precision | Recall | F1 | F0.5 |
|---|---|---|---|---|
| 0.80 | 0.53 | 0.93 | 0.67 | 0.58 |
| **0.85 (app)** | 0.64 | 0.70 | 0.67 | 0.65 |
| 0.90 | 0.89 | 0.40 | 0.55 | 0.71 |

0.80-0.85 tie on F1; 0.90 is best if false matches are the worse error (F0.5). Misses at 0.85
include agile/scrum (0.84) and relational database tools/SQL (0.80); false matches include
tableau administration/tableau (0.92) and machine learning frameworks/machine learning (0.89).
No single threshold fixes both, because bge-small scores word overlap highly.

**Ranking** (176 CV-job pairs, 11 CVs):

| | Spearman | NDCG@5 |
|---|---|---|
| Skill coverage alone | 0.302 | 0.587 |
| Document cosine alone | 0.290 | 0.592 |
| Seniority fit alone | 0.069 | 0.402 |
| **Current settings** (0.85, 0.55-0.90, weights 0.5/0.3/0.2) | **0.363** | **0.650** |
| Best setting on all 11 CVs (in-sample) | 0.463 | 0.710 |
| Tuned on 10, tested on the 11th (mean of 11) | 0.356 | 0.568 |

- **Tuning does not beat the current settings on unseen CVs** (0.356 vs 0.363 Spearman, 0.568 vs
  0.650 NDCG@5). The in-sample gain is overfitting: the chosen settings change from fold to fold
  (`results/calibration_loco_local.csv`). Recommendation: keep the current values for bge-small.
- Ranking quality is limited by the inputs, not the knobs: `cos_low`/`cos_high` barely matter
  (best Spearman 0.454-0.463 across all values), and the weights are flat between 0.3 and 0.5 for
  skills and 0.2 and 0.3 for semantic. Bigger levers are what gets embedded (the 512-token cut),
  extraction quality, and more labeled CVs.
- Skill-pair drafts were corrected after this run (data analysis/data analytics and digital
  analytics/google analytics are now "yes", 42 yes in total). With them, local 0.85 gives
  precision 0.66, recall 0.69, F1 0.67, and the ranking numbers above are unchanged.

### Step 4: calibration, Gemini embedder (gemini-embedding-001, now the app default)

**At the current `skill_threshold` of 0.80, every required skill counts as matched** (mean skill
coverage 1.00 over the 176 pairs). Half of the score is then constant, and "missing skills" is
empty for every job. Gemini cosines are high and compressed: skill pairs labeled "yes" have a
median of 0.944, "no" pairs 0.899; CV-job document cosines run 0.73-0.80 (p5-p95).

| Threshold | Precision | Recall | F1 | F0.5 | Ranking Spearman / NDCG@5 (current weights, cos range) | Mean coverage |
|---|---|---|---|---|---|---|
| **0.80 (app)** | 0.37 | 1.00 | 0.54 | 0.42 | 0.331 / 0.574 | 1.00 |
| 0.85 | 0.40 | 1.00 | 0.57 | 0.45 | 0.315 / 0.588 | 0.76 |
| 0.875 | 0.43 | 1.00 | 0.60 | 0.49 | 0.210 / 0.483 | 0.44 |
| 0.90 | 0.51 | 0.93 | 0.66 | 0.56 | 0.377 / 0.616 | 0.26 |
| 0.925 | 0.68 | 0.76 | 0.72 | 0.70 | 0.280 / 0.608 | 0.14 |
| 0.95 | 0.89 | 0.40 | 0.56 | 0.72 | | |

- Skill pairs (Claude's drafts) point to 0.925 (best F1). The ranking column jumps around between
  neighbouring thresholds, so with 11 CVs it is too noisy to pick the threshold from.
- **Changed:** `GeminiEmbedder.skill_threshold` is now 0.925 (was 0.80). Matched/missing skills
  become meaningful (mean coverage 0.14 instead of 1.00); ranking moves from 0.331 to 0.280
  Spearman and from 0.574 to 0.608 NDCG@5 on these labels. Recheck with independent labels.
- Full grid with leave-one-CV-out: tuned 0.326 vs current 0.331 Spearman, 0.573 vs 0.574 NDCG@5,
  so, as for bge-small, tuning `cos_low`/`cos_high` and the weights doesn't generalize.
- With current settings Gemini ranks slightly worse than bge-small on these labels (0.331 vs
  0.363 Spearman, 0.574 vs 0.650 NDCG@5); its document cosine alone is a little better (0.315 vs
  0.290), its skill coverage at 0.80 is useless (0.032).
- The skill pairs were sampled by bge-small cosine, so they are not stratified around Gemini's
  decision boundary; a Gemini-specific sample would firm up the threshold.

### Step 5: experience fit, total vs relevant years (Gemini embedder, gemini-flash-lite)

Same 176 pairs and labels (with the same caveat), skills and profile similarity as in the app
(skill threshold 0.925, weights 0.5/0.3/0.2); only the experience fit changes. Roles were
re-extracted with the new `roles` field (3.8 per CV); the judge ran once per CV over its 16 jobs.

| Experience fit | Alone: Spearman / NDCG@5 | Full score: Spearman / NDCG@5 |
|---|---|---|
| Total years (current) | 0.084 / 0.405 | 0.280 / 0.608 |
| Roles, threshold 0.80 (best in-sample) | 0.197 / 0.423 | 0.316 / 0.574 |
| Roles, threshold chosen on 10 CVs, tested on the 11th | | 0.254 / 0.541 |
| **Judge** | **0.349 / 0.617** | **0.333 / 0.679** |

- **Judge is best** on both measures, and it has no setting tuned on these labels.
- **Roles doesn't generalize:** Gemini role-job cosines are compressed (p10 0.755, p90 0.815), so
  relevant years jump from 7.6 to 1.7 between thresholds 0.75 and 0.80; picked on 10 CVs, it does
  worse than total years on the 11th.
- **Requiring relevant experience helps even when a job states no minimum** (82 of 176 pairs):
  scoring those jobs 1.0, as the current app does, lowers the full score to 0.250 (roles) and
  0.251 (judge) Spearman. With no minimum, one relevant year counts as a full fit.
- The judge is strict: mean 0.8 relevant years against 7.1 total; on 39% of pairs it finds none
  where roles finds some. It is right on clear cases (SA/FP&A: 7.8 years of credit analysis; LB/
  strategy consulting: 6.4 years at EY and Deloitte) and arguably too strict on transfers (MS's
  R&D leadership for a life-science consulting job: 0 years, "not consulting").
- Cost per search: judge adds one LLM call (about 17 s for 16 jobs) on the free tier's 5 requests
  per minute and daily cap; roles adds only a few embedded texts.
- Extraction can double count: MS got company dates on one role and durations on the other two
  (12.7 years against 8 total), so relevant years are capped at total years in `score_jobs`.
- **Caveat on the judge's win:** the labels are mostly Claude's drafts, and the judge is also an
  LLM deciding "is this relevant experience"; part of the agreement may be shared LLM judgment.
  Its stability across reruns (temperature 0.1) wasn't measured. Confirm with independent human
  labels before making `EXPERIENCE_FIT = "judge"` the default. (Made the default in November 2026,
  before that confirmation; set `EXPERIENCE_FIT = "total"` to go back.)
- Per-pair years and fits: `results/experience_gemini.csv`.

### Step 6: profile similarity, document vs summary vs chunks

Measured with CVs and postings cut to 3,000 characters. The app now embeds 7,000, so the Gemini
document band below is replaced by the one in "Step 6, revisited"; the comparison of the three
modes is unchanged.

Same 176 pairs and labels (same caveat). CV summaries from the latest CV extraction;
`role_summary` extracted for the 70 labeled jobs (7 calls). Skills (threshold per embedder),
experience fit (total years) and weights unchanged. Each mode's band is the p5-p95 of its own
cosines (spread only, not labels); "chunks" = mean of each job's 3 closest 40-word CV chunks
(15.6 chunks per CV).

| Gemini embedder | Band | Alone: Spearman / NDCG@5 | Full score: Spearman / NDCG@5 |
|---|---|---|---|
| Document, app band 0.50-0.90 (before) | | | 0.280 / 0.608 |
| Document, p5-p95 band | 0.731-0.795 | **0.315** / 0.533 | **0.345** / 0.599 |
| Summary | 0.783-0.859 | 0.197 / **0.558** | 0.258 / 0.587 |
| Chunks, k=3 | 0.747-0.810 | 0.221 / 0.500 | 0.330 / 0.585 |

| Local embedder (bge-small) | Band | Alone | Full score |
|---|---|---|---|
| Document, app band 0.55-0.90 | | | 0.363 / 0.650 |
| Document, p5-p95 band | 0.586-0.740 | 0.290 / 0.592 | 0.364 / **0.651** |
| Summary | 0.560-0.729 | 0.246 / 0.549 | **0.369** / 0.604 |
| Chunks, k=3 | 0.590-0.697 | 0.260 / 0.593 | 0.336 / 0.611 |

- **Neither technique beats the whole document on the Gemini embedder.** Summary ranks worst
  (0.258 vs 0.345 Spearman in the full score); chunks is close but below. On bge-small summary
  ties document on Spearman and loses on NDCG@5.
- Summary helps some top picks (AS: the 5-rated financial analyst job instead of a 3.5;
  KF: the 3.5-rated life-science analyst job instead of a 1.5) but promotes a generic job
  everywhere else: j077's summary ("a qualified product leader and business analyst ... highly
  organized ... driving complex technical and business initiatives across global teams") is
  Gemini's top match for 4 of 11 CVs, all rated 1.0-1.5. Generic leadership wording in short
  summaries hides the specifics that separate jobs (here, alternative-asset expertise).
- The summaries are not written alike: CV summaries run three sentences and end on education
  ("Holds an MBA and MS ..."), job summaries are often one sentence. A shared template (work,
  domain, level; no education) is the obvious next try, at 11 calls to re-extract the CVs.
- Chunks: k=5 > k=3 > k=1 on Gemini (0.342 / 0.330 / 0.302 full Spearman); a single best chunk
  over-rewards one matching bullet.
- The bigger, free gain is the band: the document band set to its p5-p95 raises Gemini's full
  score from 0.280 to 0.345 Spearman (NDCG@5 0.608 to 0.599). Bands come from the cosine spread,
  not the labels, but the earlier leave-one-CV-out check found band tuning didn't generalize, so
  confirm with independent labels.
- **Changed:** the app default is `"document"` with the p5-p95 band: Gemini `cos_low`/`cos_high`
  0.731/0.795 (was 0.5/0.9), bge-small 0.586/0.740 (was 0.55/0.9). `sim_bands` hold the summary and
  chunks bands above for when those modes are chosen. Per-pair cosines:
  `results/similarity_<embedder>.csv`. (Gemini's document band is now 0.709/0.786: see below.)

### Step 6, revisited: the document band at 7,000 characters

**Why.** Every text was cut to 3,000 characters before embedding, and the LLM read only the first
3,000 characters of each posting. That's too short for real postings: only 35% of the 80 in
`data/jobs/` fit (median 4,295 characters, p95 8,912). Among the labeled pairs, 8 of 11 CVs and 50 of
70 postings were cut. A posting's requirements (minimum years, required skills) usually come last,
so they were often the part dropped. For example, a LinkedIn posting tested in the app spent ~1,500
of its 3,000 characters on page text, company pitch and pay, and its "8+ years" line was never read,
so experience fit scored it against no minimum at all.

**What changed in the app.**

| Limit | Before | Now | Where |
|---|---|---|---|
| LLM reading a posting (skills, years, title) | 3,000 | 10,000 (98% of postings in full) | `Settings.max_jd_chars` |
| Embedding, Gemini | 3,000 | 7,000 (85% in full; the model reads at most 2,048 tokens) | `GeminiEmbedder.max_chars` |
| Embedding, bge-small | 3,000 | 3,000 (it reads 512 tokens anyway) | `LocalEmbedder.max_chars` |
| Text kept from an uploaded posting | 6,000 | 50,000 | `Settings.max_jd_file_chars` |

The embedding limit now belongs to each embedder, because it depends on the model, and the eval
scripts use the same limit as the app.

**How it was measured.** `doc_band.py`: the same 176 labeled pairs, embedded with
`gemini-embedding-001` at 3,000 and at 7,000 characters; the band is the p5-p95 of the cosines, as
before (spread only, not labels). Embeddings only: skills, experience fit and weights aren't
involved. The 3,000-character run reproduced the original step 6 (band 0.733-0.798 vs 0.731-0.795,
the same 0.315 Spearman; per-pair cosines within 0.01, correlation 0.992; the small gaps come from
slightly different copies of the CVs).

| Gemini, document mode | 3,000 characters (before) | 7,000 characters (now) |
|---|---|---|
| Texts cut | 8/11 CVs, 50/70 postings | 0/11 CVs, 11/70 postings |
| Band (p5-p95) | 0.731-0.795 | **0.709-0.786** |
| Median cosine | 0.766 | 0.748 |
| Similarity alone: Spearman / NDCG@5 | 0.315 / 0.542 | 0.291 / **0.581** |
| Pairs at 0% / 100%, old band 0.731-0.795 | 5% / 5% | **24%** / 4% |
| Pairs at 0% / 100%, new band 0.709-0.786 | | 5% / 5% |

- **Longer texts score lower.** The extra text is more of each posting, but also more benefits, pay
  and legal wording that no CV shares, so every cosine drops a little (median 0.766 to 0.748).
- **The band had to move with the length.** Kept at 0.731-0.795, a quarter of the pairs would show
  0% profile similarity and lose all 30 of its points; the new band gives the intended 5% / 5%.
- **Ranking: no clear change.** Whole-ranking Spearman dropped (0.315 to 0.291) and the top-5
  NDCG@5 rose (0.542 to 0.581). With 11 CVs, both differences are within noise. The case for 7,000
  is that the LLM and the embedding now see the whole posting, not a better ranking.
- **Not re-measured:** the full score (skills + similarity + experience, steps 4 and 5) needs new
  LLM extractions at 10,000 characters (`extract.py`, then `calibrate.py`); the chunks band, which
  embeds postings at the new length (the summary band compares short LLM summaries, so it holds);
  and bge-small, unchanged at 3,000 characters.
- **The judge reads more too:** `EXPERIENCE_FIT = "judge"` read only the first 1,500 characters
  of each posting in step 5, mostly page text and company pitch on LinkedIn copies. It now reads
  10,000 with Gemini, like requirement extraction, and still 1,500 with the local model. Step 5's
  judge results were measured at 1,500: re-run `experience.py` (about 11 LLM calls) to update them.
- **In the app:** one search now embeds up to about 19,000 tokens (a CV and 10 postings) of the free
  tier's 30,000 a minute, so two quick searches can reach the limit. The app then waits as long as
  Gemini asks (at most a minute), says so in the progress panel, and tries once more
  (`embed_with_wait` in `core/embeddings.py`).
- **Changed:** Gemini `cos_low`/`cos_high` 0.709/0.786 (was 0.731/0.795). Per-pair cosines at both
  lengths: `results/doc_band_gemini.csv`.

### Step 7: profile similarity adjusted by the judge's field and kind-of-work rating

**Why.** The embedding rates two career documents as alike when they share generic wording
(management, operations, team, Boston). In the app, a data analyst CV against a café head chef
posting got 36% profile similarity although nothing about the two is alike. The judge (the default
experience fit) already reads the CV and every posting in one call, so it now also rates each job's
**field** and **kind of work** against the CV: same, related or different (`field_match`,
`work_match`). Profile similarity = banded embedding score × a factor from that rating, which can
only lower it. No extra call.

**How it was measured.** `background.py`: the real judge prompt on the 176 labeled pairs (one call
per CV; 175 pairs rated, one job missing from a reply), with requirements not extracted (the judge
saw "Required skills: not extracted"). Embedding scores at 7,000 characters from `doc_band.py`. Six
factor sets compared, on similarity alone.

| Judge's rating (field, work) | Pairs | Mean label |
|---|---|---|
| different, different | 95 | 1.41 |
| different, related | 20 | 1.65 |
| related, different | 3 | 1.33 |
| related, related | 27 | 1.78 |
| related, same | 5 | 2.00 |
| same, related | 3 | 2.50 |
| same, same | 22 | 3.14 |

| Profile similarity (similarity alone vs labels) | Spearman | NDCG@5 | Mean on pairs rated 1-1.5 | Share of those at 0% |
|---|---|---|---|---|
| Embedding only (before) | 0.288 | 0.577 | 43% | 5% |
| **Embedding × average, same 1 / related 0.5 / different 0 (now)** | **0.459** | **0.710** | 7% | 71% |
| Embedding × average, related 0.75 | 0.448 | 0.682 | 9% | 71% |
| Embedding × average, only "different" lowers | 0.431 | 0.659 | 12% | 71% |
| Embedding × 0 only when both are different | 0.437 | 0.641 | 15% | 71% |
| Embedding × 0 when either is different | 0.380 | 0.640 | 7% | 84% |
| Rating alone, no embedding | 0.465 | 0.674 | | |

- **The rating tracks the labels closely:** mean label 1.41 for "different, different" up to 3.14
  for "same, same".
- **Adjusting by the rating is the biggest gain in the eval so far:** Spearman 0.288 to 0.459,
  NDCG@5 0.577 to 0.710. Pairs the team rated as wrong field drop from 43% to 7% on average.
- **The embedding still earns its place:** the rating alone ranks about as well overall (0.465) but
  worse at the top (NDCG@5 0.674 vs 0.710), because it gives the many "same, same" jobs one value.
- **Not tuned to the labels:** the chosen set was picked first on every leave-one-CV-out split, and
  the held-out mean equals the in-sample 0.459. Per CV it beats the embedding on 6 of 11, so the
  gain is concentrated in CVs with clearly off-field jobs.
- **Misses:** ADW's best-labeled job (j022, label 3.0) was rated "different, different" and drops
  from 61% to 0%. CVs whose fit is transferable rather than same-field lose most.
- **Caveats:** the labels are mostly Claude's drafts, and the judge is an LLM, so part of the
  agreement may be shared LLM judgment (as in step 5). The full score wasn't re-measured (it needs
  new skill extractions). Rerun stability wasn't measured.
- **In the app:** applied whenever the judge runs (the default experience fit). With
  `EXPERIENCE_FIT = "total"` or `"roles"`, or when the judge call fails, profile similarity is the
  embedding score alone. Per-pair ratings and scores: `results/background_gemini-flash-lite-latest.csv`.

### Success metric: requirement accuracy test set

The metric fixed in the workshop checkpoint is **requirement classification accuracy**: the share of
requirements where the app's label (met / partial / missing) matches the team's answer key.
`requirement_set.py` builds the test set once:

| File | What it holds |
|---|---|
| `labels/requirements_frozen.json` | The 12 test postings' requirements (97 skills + 13 minimum-years rows), extracted once by the app's prompt with gemini-flash-lite and kept fixed. |
| `labels/requirements/<name>.csv` | The answer key, one file per labeler. 16 CV-job pairs: 14 good or stretch fits from `cv_job_reviewed.csv` and 2 confusing pairs (poor fits that share several skills). 4 of the pairs are labeled twice, to measure agreement. |
| `labels/requirements/<name>_extraction.csv` | Requirements the extraction missed or invented, for each posting. |
| `labels/awkward_inputs.csv` | 4 behavior checks (empty CV, a recipe, a news item, a two-line posting), with the expected behavior written in advance. |
| `labels/requirement_postings.md` | The test postings in full, for reading while labeling. |

How to label is in `labeling_guide.md`. The team labels **before** the app's labels are run on these
pairs, and commits the files; the commit history shows the order.

**Results** (`python eval/requirement_accuracy.py --app 3`, gemini-flash-lite for every LLM step,
per-requirement rows in `results/requirement_accuracy.csv`):

| | Accuracy | Balanced | Met right | Partial right | Missing right |
|---|---|---|---|---|---|
| Keyword baseline (met only if the requirement's words are in the CV) | 50.9% (56/110) | 46.5% | 39% | 0% | 100% |
| App, run 1 | 67.3% (74/110) | 64.4% | 87% | 26% | 80% |
| App, run 2 | 66.4% (73/110) | 63.6% | 87% | 26% | 78% |
| App, run 3 | 76.4% (84/110) | 73.9% | 89% | 42% | 90% |
| **App, mean** | **70.0%** | 67.3% | 88% | 31% | 83% |

- **Target ≥ 75%: not reached on average.** The runs differ because the LLM's labels differ even at
  temperature 0: 93 of the 110 requirements got the same label in all 3 runs; the other 17 flipped,
  almost all between met and partial or between partial and missing.
- **By part (mean):** skills 70% (n=97), minimum years 69% (n=13), confusing pairs 90% (n=17),
  regular pairs 66% (n=93).
- **Labeler agreement:** 88% on the 4 double-labeled pairs (25 requirements), Cohen's kappa 0.82. All
  3 disagreements were met vs partial, the same boundary where the app errs most.
- **Quotes:** 170 of 171 quotes on met or partial labels were found word for word in the CV; the
  check dropped the other.

**How the labeler was improved without tuning on the test set.**
- **First version: 63.3%** mean over 3 runs (61.8-65.5%), with partial labels 6% right. Looking at
  its errors showed one bug: on one CV, SQL, Python and ETL were labeled missing in every run although
  the CV names them. Embedding search ranked the line naming them 4th to 8th of 28, behind lines
  about general technical work, so it never reached the model. Retrieval now adds CV lines that name
  the requirement (or a related skill from the knowledge base) word for word
  (`core/requirements.py: retrieve_evidence`).
- **Prompt changes were tuned on a practice set only** (`practice_set.py`): 8 pairs on the 2 CVs that
  are in no test pair (AYP, VV) and 8 postings that aren't test postings, 64 requirements. Its answer
  key was labeled by Claude (an AI assistant) by `labeling_guide.md`, before the app ran on those
  pairs; it is for tuning only and never reported as the metric. The labeling prompt now follows the
  guide's rules (study only = partial, soft skills shown in action, broad areas, same-kind tools).
  Practice accuracy: 75.8% before, 78.1% after (mean of 2 and 3 runs).
- Then the test set was run once with the new version (`VERSION = "v2"` in
  `requirement_accuracy.py`): the table above. Tried and dropped: majority vote over 3 labeling
  calls (same accuracy as the mean, 3 times the calls); gemini-flash for labeling (overloaded, 503).

**Awkward inputs** (`labels/awkward_inputs.csv`, run through the real app): all 4 pass. A2 (a recipe
as the CV) and A3 (a news item as the posting) first got match scores of 0 and 25 as if they were
real; `core/input_checks.py` now stops both with a clear message. The extraction step says what
kind of document it read (no extra call), and a posting with no requirements is never scored.

### Efficiency: calls, tokens and time per search

`python eval/measure_cost.py one` and `... ten` drive the real app headlessly and record every Gemini
call (rows in `results/cost_per_search.csv`). Gemini Flash Lite, a fresh session each time:

| Step | 1 CV vs 1 posting: tokens in / out | 1 CV vs 10 postings: tokens in / out | Seconds (10 postings) |
|---|---|---|---|
| Extract the CV profile | 618 / 325 | 645 / 349 | 3.4 |
| Extract the postings' requirements | 459 / 233 | 1,000 / 2,150 | 7.5 |
| Judge relevant experience | 744 / 173 | 1,549 / 955 | 3.8 |
| Label each requirement (retrieval) | 1,620 / 386 | 10,180 / 2,112 | 5.2 |
| **Search total: 4 LLM calls** | **3,441 / 1,117** | **13,374 / 5,566** | **29-33 s** wall |
| Texts embedded (3 calls) | 25 | 70 | 1.7-11.8 |
| Advice on one job: 2 calls (tool calling) | | 1,804 / 129 | 5 s |

- **LLM calls don't grow with the number of jobs:** each step sends all jobs in one call (4 calls a
  search; the offline model makes one call per job instead).
- **Tokens grow with jobs, mostly in the requirement check** (76% of input tokens at 10 jobs): each
  requirement carries its retrieved CV lines and knowledge-base families.
- **The binding free-tier limit is embeddings**, 1,000 texts a day per Google project: about 40
  one-job or 14 ten-job searches a day. Repeated texts are embedded once per session; the knowledge
  base is embedded ahead of time.
- **Ways to cut:** send each distinct requirement once when several jobs share it; drop CV lines
  retrieved for more than one requirement of the same job; use a separate Google project (its own
  quota) for the demo.
- **Advice** in this run called `compute_match_score` only, since the top job had no gaps to search
  for; `search_cv` runs when a gap needs checking.

## Data

| File | Source | License | In git |
|---|---|---|---|
| `data/skillmatch/test.csv` | [TechWolf/SkillMatch-1K](https://huggingface.co/datasets/TechWolf/SkillMatch-1K) ([paper](https://arxiv.org/abs/2410.05006)) | MIT | yes |
| `data/esco/skills_en.csv` | ESCO v1.2 skills, English | ESCO terms of use | no (9 MB) |
| `data/jobs/jobs_jsearch.json` | 80 postings collected through JSearch | posting sites' terms | yes |
| `data/resume/*.txt` | 11 classmate CVs, initials only | private | **no**: real people, share only within the team |

The ESCO copy came from an unofficial
[Hugging Face mirror](https://huggingface.co/spaces/edouardlgp/Job_Classification/blob/main/skills_en.csv)
because the ESCO site blocked automated downloads. For anything cited in the report, download the
official CSV (English, skills) from the ESCO portal's download section
([file structure](https://esco.ec.europa.eu/en/structure-esco-downloadable-datasets))
into `data/esco/`.
