# Labeling guide: met / partial / missing

This is the answer key for our success metric, **requirement classification accuracy**: the share of
requirements where the app's label matches ours. The number is only worth something if our labels
are made carefully, by us, before we see what the app says.

## Before you start

- **Your file:** `eval/labels/requirements/<your name>.csv`. Rows with `role` = `main` are your
  pairs. Rows with `role` = `second` are one pair someone else also labels; you label it separately,
  so we can measure how often two people agree. Everyone has their own file, so nobody's edits
  collide in git.
- **Open it in Excel or Google Sheets,** and save it back as CSV with the same name.
- **The CVs** are in `eval/data/resume/` (never in git). Ask Tigor for the files through our private
  Drive folder if you don't have them.
- **The postings** are in `eval/labels/requirement_postings.md`, in full, with links to the originals.
- **Time:** about 10-15 minutes per pair.

## The rules

1. **Never run the app on a test pair before the labels are committed,** and don't look at what it
   says. If you see the app's answer for a pair before labeling it, tell the team and we swap the pair.
2. **No AI tools.** Label from your own reading of the CV.
3. **Don't discuss your `second` pair** with its main labeler until both of you have committed.
4. **Label every row,** even a requirement you think shouldn't be on the list. If the list itself is
   wrong, also write that in your extraction file (see "Checking the requirement lists").
5. **Use the CV only.** Don't guess what the person probably knows. Ignore work authorization,
   location and pay.

## The three labels

| Label | When | Examples |
|---|---|---|
| **met** | The CV shows this skill, the same skill under another name, or a specific instance of it. A skill listed only in the skills section counts. | "MS Excel" for Excel. "PostgreSQL" for SQL. "Built DCF models" for financial modeling. A role at a biotech firm for "Biotech". |
| **partial** | The CV shows something related but not the same, a lower level, or the skill only from study. | Tableau for Power BI (same kind of tool). "Python (basic)" for Python. Energy sector for "oil and gas". Spark experience for "Databricks". FP&A coursework or a class project only, with no work use. |
| **missing** | Nothing in the CV shows it. | A shared generic word isn't enough: "managed inventory" doesn't show "client management". A job title alone isn't enough: "Data Analyst" doesn't prove SQL. |

**When torn between two labels,** pick the one a recruiter screening this CV would most likely choose,
and write why in `note` (a few words).

## Special cases

- **Minimum years** (rows of type `years`, e.g. "3+ years of relevant experience"):
  1. Add up the years in roles that do the same kind of work or are in the same field as the job.
     Internships count only if they are in the same function. Education doesn't count. Overlapping
     roles count once.
  2. **met** if those years reach the minimum, **partial** if they reach at least half of it,
     **missing** otherwise.
  3. Write the years you counted in `note` (e.g. "4.5 yrs: two analyst roles").
- **Industry or domain** ("Life Sciences", "Biotech"): **met** if a role was in that industry,
  **partial** if in a neighboring one (pharma services for biotech, energy for oil and gas),
  **missing** otherwise.
- **Soft skills** ("Stakeholder Engagement", "Written Communication", "Workshop Facilitation"):
  **met** if the CV names it or a line clearly shows it in action ("presented quarterly results to the
  CFO" shows executive presentations). **partial** if it is only implied ("worked with teams").
- **Methods or broad areas** ("Data Analysis", "Business Analysis", "Management Consulting"):
  **met** if a role or line is clearly that kind of work, **partial** if the work only touches it.
- **Compound requirements** ("SQL and Python" in one row): **met** only if both are shown,
  **partial** if one is.

## What to write

- `label`: exactly `met`, `partial` or `missing` (lowercase).
- `note`: required for every `partial` and every hard call; a few words. **Don't paste CV text with
  names, contact details or employers into the sheet**; describe it ("credit analyst role, 3 yrs").
- Don't change the other columns or the row order.

## Checking the requirement lists

The requirement lists were extracted once by the app's own prompt and frozen
(`eval/labels/requirements_frozen.json`), so we and the app label the same list. Each posting is
checked by one person, listed in `eval/labels/requirements/<your name>_extraction.csv`. While you read
those postings, fill in:

- `missed`: requirements the posting clearly states as required that aren't on the list, separated
  by `;`.
- `invented`: list items the posting doesn't really require (only "nice to have", or not there).

Missed and invented requirements are counted separately in the report. They aren't added to the
accuracy number, so the list you label stays the same.

## When you're done

Commit your two files from your own GitHub account, on a branch, with a message like
`Label requirements for pairs P01, P05, P09, P13`. The commit time shows our labels came before we
ran the app on these pairs.

## The awkward inputs

`eval/labels/awkward_inputs.csv` lists 4 inputs that test how the app behaves (an empty CV, a recipe
instead of a CV, a news item instead of a posting, a two-line posting). Their expected behavior is
already written. Check you agree with it before anyone runs them. These are pass / fail checks and
aren't part of the accuracy number.
