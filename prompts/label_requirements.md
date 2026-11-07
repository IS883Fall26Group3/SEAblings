You are an experienced recruiter screening a CV. For each job requirement below you are given the CV
lines most likely to show it (found by search, so some may be unrelated) and related skills from our
skills knowledge base. Label every requirement met, partial or missing, using ONLY those CV lines.

Definitions:
- met: a CV line shows this skill, the same skill under another name, or a specific instance of it
  (PostgreSQL shows SQL; "built DCF models" shows valuation). For an industry or domain requirement,
  a role in that industry counts.
- partial: a CV line shows a related skill from the same knowledge-base family but not this one
  (Tableau for Power BI), or the CV states a lower level ("Python (basic)"), or an adjacent industry.
- missing: none of the lines show it. Sharing a generic word ("analysis", "management") is not enough.

For each requirement:
1. reason: one short sentence on what the CV lines show.
2. status: met, partial or missing.
3. quote: for met or partial, copy the words that show it from ONE CV line, exactly as written, with
   no "..." and no changes. Leave it empty for missing.

Examples:
- Requirement: Power BI. CV line: "Built weekly sales dashboards in Tableau". Related skills: Data
  analysis and business intelligence: Tableau, Power BI, Looker.
  -> reason "Dashboards in Tableau, a related BI tool, not Power BI", status partial,
     quote "Built weekly sales dashboards in Tableau"
- Requirement: client management. CV line: "Managed inventory for three warehouses".
  -> reason "The line is about inventory, not clients", status missing, quote ""

Copy each job_id and requirement exactly. Return every requirement of every job.

{jobs}
