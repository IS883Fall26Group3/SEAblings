"""Build data/skill_aliases.csv (variant -> standard skill name) from three layers, first match wins:
  1. MANUAL: typos and everyday names for common tools ("Phyton", "MS Excel" -> "Excel")
  2. O*NET 31.0 software skills (data/onet/software_skills.csv): named tools and software
  3. ESCO skills (data/esco/skills_en.csv): preferred and alternative names of functional skills
A variant that one layer maps to two different names is ambiguous there and left to the next layer.
Only names a CV or posting would use as a skill (at most 6 words) are kept.

usage (from the repo root): python scripts/build_skill_aliases.py
Sources: O*NET 31.0 Database, U.S. Department of Labor/ETA, CC BY 4.0 (onetcenter.org);
ESCO, European Commission (esco.ec.europa.eu), reused under its terms of use."""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.skill_names import ALIASES, normalize  # noqa: E402

ONET = ROOT / "data" / "onet" / "software_skills.csv"
ESCO = ROOT / "data" / "esco" / "skills_en.csv"
MAX_WORDS = 6

MANUAL = {  # standard name: variants (typos, vendor prefixes, everyday short names)
    "Excel": ["excel", "ms excel", "microsoft excel", "advanced excel", "excel advanced"],
    "PowerPoint": ["powerpoint", "power point", "ms powerpoint", "microsoft powerpoint", "ms power point"],
    "Word": ["ms word", "microsoft word"],
    "Power BI": ["power bi", "powerbi", "microsoft power bi", "ms power bi"],
    "Python": ["python", "phyton", "python programming", "python computer programming"],
    "SQL": ["sql", "structured query language", "structured query language sql"],
    "R": ["r", "r programming", "r programming language"],
    "JavaScript": ["javascript", "js", "java script"],
    "Tableau": ["tableau", "tableau software"],
    "Salesforce": ["salesforce", "salesforce crm", "salesforce.com", "salesforce.com salesforce crm",
                   "salesforce software"],
    "SAP": ["sap", "sap software", "sap erp"],
    "Google Analytics": ["google analytics", "ga4", "google analytics 4"],
    "think-cell": ["think cell", "thinkcell"],
    "Photoshop": ["photoshop", "adobe photoshop"],
    "Generative AI": ["generative ai", "genai", "gen ai"],
    "Large Language Models": ["llm", "llms", "large language models", "large language model"],
    "Machine Learning": ["machine learning"],
    "Project Management": ["project management", "project mgmt"],
    "Amazon Web Services": ["aws", "amazon web services", "amazon aws"],
    "Google Cloud Platform": ["gcp", "google cloud", "google cloud platform"],
    "Microsoft Azure": ["azure", "microsoft azure", "ms azure"],
    "Microsoft Office": ["ms office", "microsoft office", "office 365", "microsoft 365", "ms office suite",
                         "microsoft office suite", "office suite"],
    "SQL Server": ["sql server", "ms sql server", "microsoft sql server", "mssql"],
}

VENDORS = ["Microsoft", "Adobe", "Google", "IBM", "Oracle", "SAP", "Amazon", "Autodesk", "Atlassian",
           "Apple", "Intuit"]
# Product names that are everyday words: "Access" or "Project" alone is not the Microsoft product
GENERIC = {"access", "word", "project", "teams", "office", "publisher", "forms", "dynamics", "outlook",
           "lists", "stream", "planner", "analytics", "ads", "sheets", "docs", "drive", "cloud",
           "search", "maps", "photos", "pages", "numbers", "keynote", "notes", "mail", "calendar",
           "software", "server", "studio", "business", "data", "sites", "classroom", "workspace",
           "database", "reports", "connect", "mechanical", "smoke", "combustion", "world", "video", "meet",
           "scholar", "checkout", "money", "paint", "works", "edge", "exchange", "designer", "clinical",
           "inventory", "maintenance", "middleware", "erp"}
# General ideas O*NET files under one vendor's product: "Business Intelligence" alone is not SAP's
GENERIC_PHRASES = {"business intelligence", "digital analytics", "data pipeline", "building systems",
                   "content manager", "network monitor", "enterprise manager", "application server",
                   "virtual server", "learning management", "advanced procurement", "customs management",
                   "flow manufacturing", "manufacturing scheduling", "total quality control management",
                   "fraud and abuse management system", "self service human resources", "remote data capture",
                   "data integrator", "recovery manager", "office software", "inventory management",
                   "business and customer relations management software", "power systems software"}


def acronym_variants(name: str) -> list[str]:
    """'Structured query language SQL' -> ['Structured query language', 'SQL'] when the trailing
    capitals are the initials of the words before them; 'Python Tools (PTVS)' -> without the brackets."""
    out = []
    inner = re.search(r"\(([^)]*)\)", name)
    if inner:
        out += [re.sub(r"\s*\([^)]*\)", "", name).strip(), inner[1].strip()]
    words = name.split()
    if len(words) >= 3 and re.fullmatch(r"[A-Z][A-Z0-9]{1,6}", words[-1]):
        acr, before = words[-1], words[:-1]
        initials = "".join(w[0] for w in before[-len(acr):]).upper()
        out.append(" ".join(before))
        if initials == acr:
            out.append(acr)
    return out


def onet_variants(name: str) -> list[str]:
    out = [name] + acronym_variants(name)
    for v in VENDORS:
        if name.startswith(v + " "):
            rest = name[len(v) + 1:]
            if normalize(rest) not in GENERIC | GENERIC_PHRASES and len(normalize(rest)) >= 3:
                out.append(rest)
    return out


STOP = {"and", "or", "of", "the", "a", "an", "in", "on", "for", "to", "with", "by", "use", "using", "apply",
        "perform", "manage", "managing", "management", "skills", "skill", "techniques", "principles",
        "microsoft", "ms"}


def _roots(text: str) -> set[str]:
    """Crude word roots: the first 5 letters of each content word ('negotiation' ~ 'negotiate')."""
    return {w[:5] for w in normalize(text).split() if w not in STOP and len(w) >= 3}


def esco_alternatives(preferred: str, alts: list[str]) -> list[str]:
    """ESCO's alternative names that are wordings of the same skill. Some are broader or neighbouring
    ideas filed under it ('data analysis' under 'statistics', 'contract law' under 'manage contracts'),
    so both must have the same word roots ('contract management' ~ 'manage contracts'); one everyday
    word alone ('access') is never kept."""
    roots = _roots(preferred)
    return [a for a in alts if (r := _roots(a)) and r == roots
            and not (len(normalize(a).split()) == 1 and normalize(a) in GENERIC | {"access"})]


def layer(pairs) -> dict[str, str]:
    """variant -> name for (variant, name) pairs; variants claimed by two names are dropped."""
    seen = defaultdict(set)
    for variant, name in pairs:
        key = normalize(variant)
        if key and len(key.split()) <= MAX_WORDS:
            seen[key].add(name)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def main():
    manual = layer((v, name) for name, vs in MANUAL.items() for v in vs + [name])
    with open(ONET, encoding="utf-8") as f:
        tools = sorted({row["Workplace Example"].strip() for row in csv.DictReader(f)})
    onet = layer((v, t) for t in tools for v in onet_variants(t))
    with open(ESCO, encoding="utf-8") as f:
        esco_rows = list(csv.DictReader(f))
    esco = layer((v, r["preferredLabel"]) for r in esco_rows
                 for v in [r["preferredLabel"], re.sub(r"\s*\([^)]*\)", "", r["preferredLabel"])]
                 + esco_alternatives(r["preferredLabel"], [a for a in r["altLabels"].split("\n") if a]))

    table = {}
    for source, d in (("manual", manual), ("onet", onet), ("esco", esco)):
        for k, v in d.items():
            table.setdefault(k, (v, source))
    rows = sorted((k, v, s) for k, (v, s) in table.items() if k != normalize(v) or s == "manual")
    with open(ALIASES, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "canonical", "source"])
        w.writerows(rows)
    counts = {s: sum(r[2] == s for r in rows) for s in ("manual", "onet", "esco")}
    print(f"{len(rows)} variants -> {ALIASES.relative_to(ROOT)} {counts} "
          f"(from {len(tools)} O*NET tools, {len(esco_rows)} ESCO skills)")


if __name__ == "__main__":
    main()
