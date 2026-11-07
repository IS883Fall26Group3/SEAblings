"""Build the skills knowledge base (data/skill_kb.csv): the corpus the app retrieves from when it
decides whether a job requirement is met, partially met or missing (core/knowledge.py).

Each entry is one family of related skills, so "Tableau" and "Power BI" sit together and the LLM can
call one a partial match for the other. Two sources:
  1. HAND: business-skill families an MBA CV or posting uses (finance, strategy, marketing, ...),
     written by the team.
  2. O*NET 31.0 software categories (data/onet/software_skills.csv): each category with its 15 most
     common tools, hot technologies first.

usage (from the repo root):
  python scripts/build_skill_kb.py            # write data/skill_kb.csv
  python scripts/build_skill_kb.py --embed    # also embed it with Gemini (needs GEMINI_API_KEY in
                                              # .streamlit/secrets.toml), so the app doesn't have to
Source: O*NET 31.0 Database, U.S. Department of Labor/ETA, CC BY 4.0 (onetcenter.org)."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.knowledge import KB_CSV, load_kb, save_vectors  # noqa: E402

ONET = ROOT / "data" / "onet" / "software_skills.csv"
TOOLS_PER_CATEGORY = 15

HAND = [  # (family, description, related skills)
    ("Financial planning and analysis", "Planning, budgeting and forecasting a company's finances",
     "budgeting; forecasting; financial planning; FP&A; variance analysis; scenario analysis; cost analysis; "
     "annual operating plan"),
    ("Financial modeling and valuation", "Building models that value a business or an investment",
     "financial modeling; valuation; DCF; discounted cash flow; LBO modeling; three-statement model; "
     "sensitivity analysis; comparable companies analysis"),
    ("Accounting and financial reporting", "Recording and reporting financial results",
     "accounting; financial reporting; GAAP; IFRS; reconciliation; month-end close; audit; bookkeeping"),
    ("Corporate finance and investing", "Deciding where a company or investor puts its money",
     "capital budgeting; M&A; mergers and acquisitions; due diligence; investment analysis; "
     "portfolio management; credit analysis; private equity; venture capital"),
    ("Market and customer research", "Learning about markets, competitors and customers",
     "market research; competitive analysis; customer insights; surveys; focus groups; market sizing; "
     "consumer research; industry analysis"),
    ("Marketing and brand management", "Planning and running marketing for a brand or product",
     "brand management; digital marketing; marketing strategy; campaign management; content marketing; "
     "SEO; SEM; social media marketing; go-to-market strategy; marketing analytics"),
    ("Sales and business development", "Winning and growing customers and partners",
     "sales; business development; account management; client management; lead generation; CRM; "
     "negotiation; partnerships; pipeline management"),
    ("Strategy and consulting", "Analyzing business problems and recommending what to do",
     "corporate strategy; strategic planning; business strategy; structured problem solving; "
     "hypothesis-driven analysis; management consulting; business case development; issue trees"),
    ("Product management", "Deciding what a product does and guiding it to launch",
     "product management; product strategy; product roadmap; user research; requirements gathering; "
     "product launch; user stories; product discovery"),
    ("Product analytics and experimentation", "Measuring how a product is used and testing changes",
     "product analytics; A/B testing; experimentation; funnel analysis; cohort analysis; "
     "causal inference; growth analytics; user metrics"),
    ("Data analysis and business intelligence", "Turning data into reports, dashboards and insights",
     "data analysis; data visualization; dashboards; KPI reporting; business intelligence; Tableau; "
     "Power BI; Looker; Excel; reporting automation"),
    ("Data engineering", "Building the pipelines and stores that analysis runs on",
     "ETL pipelines; data modeling; data warehousing; SQL; dbt; Snowflake; Airflow; data pipelines"),
    ("Statistics and machine learning", "Statistical and predictive models",
     "statistics; regression; machine learning; predictive modeling; forecasting models; Python; R; "
     "time series analysis"),
    ("Operations and supply chain", "Running and improving how goods and services are delivered",
     "operations management; supply chain management; logistics; procurement; inventory management; "
     "lean; Six Sigma; process improvement; demand planning"),
    ("Project and program management", "Planning and delivering projects on time",
     "project management; program management; PMP; Agile; Scrum; timeline management; "
     "cross-functional coordination; risk tracking"),
    ("Stakeholder management and communication", "Working with and presenting to the people involved",
     "stakeholder management; stakeholder communication; executive communication; presentation skills; "
     "cross-functional collaboration; client communication; storytelling"),
    ("Leadership and people management", "Leading and developing a team",
     "team leadership; people management; coaching; mentoring; hiring; performance management"),
    ("Energy and oil and gas industry", "Experience in energy companies and markets",
     "oil and gas industry; energy industry knowledge; downstream operations; upstream operations; "
     "fuel distribution; renewable energy; energy transition; utilities"),
    ("Healthcare industry", "Experience in healthcare providers, payers or life sciences",
     "healthcare data; healthcare operations; clinical operations; payer; provider; "
     "electronic health records; pharmaceuticals; medical devices"),
    ("Financial services industry", "Experience in banks, insurers and investment firms",
     "banking; capital markets; asset management; insurance; fintech; wealth management; payments"),
    ("Technology industry", "Experience in software and internet companies",
     "SaaS; B2B software; cloud software; platforms; technology industry; startups"),
    ("Consumer goods and retail", "Experience in consumer brands and stores",
     "CPG; consumer goods; retail; e-commerce; merchandising; category management; trade marketing"),
    ("Cloud and DevOps", "Running software on cloud platforms",
     "AWS; Amazon Web Services; Azure; Google Cloud Platform; Docker; Kubernetes; MLOps; CI/CD"),
    ("Programming languages", "Writing software",
     "Python; Java; JavaScript; C++; SQL; R; Rust; TypeScript"),
    ("Enterprise systems", "Business software for finance, HR and customers",
     "SAP; Oracle; Workday; Salesforce; NetSuite; Microsoft Dynamics; ERP systems"),
    ("Risk and compliance", "Keeping a business within rules and limits",
     "risk management; regulatory compliance; internal controls; SOX; AML; KYC; credit risk"),
    ("Human resources", "Hiring and supporting employees",
     "talent acquisition; recruiting; HR analytics; compensation and benefits; employee relations; "
     "organizational development"),
    ("Entrepreneurship", "Starting and funding new ventures",
     "entrepreneurship; startup experience; fundraising; business plan; pitch decks; venture building"),
    ("Sustainability and ESG", "Measuring and improving environmental and social impact",
     "ESG; sustainability reporting; carbon accounting; impact measurement; sustainability strategy"),
    ("Office productivity", "Everyday office software",
     "Excel; PowerPoint; Word; Google Sheets; Google Slides; Microsoft Office; spreadsheets"),
]


def onet_entries() -> list[tuple[str, str, str]]:
    counts, hot = defaultdict(Counter), defaultdict(set)
    with ONET.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat, tool = row["Element Name"].strip(), row["Workplace Example"].strip()
            counts[cat][tool] += 1
            if row["Hot Technology"] == "Y":
                hot[cat].add(tool)
    out = []
    for cat in sorted(counts):
        tools = sorted(counts[cat], key=lambda t: (t not in hot[cat], -counts[cat][t], t))
        out.append((cat, f"Software in the O*NET category '{cat}'",
                    "; ".join(tools[:TOOLS_PER_CATEGORY])))
    return out


def main(embed: bool) -> None:
    rows = [("hand", *e) for e in HAND] + [("onet", *e) for e in onet_entries()]
    with KB_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "family", "description", "skills", "source"])
        for i, (source, family, description, skills) in enumerate(rows, 1):
            w.writerow([f"K{i}", family, description, skills, source])
    print(f"wrote {len(rows)} entries ({len(HAND)} hand-made, {len(rows) - len(HAND)} O*NET) to {KB_CSV}")
    if embed:
        sys.path.insert(0, str(ROOT / "eval"))
        import env  # noqa: F401  copies .streamlit/secrets.toml into the environment
        from core.config import get_settings
        from core.embeddings import embed_with_wait
        from core.knowledge import entry_text
        from core.llm import build_embedder
        emb = build_embedder(get_settings())
        texts = [entry_text(e) for e in load_kb()]
        import time
        vecs = []
        for i in range(0, len(texts), 40):  # Gemini's free tier embeds about 100 texts per minute
            if i:
                time.sleep(35)
            vecs.append(embed_with_wait(emb.embed, texts[i:i + 40], lambda s: print(f"waiting {s}s")))
        print("saved", save_vectors(emb.name, texts, __import__("numpy").vstack(vecs)))


if __name__ == "__main__":
    main("--embed" in sys.argv)
