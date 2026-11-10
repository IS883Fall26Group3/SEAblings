"""Retrieval for per-requirement labels (core/requirements.py, core/knowledge.py). No API keys:
embeddings are a bag of words and the LLM is a fake that returns fixed JSON."""
import json

from core.config import get_settings
from core.job_sources.sample import SampleSource
from core.knowledge import KBEntry, SkillKB, entry_text, load_kb
from core.llm.base import LLMProvider
from core.matcher import score_jobs
from core.requirements import cv_lines, quote_in_cv, retrieve_evidence, to_checks
from core.schemas import CandidateProfile, JobRequirements, RequirementCheck, RequirementLabel
from core.skill_extractor import PROMPTS, label_requirements
from test_matcher import FakeEmbedder, fake_embed

CV = """ALEX RIVERA
EXPERIENCE
- Wrote SQL queries to consolidate sales and supply data from SAP
• Automated monthly KPI dashboards in Tableau and Excel
Managed inventory for three warehouses
SKILLS
SQL, Excel, Tableau"""

# The fake embedding splits on spaces, so "inventory" goes last (no comma after it in entry_text).
KB = [KBEntry("K1", "Data analysis and business intelligence", "", ("tableau", "power bi", "looker"), "hand"),
      KBEntry("K2", "Supply chain", "", ("logistics", "procurement", "inventory"), "hand")]


def kb():
    return SkillKB(KB, fake_embed([entry_text(e) for e in KB]))


def job_with(skills):
    job = SampleSource().search("analyst", "", 1)[0]
    job.requirements = JobRequirements(required_skills=skills, nice_to_have_skills=[])
    return job


def test_cv_lines_drops_bullets_headings_and_repeats():
    lines = cv_lines(CV + "\nSQL, Excel, Tableau")
    assert "EXPERIENCE" not in lines and "SKILLS" not in lines
    assert lines[0] == "Wrote SQL queries to consolidate sales and supply data from SAP"  # name line dropped too
    assert lines.count("SQL, Excel, Tableau") == 1
    long = cv_lines(" ".join(["word"] * 50) + ". Second sentence here.", max_words=40)
    assert all(len(x.split()) <= 40 for x in long) and long[-1] == "Second sentence here."


def test_quote_check_is_word_for_word():
    assert quote_in_cv("wrote SQL queries to  consolidate sales", CV)       # case and spacing ignored
    assert quote_in_cv("Automated monthly KPI dashboards in Tableau", CV)
    assert not quote_in_cv("Wrote complex SQL queries", CV)                 # a paraphrase fails
    assert not quote_in_cv("in", CV)                                        # too short to prove anything


def test_retrieval_finds_the_closest_cv_lines_and_knowledge_base_family():
    job = job_with(["power bi", "inventory"])
    got = retrieve_evidence([job], CV, fake_embed, kb(), k_lines=1, k_kb=1)[job.id]
    assert [r.requirement for r in got] == ["power bi", "inventory"]
    assert got[0].related[0].id == "K1" and got[1].related[0].id == "K2"
    assert got[1].lines == ["Managed inventory for three warehouses"]


def test_retrieval_always_includes_lines_that_name_the_requirement_or_a_related_skill():
    cv = "Led ten projects for clients\nTools: Excel, PowerPoint, Python, R, Tableau\nR&D budget review"
    job = job_with(["Python / SQL", "power bi"])
    # The fake embedding reads "Python," (with the comma) as another word, so only the word match finds it.
    got = retrieve_evidence([job], cv, fake_embed, kb(), k_lines=1, k_kb=1)[job.id]
    tools = "Tools: Excel, PowerPoint, Python, R, Tableau"
    assert got[0].lines[0] == tools           # names Python (one part of "Python / SQL")
    assert got[1].lines[0] == tools           # names Tableau, related to Power BI in the knowledge base
    assert len(got[0].lines) <= 2             # word matches, then the k_lines=1 closest by embedding


def test_checks_keep_only_verified_quotes_and_known_statuses():
    retrieved = retrieve_evidence([job := job_with(["sql", "power bi", "python", "rust"])], CV, fake_embed, kb())
    labels = [RequirementLabel(requirement="SQL", status="met", quote="Wrote SQL queries"),
              RequirementLabel(requirement="power bi", status="Partial", quote="Built Power BI models"),
              RequirementLabel(requirement="python", status="missing", quote="anything"),
              RequirementLabel(requirement="rust", status="maybe")]
    checks = {c.requirement: c for c in to_checks(retrieved[job.id], labels, CV)}
    assert checks["sql"].quote_verified and checks["sql"].quote == "Wrote SQL queries"
    assert checks["power bi"].status == "partial" and not checks["power bi"].quote_verified  # not in the CV
    assert checks["python"].quote == ""
    assert "rust" not in checks  # unknown status: the matcher decides


def test_missing_label_keeps_a_retrieved_line_that_names_it():
    retrieved = retrieve_evidence([job := job_with(["tableau", "power bi", "sql"])], CV, fake_embed, kb())
    labels = [RequirementLabel(requirement="tableau", status="missing"),
              RequirementLabel(requirement="power bi", status="missing"),
              RequirementLabel(requirement="sql", status="met", quote="Wrote SQL queries")]
    checks = {c.requirement: c for c in to_checks(retrieved[job.id], labels, CV)}
    assert "Tableau" in checks["tableau"].mention  # the LLM said missing, but a line names it
    assert checks["power bi"].mention == ""        # no line names it: a gap as far as the CV shows
    assert checks["sql"].mention == ""             # only missing requirements get a mention


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, *, json_mode, temperature, what, schema=None):
        self.prompts.append(prompt)
        return json.dumps(self.reply)


def test_labeling_prompt_holds_retrieved_lines_not_the_whole_cv():
    job = job_with(["inventory"])
    retrieved = retrieve_evidence([job], CV, fake_embed, kb(), k_lines=1)
    llm = FakeLLM({"items": [{"job_id": job.id, "labels": [
        {"requirement": "inventory", "reason": "", "status": "met", "quote": "Managed inventory"}]}]})
    checks = label_requirements(llm, [job], retrieved, CV)
    assert checks[job.id][0].quote_verified
    prompt = llm.prompts[0]
    assert '"Managed inventory for three warehouses"' in prompt and "Supply chain" in prompt
    assert "Wrote SQL queries" not in prompt  # not retrieved for this requirement, so not sent


def test_labels_decide_skill_coverage_and_partial_counts_half():
    profile = CandidateProfile(summary="analyst", years_experience=3, seniority="mid",
                               skills=["SQL"], job_titles=[])
    job = job_with(["SQL", "Power BI", "Python", "Rust"])
    checks = {job.id: [RequirementCheck(requirement="power bi", status="partial"),
                       RequirementCheck(requirement="sql", status="met", quote="Wrote SQL queries",
                                        quote_verified=True),
                       RequirementCheck(requirement="python", status="missing")]}
    r = score_jobs(profile, CV, [job], FakeEmbedder(), get_settings(), requirement_checks=checks)[0]
    assert r.matched_skills == ["SQL"] and r.partial_skills == ["Power BI"]
    assert r.missing_skills == ["Python", "Rust"]  # Rust wasn't labeled: the matcher found it missing
    assert r.skill_coverage == (1 + 0.5) / 4
    plain = score_jobs(profile, CV, [job], FakeEmbedder(), get_settings())[0]
    assert plain.skill_coverage == 1 / 4 and not plain.partial_skills


def test_knowledge_base_file_loads():
    entries = load_kb()
    assert len(entries) > 100 and {e.source for e in entries} == {"hand", "onet"}
    bi = next(e for e in entries if e.family == "Data analysis and business intelligence")
    assert "Tableau" in bi.skills and "Power BI" in bi.skills


def test_label_only_variant_asks_for_no_reason_and_still_verifies_quotes():
    """The chain-of-thought comparison (eval/requirement_accuracy.py --compare): same rules, examples and
    retrieved lines; only the reason before each label is dropped."""
    job = job_with(["inventory"])
    retrieved = retrieve_evidence([job], CV, fake_embed, kb(), k_lines=1)
    llm = FakeLLM({"items": [{"job_id": job.id, "labels": [
        {"requirement": "inventory", "status": "met", "quote": "Managed inventory"}]}]})
    checks = label_requirements(llm, [job], retrieved, CV, variant="label_only")
    assert checks[job.id][0].status == "met" and checks[job.id][0].quote_verified
    assert checks[job.id][0].reason == ""
    reason_first = (PROMPTS / "label_requirements.md").read_text(encoding="utf-8")
    label_only = llm.prompts[0]
    assert "1. reason:" in reason_first and "1. reason:" not in label_only
    assert '-> reason "' not in label_only and "Don't explain your choice." in label_only
    assert '"Managed inventory for three warehouses"' in label_only
