"""The PDF report (ui/report.py): it builds, holds the page's content and survives odd characters."""
import io

from pypdf import PdfReader

from core.schemas import RequirementCheck
from test_score_notes import JOB, PROFILE, result
from ui.report import build_pdf, latin1


def pdf_text(data: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages)


def test_report_has_the_profile_jobs_breakdown_and_advice():
    r = result(score=72, skill_coverage=0.5, semantic_similarity=0.4, seniority_fit=1,
               matched_skills=["SQL"], partial_skills=["Dashboards"], missing_skills=["Python"],
               field_match="different", work_match="same", embedding_similarity=0.6,
               requirement_checks=[RequirementCheck(requirement="Dashboards", status="partial",
                                                    reason="Related BI tools.")])
    data = build_pdf([r], PROFILE.model_copy(update={"summary": "Analyst with 8 years’ experience — retail."}),
                     summary="Matched 1 job", models="gemini-flash-lite-latest", embedder="gemini-embedding-001",
                     explanations={JOB.id: "### What to do\n- Learn **Python** → pandas\n1. Build a demo"})
    assert data[:4] == b"%PDF"
    text = pdf_text(data)
    for expected in ("Data Analyst", "72/100", "Skill coverage: 50%", "Profile similarity: 40%",
                     "Experience fit: 100%", "Missing", "Met", "Advice for this role", "What to do",
                     "Learn Python -> pandas", "check important decisions with a career advisor", "1. Build a demo", "Analyst with 8 years' experience - retail"):
        assert expected in text, expected


def test_report_without_advice_and_with_characters_helvetica_lacks():
    data = build_pdf([result(score=10)], PROFILE, summary="Matched 1 job", models="m", embedder="e")
    assert "Advice for this role" not in pdf_text(data)
    assert latin1("Café “ok” 東京") == 'Café "ok" ??'
