"""Score explanations: every text case, built from results the matcher would record. No API keys."""
import numpy as np

from core.config import get_settings
from core.job_sources.sample import SampleSource
from core.matcher import background_factor, match_skills_detail, score_jobs
from core.schemas import CandidateProfile, Job, JobRequirements, MatchResult, RequirementCheck, Role
from core.score_notes import Group, evidence_groups, experience_note, role_line, similarity_note, skill_note
from test_matcher import FakeEmbedder

JOB = Job(id="j1", title="Data Analyst", company="ATC", location="", description="",
          requirements=JobRequirements(required_skills=[], nice_to_have_skills=[], field="Technology consulting",
                                       work_type="Data analysis", level="entry"))
PROFILE = CandidateProfile(summary="", years_experience=8, seniority="senior", job_titles=[],
                           skills=["SQL", "Statistical Analysis"], field="Retail technology",
                           work_type="data analysis")


def result(**kw) -> MatchResult:
    base = dict(job=JOB, score=0, skill_coverage=0, semantic_similarity=0, seniority_fit=0,
                matched_skills=[], missing_skills=[], total_years=8)
    return MatchResult(**(base | kw))


# ---------- skill coverage ----------

SKILLS_LINE = "Software: SAP, Salesforce, Tableau, Power BI, MS Excel (modeling and optimization), Google Analytics, Canva"


def checks_result(*checks):
    return result(matched_skills=["Tableau"], partial_skills=["Dashboards"], missing_skills=["Python", "Excel"],
                  requirement_checks=list(checks))


def test_met_quotes_are_hidden():
    r = checks_result(RequirementCheck(requirement="Tableau", status="met", quote=SKILLS_LINE, quote_verified=True))
    assert evidence_groups(r) == []


def test_partly_met_shows_the_reason_and_a_short_quote():
    r = checks_result(RequirementCheck(requirement="Dashboards", status="partial", quote=SKILLS_LINE,
                                       quote_verified=True, reason="Lists BI tools, but no dashboard built."))
    assert evidence_groups(r) == [Group("Partly met:", ['Dashboards: Lists BI tools, but no dashboard built. '
                                                        'Your CV: "Software: SAP, Salesforce, Tableau, Power BI, '
                                                        'MS Excel (modeling and optimization), Google Analytics, '
                                                        'Canva"'])]
    long = RequirementCheck(requirement="Data checks", status="partial", reason="Forecasting touches it.",
                            quote=" ".join(f"w{i}" for i in range(20)), quote_verified=True)
    assert evidence_groups(checks_result(long))[0].items[0].endswith('w13 w14…"')


def test_partly_met_with_an_unverified_quote_shows_only_the_reason():
    r = checks_result(RequirementCheck(requirement="Dashboards", status="partial", quote="made up",
                                       reason="Related BI tools."))
    assert evidence_groups(r) == [Group("Partly met:", ["Dashboards: Related BI tools."])]


def test_missing_skills_are_listed_on_one_line():
    r = checks_result(RequirementCheck(requirement="Python", status="missing"),
                      RequirementCheck(requirement="Rust", status="missing"))
    assert evidence_groups(r) == [Group("Not found in your CV:", ["Python, Rust"], inline=True)]


def test_each_group_is_capped():
    r = checks_result(*[RequirementCheck(requirement=f"S{i}", status="partial", reason="Related.") for i in range(5)])
    assert evidence_groups(r)[0].items[3:] == ["2 more."]


def test_no_labels_means_no_bullets():
    assert evidence_groups(checks_result()) == []

def test_partial_coverage_names_close_matches_and_gives_the_tip():
    r = result(matched_skills=["Data Analysis", "Power BI", "SQL"],
               missing_skills=["Microsoft Excel", "Tableau", "Critical Thinking", "Communication"],
               close_matches={"Data Analysis": "Statistical Analysis"})
    n = skill_note(r, PROFILE)
    assert n.label == "Some required skills"
    assert n.text.startswith("You have 3 of the 7 skills this job requires.")
    assert '"Data Analysis" counted because your CV lists "Statistical Analysis", a close match.' in n.text
    assert 'If you have any of the 4 under "To build", name them in your CV' in n.text


def test_most_skills_says_what_was_not_found():
    r = result(matched_skills=["SQL", "Python", "Excel"], missing_skills=["Tableau"])
    n = skill_note(r, PROFILE)
    assert n.label == "Most required skills"
    assert n.text == ('You have 3 of the 4 skills this job requires. The one under "To build" '
                      "wasn't found in your CV.")


def test_skill_edge_cases():
    assert skill_note(result(matched_skills=["SQL", "Python"]), PROFILE).text == \
        "You have all 2 skills this job requires."
    none = skill_note(result(missing_skills=["A", "B"]), PROFILE)
    assert none.label == "No required skills found" and none.text.startswith("None of the 2 skills")
    assert "neutral 50%" in skill_note(result(), PROFILE).text
    empty_cv = PROFILE.model_copy(update={"skills": []})
    assert "couldn't read any skills from your CV" in skill_note(result(missing_skills=["A"]), empty_cv).text


# ---------- profile similarity ----------

def test_similarity_labels_by_score():
    labels = {p: similarity_note(result(semantic_similarity=p / 100), PROFILE).label for p in (100, 80, 50, 10, 0)}
    assert labels == {100: "Very strong match", 80: "Strong match", 50: "Partial match", 10: "Weak match",
                      0: "No clear match"}
    assert similarity_note(result(semantic_similarity=1), PROFILE).text == \
        "Your CV and this posting describe very similar work."


def test_similarity_table_shows_rows_both_sides_have():
    rows = similarity_note(result(semantic_similarity=0.5), PROFILE).table
    assert rows == [("Field", "Retail technology", "Technology consulting"),
                    ("Type of work", "Data analysis", "Data analysis"), ("Level", "Senior", "Entry")]
    bare = PROFILE.model_copy(update={"field": "", "work_type": ""})
    assert similarity_note(result(), bare).table == []  # level alone isn't shown


def test_rating_that_lowers_the_score_is_explained():
    r = result(semantic_similarity=0, embedding_similarity=0.36, field_match="different", work_match="different")
    n = similarity_note(r, PROFILE)
    assert n.label == "No clear match"
    assert n.text == ("The wording of your CV and this posting overlaps (36%), but the field and type of "
                      "work are rated Missing, so this counts as 0%.")
    assert n.table[0] == ("Field", "Retail technology", "Technology consulting", "Missing")
    assert n.table[2] == ("Level", "Senior", "Entry", "")  # level isn't rated


def test_rating_that_keeps_the_score_uses_the_usual_sentence():
    r = result(semantic_similarity=1, embedding_similarity=1, field_match="same", work_match="same")
    assert similarity_note(r, PROFILE).text == "Your CV and this posting describe very similar work."
    assert similarity_note(r, PROFILE).table[0][3] == "Met"  # "same" is shown this way
    mixed = result(semantic_similarity=0.75, embedding_similarity=1, field_match="same", work_match="related")
    assert "but the type of work is rated Partial, so this counts as 75%." in similarity_note(mixed, PROFILE).text


# ---------- experience fit ----------

DATA = Role(title="Data Analyst", start="2023-01", end="present")
CHEF = Role(title="Pastry Chef", duration_months=11)


def test_partly_relevant_lists_the_roles_that_counted():
    r = result(relevant_only=True, years_counted=3, min_years=5, seniority_fit=0.6, counted_roles=[DATA])
    n = experience_note(r)
    assert n.label == "Partly meets the requirement"
    assert n.text == "This job asks for 5+ years; 3 of your 8 years count as relevant:"
    assert n.bullets == ["Data Analyst (2023–present)"]
    assert n.after == "Your other roles weren't counted as experience for this job."


def test_meets_with_all_years_relevant():
    r = result(relevant_only=True, years_counted=8, min_years=8, seniority_fit=1, counted_roles=[DATA])
    n = experience_note(r)
    assert n.label == "Meets the requirement" and n.after == ""
    assert n.text == "This job asks for 8+ years; we counted 8 years of relevant experience:"


def test_no_relevant_roles():
    with_min = experience_note(result(relevant_only=True, min_years=3))
    assert with_min.label == "No relevant experience"
    assert with_min.text == "This job asks for 3+ years. None of your roles were counted as experience for this job."
    no_min = experience_note(result(relevant_only=True))
    assert no_min.text == ("This posting doesn't state a minimum, but none of your roles were counted as "
                           "experience for this job.")


def test_no_minimum_with_relevant_experience_and_total_years():
    full = experience_note(result(relevant_only=True, years_counted=2, seniority_fit=1, counted_roles=[DATA]))
    assert full.text.startswith("This posting doesn't state a minimum; you have relevant experience")
    total = experience_note(result(years_counted=8, min_years=10, seniority_fit=0.8))
    assert total.label == "Partly meets the requirement"
    assert total.text == "This job asks for 10+ years; we counted all 8 years on your CV."
    assert "any experience counts" in experience_note(result(years_counted=8, seniority_fit=1)).text


def test_role_lines():
    assert role_line(Role(title="Analyst", start="2020-01", end="2023-06")) == "Analyst (2020–2023)"
    assert role_line(CHEF) == "Pastry Chef (11 months)"
    assert role_line(Role(title="Lead", duration_months=32)) == "Lead (2.7 years)"
    assert role_line(Role(title="Intern")) == "Intern"


# ---------- what the matcher records for them ----------

def test_matcher_records_close_matches_and_counted_roles():
    vec = {"sql": np.array([1.0, 0]), "structured query language": np.array([0.99, 0.141]),
           "tableau": np.array([0, 1.0])}
    matched, missing, close = match_skills_detail(["SQL"], ["Structured Query Language", "Tableau"], vec, 0.9)
    assert matched == ["Structured Query Language"] and missing == ["Tableau"]
    assert close == {"Structured Query Language": "SQL"}

    [job] = SampleSource().search("analyst", "", 1)
    job.requirements = JobRequirements(required_skills=["sql"], nice_to_have_skills=[], min_years_experience=4)
    profile = CandidateProfile(summary="", years_experience=10, seniority="senior", skills=["sql"], job_titles=[])
    two_years = [Role(title="Analyst", start="2020-01", end="2021-12")]
    [r] = score_jobs(profile, "cv", [job], FakeEmbedder(), get_settings(), relevant_roles={job.id: two_years})
    assert r.relevant_only and r.counted_roles == two_years
    assert (r.years_counted, r.total_years, r.min_years, r.seniority_fit) == (2.0, 10, 4, 0.5)


def test_background_factor_and_its_effect_on_the_score():
    assert background_factor("same", "same") == 1.0
    assert background_factor("different", "different") == 0.0
    assert background_factor("", "") == 1.0  # not rated: no evidence, no change
    assert background_factor("Different", " related ", {"same": 1, "related": 0.5, "different": 0}, "max") == 0.5
    assert background_factor("same", "different", {"same": 1, "related": 1, "different": 0}, "min") == 0.0

    [job] = SampleSource().search("analyst", "", 1)
    job.requirements = JobRequirements(required_skills=[], nice_to_have_skills=[])
    profile = CandidateProfile(summary="", years_experience=1, seniority="mid", skills=[], job_titles=[])
    plain = score_jobs(profile, job.description, [job], FakeEmbedder(), get_settings())[0]
    rated = score_jobs(profile, job.description, [job], FakeEmbedder(), get_settings(),
                       background={job.id: ("different", "different")})[0]
    assert plain.semantic_similarity == plain.embedding_similarity > 0
    assert rated.semantic_similarity == 0 and rated.embedding_similarity == plain.embedding_similarity
    assert (rated.field_match, rated.work_match) == ("different", "different")
    assert rated.score < plain.score


def test_card_markdown_escapes_cv_text():
    from ui.components import note_markdown
    from core.score_notes import Note
    md = note_markdown(Note("Strong match", "Grew sales to $11M", table=[("Field", "Oil | gas", "Energy")]))
    assert "\\$11M" in md and "Oil \\| gas" in md
