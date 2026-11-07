import json
from pathlib import Path

import numpy as np

from core.advice import (AdviceReply, ListedRequirement, RequirementRating, build_advice, chunk_cv, generate_advice,
                         list_requirements, retrieve_evidence)
from core.llm.base import LLMError, LLMProvider
from core.schemas import CandidateProfile, Job, MatchResult, Role

ROOT = Path(__file__).resolve().parents[1]
CV = (ROOT / "data" / "sample" / "sample_cv.txt").read_text(encoding="utf-8")
ROLES = [Role(title="Corporate Planning Analyst"), Role(title="Business Analyst Intern")]
PLANNING = "[Corporate Planning Analyst, Example Energy Co. (2021 - 2025)]"
# A dummy resume laid out like a real PDF: no blank lines, long role lines, wrapped bullets
JOHN = (ROOT / "tests" / "data" / "john_doe_resume.txt").read_text(encoding="utf-8")
JOHN_ROLES = [Role(title=t) for t in ("Senior Data Analyst", "Software Engineer / Data Analyst", "Software Engineer",
                                      "Junior Software Engineer")]


def reqs(*texts, chip="", kind="work"):
    return [ListedRequirement(text=t, chip=chip, kind=kind) for t in texts]


def rate(id, rating, quote=""):
    return RequirementRating(id=id, rating=rating, quote=quote)


def result(missing=(), field_match="", work_match=""):
    job = Job(id="j1", title="Data Analyst", company="Acme", location="", description="Needs SQL and a team lead.")
    return MatchResult(job=job, score=50, skill_coverage=0.5, semantic_similarity=0.5, seniority_fit=0.5,
                       matched_skills=[], missing_skills=list(missing), field_match=field_match, work_match=work_match)


def profile(roles=None):
    return CandidateProfile(summary="", years_experience=4, seniority="mid", skills=[], job_titles=[], roles=roles or [])


class FakeLLM(LLMProvider):
    def __init__(self, listed, reply=None):
        self.listed, self.reply, self.prompts = listed, reply, []

    def complete(self, prompt, *, json_mode, temperature, what, schema=None):
        self.prompts.append(prompt)
        return json.dumps(self.listed if what == "listing the job's requirements" else self.reply)


def by_words(texts):
    """Cosine 1 between texts sharing a keyword, 0 otherwise."""
    keys = ["sql", "power bi", "market research"]
    vecs = [[1.0 if k in t.lower() else 0.0 for k in keys] + [0.01] for t in texts]
    return np.array(vecs, dtype=np.float32) / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------- chunking ----------
def test_chunks_keep_whole_lines_and_do_not_mix_roles_or_sections():
    chunks = chunk_cv(CV, ROLES)
    intern = next(c for c in chunks if "Conducted market research" in c)
    assert intern.startswith("[Business Analyst Intern, Sample Consulting (2020)] ")
    assert "SQL" not in intern and "MBA" not in intern
    assert not any("EDUCATION" in c and "Wrote" in c for c in chunks)
    assert not any(c.endswith("Wrote SQL queries") or c.endswith("Built") for c in chunks)


def test_every_chunk_of_a_split_role_names_the_role_and_repeats_one_line():
    lines = [f"- Bullet {name} " + "word " * 7 for name in "ABCDE"]  # 10 words each
    cv = "Data Analyst, Acme (2020 - 2024)\n" + "\n".join(lines)
    chunks = chunk_cv(cv, [Role(title="Data Analyst")])
    assert len(chunks) == 2
    assert all(c.startswith("[Data Analyst, Acme (2020 - 2024)] ") for c in chunks)
    assert "Bullet D" in chunks[0] and "Bullet D" in chunks[1]  # one line of overlap
    assert "Bullet E" not in chunks[0] and "Bullet C" not in chunks[1]


def test_a_role_that_fits_is_not_split():
    chunks = [c for c in chunk_cv(CV, ROLES) if c.startswith(PLANNING)]
    assert len(chunks) == 1
    assert "Built financial models" in chunks[0] and "Wrote SQL queries" in chunks[0]


def test_word_limit_does_not_count_the_role_line():
    for chunk in chunk_cv(CV, ROLES):
        assert len(chunk.split("] ")[-1].split()) <= 40


def test_without_roles_it_falls_back_to_plain_lines():
    chunks = chunk_cv(CV)
    assert any("Conducted market research" in c for c in chunks)
    assert not any(c.startswith("[") for c in chunks)


def test_a_cv_with_no_line_breaks_still_gets_40_word_chunks():
    chunks = chunk_cv(" ".join(CV.split()))
    assert len(chunks) > 1
    assert all(len(c.split()) <= 40 for c in chunks)


def test_a_role_with_no_lines_of_its_own_is_kept():
    assert "[Analyst, Acme (2020)]" in chunk_cv("Analyst, Acme (2020)\n\nEDUCATION\nMBA", [Role(title="Analyst")])


def test_pdf_layout_keeps_each_bullet_with_its_own_employer():
    chunks = chunk_cv(JOHN, JOHN_ROLES)
    for bullet, employer in (("mentored two junior analysts", "NORTHSTAR"), ("code reviews.", "BLUEWAVE"),
                             ("stored procedures", "BLUEWAVE"), ("defect resolution", "INNOVATECH")):
        found = [c for c in chunks if bullet in c]
        assert found and all(c.startswith(f"[{employer} ") for c in found), bullet
    bullets = [c for c in chunks if "•" in c]
    assert len(bullets) >= 7 and all(c.startswith("[") for c in bullets)


def test_wrapped_lines_are_rejoined():
    chunks = chunk_cv(JOHN, JOHN_ROLES)
    assert any("operational issues, and performance opportunities." in c for c in chunks)
    assert not any(c.startswith(("performance opportunities", "requirements.", "discrepancies.")) for c in chunks)


def test_a_summary_sentence_naming_a_job_title_is_not_a_role():
    summary = next(c for c in chunk_cv(JOHN, JOHN_ROLES) if "former software engineer" in c)
    assert summary.startswith("PROFESSIONAL SUMMARY")


# ---------- requirements and retrieval ----------
def test_every_chip_gets_checked_even_if_the_llm_skips_it():
    llm = FakeLLM({"items": [{"text": "Write SQL", "chip": "sql"}, {"text": "Lead a team", "chip": ""}]})
    items = list_requirements(llm, result(["SQL", "Tableau"]))
    assert {i.chip for i in items} == {"SQL", "Tableau", ""}
    assert ListedRequirement(text="Tableau", chip="Tableau") in items


def test_a_chip_the_llm_made_up_is_dropped():
    llm = FakeLLM({"items": [{"text": "Write SQL", "chip": "Invented"}]})
    items = list_requirements(llm, result(["Tableau"]))
    assert next(i for i in items if i.text == "Write SQL").chip == ""
    assert {i.chip for i in items} == {"", "Tableau"}  # the real chip still gets its own check


def test_the_requirement_list_is_capped_but_keeps_all_chips():
    listed = {"items": [{"text": f"Req {i}"} for i in range(30)]}
    chips = [f"Chip {i}" for i in range(14)]
    items = list_requirements(FakeLLM(listed), result(chips))
    assert len(items) == 15
    assert {i.chip for i in items if i.chip} == set(chips)


def test_an_unknown_kind_counts_as_work():
    llm = FakeLLM({"items": [{"text": "Valid licence", "kind": "BASIC"}, {"text": "Write SQL", "kind": "duty"}]})
    assert [i.kind for i in list_requirements(llm, result())] == ["basic", "work"]


def test_retrieval_returns_the_closest_chunks_first():
    chunks = ["Built Power BI reports", "Wrote SQL queries", "Did market research"]
    found = retrieve_evidence(["SQL experience", "market research"], chunks, by_words, top_k=1)
    assert found == [["Wrote SQL queries"], ["Did market research"]]


# ---------- the three bullets ----------
def test_three_bullets_use_the_requirement_and_the_quote_from_the_cv():
    items = reqs("Write SQL", "Lead a team")
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries"), rate(2, "not_shown")],
                        match_id=1, gap_id=2, next_step="Add an example of leading people.")
    text = build_advice(items, reply, CV)
    assert '**Strongest match:** Write SQL. Your CV shows: "Wrote SQL queries"' in text
    assert "**Main gap:** Lead a team. Your CV doesn't show this." in text
    assert "**Most useful next step:** Add an example of leading people." in text


def test_a_made_up_quote_is_never_shown_as_a_match():
    reply = AdviceReply(ratings=[rate(1, "strong", "Built compilers in Rust for 10 years")], match_id=1)
    text = build_advice(reqs("Write Rust"), reply, CV)
    assert "Rust" not in text
    assert "No strong match found" in text


def test_quote_check_ignores_line_wraps_case_and_a_leading_bullet():
    reply = AdviceReply(ratings=[rate(1, "strong", "- wrote  SQL\nqueries")], match_id=1)
    assert "Your CV shows" in build_advice(reqs("Write SQL"), reply, CV)


def test_a_wrong_pick_by_the_llm_is_replaced():
    items = reqs("Write SQL", "Lead a team")
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries"), rate(2, "not_shown")],
                        match_id=2, gap_id=1, next_step="Advice about the wrong gap.")
    text = build_advice(items, reply, CV)
    assert "**Strongest match:** Write SQL" in text
    assert "**Main gap:** Lead a team" in text
    assert "wrong gap" not in text  # that step was written for a different gap
    assert "start building it: Lead a team." in text


def test_nothing_to_show_says_so_instead_of_inventing():
    text = build_advice(reqs("Write SQL"), AdviceReply(ratings=[]), CV)
    assert "No strong match found" in text and "No clear gap found" in text


def test_a_weak_link_is_worded_as_weak():
    reply = AdviceReply(ratings=[rate(1, "weak", "Python (basic)")], gap_id=1, next_step="Do a Python project.")
    text = build_advice(reqs("Python for analysis"), reply, CV)
    assert 'Your CV only shows a weak link: "Python (basic)"' in text
    assert "Do a Python project." in text


def test_dollar_signs_can_not_turn_into_a_math_formula():
    reply = AdviceReply(ratings=[rate(1, "strong", "$2B downstream investment portfolio")], match_id=1,
                        next_step="Aim for $5B.")
    text = build_advice(reqs("Manage budgets"), reply, CV)
    assert r"\$2B" in text and "$2B" not in text.replace(r"\$", "")


def test_requirement_punctuation_does_not_double_up():
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries")], match_id=1)
    text = build_advice(reqs("Write SQL for reports."), reply, CV)
    assert "Write SQL for reports. Your CV" in text and ".." not in text


def test_long_quotes_are_shortened_on_the_card():
    quote = "Built financial models and scenario analyses for a $2B downstream investment portfolio"
    long_cv = f"EXPERIENCE\n- {quote} {quote}"
    reply = AdviceReply(ratings=[rate(1, "strong", f"{quote} {quote}")], match_id=1)
    text = build_advice(reqs("Build models"), reply, long_cv)
    assert "…" in text and text.count("financial models") == 2


# ---------- real work as evidence (fix 3) ----------
def test_a_skills_list_alone_can_not_be_a_strong_match():
    reply = AdviceReply(ratings=[rate(1, "strong", "Python (basic)")], match_id=1)
    assert "No strong match found" in build_advice(reqs("Python for analysis"), reply, CV, ROLES)


def test_a_summary_alone_can_not_be_a_strong_match():
    reply = AdviceReply(ratings=[rate(1, "strong", "Skilled in SQL, Python, Power BI")], match_id=1)
    assert "No strong match found" in build_advice(reqs("Write SQL"), reply, JOHN, JOHN_ROLES)


def test_work_from_a_job_or_project_stays_strong():
    for quote in ("created REST APIs, database queries, and stored procedures",
                  "Used Python and SQL to analyze transaction data"):
        reply = AdviceReply(ratings=[rate(1, "strong", quote)], match_id=1)
        assert "Your CV shows" in build_advice(reqs("Build APIs"), reply, JOHN, JOHN_ROLES)


def test_a_degree_can_be_a_strong_match():
    reply = AdviceReply(ratings=[rate(1, "strong", "Bachelor of Science in Computer Science")], match_id=1)
    assert "Your CV shows" in build_advice(reqs("Degree in Computer Science"), reply, JOHN, JOHN_ROLES)


def test_a_cv_without_headings_keeps_its_ratings():
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries daily")], match_id=1)
    assert "Your CV shows" in build_advice(reqs("Write SQL"), reply, "Wrote SQL queries daily at Acme.")


def test_a_lowered_rating_does_not_become_the_main_gap():
    reply = AdviceReply(ratings=[rate(1, "strong", "Skilled in SQL, Python, Power BI")], gap_id=1)
    assert "No clear gap found" in build_advice(reqs("Write SQL"), reply, JOHN, JOHN_ROLES)


def test_years_are_left_to_the_cards_experience_fit():
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries"), rate(2, "not_shown")], match_id=1, gap_id=2)
    text = build_advice(reqs("Minimum of seven years of experience", "At least 5 years of relevant work"), reply, CV)
    assert "No strong match found" in text and "No clear gap found" in text


# ---------- option C: "To build" chips ----------
def test_a_chip_the_cv_contradicts_gets_its_own_bullet_between_the_gap_and_the_step():
    items = reqs("Build dashboards") + reqs("Write queries for reporting", chip="Reporting") + reqs("Lead a team")
    reply = AdviceReply(ratings=[rate(1, "strong", "Power BI and Excel"), rate(2, "strong", "Wrote SQL queries"),
                                 rate(3, "not_shown")], match_id=1, gap_id=3, next_step="Practise leading a team.")
    lines = build_advice(items, reply, CV).splitlines()
    assert lines[0].startswith("- **Strongest match:**")
    assert lines[1].startswith("- **Main gap:**")
    assert lines[2] == '- **May not be a real gap:** Reporting is listed under "To build", but your CV shows "Wrote SQL queries"'
    assert lines[3] == "- **Most useful next step:** Practise leading a team."


def test_a_chip_already_shown_as_the_strongest_match_is_not_repeated():
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries")], match_id=1)
    text = build_advice(reqs("Knowledge of relational databases", chip="Relational Databases"), reply, CV)
    assert "**Strongest match:** Knowledge of relational databases" in text
    assert "May not be a real gap" not in text


def test_a_confirmed_chip_gets_no_note():
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1)
    text = build_advice(reqs("Use Tableau", chip="Tableau"), reply, CV)
    assert "**Main gap:** Use Tableau" in text
    assert "May not be a real gap" not in text


def test_a_weak_link_never_says_it_is_not_a_real_gap():
    items = reqs("Develop in JavaScript", chip="JavaScript") + reqs("Lead a team")
    reply = AdviceReply(ratings=[rate(1, "weak", "Wrote SQL queries"), rate(2, "not_shown")], gap_id=2)
    text = build_advice(items, reply, CV)
    assert "JavaScript" not in text


def test_a_chip_named_in_the_cv_can_not_be_called_not_shown():
    items = reqs("Know SAP well", chip="SAP") + reqs("Lead a team")
    reply = AdviceReply(ratings=[rate(1, "not_shown"), rate(2, "not_shown")], gap_id=2)
    text = build_advice(items, reply, CV)
    assert "**Main gap:** Lead a team" in text
    assert '**May not be a real gap:** SAP is listed under "To build", and your CV mentions it.' in text
    assert "but your CV shows" not in text


def test_chip_check_needs_a_whole_word():
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1)
    text = build_advice(reqs("Know R", chip="R"), reply, "Worked as a RESEARCHER.")
    assert "**Main gap:** Know R. Your CV doesn't show this." in text


def test_notes_are_capped_so_they_do_not_outweigh_the_bullets():
    items = [ListedRequirement(text=f"Requirement {i}", chip=f"Chip {i}") for i in range(6)]
    reply = AdviceReply(ratings=[rate(i, "strong", "Wrote SQL queries") for i in range(1, 7)], match_id=1)
    text = build_advice(items, reply, CV)  # the first is the strongest match, so five notes are left
    assert text.count('is listed under "To build"') == 3
    assert '**May not be a real gap:** 2 more "To build" items may also be covered by your CV.' in text


# ---------- next step and basic conditions ----------
def test_a_next_step_that_ignores_the_gap_is_replaced():
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1,
                        next_step="Highlight your decade of experience in your summary.")
    text = build_advice(reqs("Experience designing Oracle APEX applications"), reply, CV)
    assert "decade" not in text
    assert "start building it: Experience designing Oracle APEX applications." in text


def test_a_next_step_that_names_the_gap_stands():
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1, next_step="Build a small Oracle APEX app.")
    assert "Build a small Oracle APEX app." in build_advice(reqs("Designing Oracle APEX applications"), reply, CV)


def test_a_work_gap_beats_a_basic_condition():
    items = reqs("Valid driver's license", kind="basic") + reqs("Wash and detail vehicles")
    reply = AdviceReply(ratings=[rate(1, "not_shown"), rate(2, "not_shown")], gap_id=1, next_step="Add your license.")
    text = build_advice(items, reply, CV)
    assert "**Main gap:** Wash and detail vehicles" in text
    assert "license" not in text


def test_a_basic_condition_is_the_gap_only_if_nothing_else_is_left():
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1)
    assert "**Main gap:** Valid driver's license" in build_advice(reqs("Valid driver's license", kind="basic"),
                                                                  reply, CV)


def test_a_basic_condition_is_never_the_strongest_match():
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries")], match_id=1)
    assert "No strong match found" in build_advice(reqs("Must be 18", kind="basic"), reply, CV)


def test_poor_fit_says_to_look_elsewhere_instead_of_editing_the_cv():
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1, next_step="Add vehicle detailing to your CV.")
    text = build_advice(reqs("Wash and detail vehicles"), reply, CV, poor_fit=True)
    assert "focus on roles closer to your field" in text
    assert "Add vehicle detailing" not in text


# ---------- the two jobs tested by hand ----------
def test_car_wash_job_for_a_data_analyst():
    """An off-field job: nothing matches, and a license shouldn't be presented as the thing to fix."""
    listed = {"items": [{"text": "Valid driver's license", "chip": "Driver's License", "kind": "basic"},
                        {"text": "Must be 18 years of age or older", "kind": "basic"},
                        {"text": "Wash and detail vehicles", "chip": "Vehicle Detailing"},
                        {"text": "Perform service and maintenance checks", "chip": "Vehicle Maintenance"}]}
    reply = {"ratings": [{"id": n, "rating": "not_shown"} for n in range(1, 5)], "gap_id": 1,
             "next_step": "Add your driver's license to your CV."}
    missing = ["Driver's License", "Vehicle Detailing", "Vehicle Maintenance"]
    text = generate_advice(FakeLLM(listed, reply), JOHN, profile(JOHN_ROLES),
                           result(missing, field_match="different", work_match="different"), by_words)
    assert "No strong match found" in text
    assert "**Main gap:** Wash and detail vehicles" in text
    assert "focus on roles closer to your field" in text
    assert "license" not in text.lower()


def test_harvard_software_engineer_for_a_former_engineer():
    """A related job. The ratings repeat the mistakes of the first live run."""
    items = (reqs("Minimum of seven years' post-secondary education or relevant work experience.")
             + reqs("Experience designing interactive Oracle APEX applications.", chip="Oracle APEX")
             + reqs("Ability to develop software in JavaScript", chip="JavaScript")
             + reqs("Use web services", chip="Web Services")
             + reqs("Adhere to Software Development Life Cycle", chip="Software Development Lifecycle")
             + reqs("Knowledge of relational databases"))
    reply = AdviceReply(ratings=[
        rate(1, "strong", "with 10 years of experience across software development"),
        rate(2, "not_shown"),
        rate(3, "weak", "Developed internal applications, data-processing tools, and automated pipelines using "
                        "Python, Java, SQL, and REST APIs."),
        rate(4, "strong", "created REST APIs, database queries, and stored procedures"),
        rate(5, "strong", "Development: Java, REST APIs, Git, JSON, Agile"),
        rate(6, "strong", "Designed backend functionality for enterprise applications using Java and SQL"),
    ], match_id=1, gap_id=2, next_step="Highlight your decade of software engineering in your summary.")
    text = build_advice(items, reply, JOHN, JOHN_ROLES)
    assert "seven years" not in text and "decade" not in text and ".." not in text
    assert "**Strongest match:** Use web services" in text
    assert "**Main gap:** Experience designing interactive Oracle APEX applications" in text
    assert "JavaScript" not in text  # Java doesn't show JavaScript
    assert "May not be a real gap" not in text  # Web Services is already the strongest match
    assert "Software Development Lifecycle" not in text  # a skills line can't call it not a real gap


def test_the_prompt_tells_the_llm_that_similar_tools_do_not_count():
    assert "Java\n   is not JavaScript" in (ROOT / "prompts" / "write_advice.md").read_text(encoding="utf-8")


# ---------- end to end ----------
def test_generate_advice_sends_requirements_evidence_and_the_full_cv():
    listed = {"items": [{"text": "Write SQL", "chip": ""}]}
    reply = {"ratings": [{"id": 1, "rating": "strong", "quote": "Wrote SQL queries"}], "match_id": 1}
    llm = FakeLLM(listed, reply)
    text = generate_advice(llm, CV, profile(ROLES), result(), by_words)
    assert len(llm.prompts) == 2
    assert f"- {PLANNING} " in llm.prompts[1]
    assert "B.Eng. Industrial Engineering" in llm.prompts[1]  # the full CV, to check before calling something a gap
    assert 'Your CV shows: "Wrote SQL queries"' in text


def test_no_requirements_makes_no_second_call():
    llm = FakeLLM({"items": []})
    text = generate_advice(llm, CV, profile(), result(), by_words)
    assert len(llm.prompts) == 1
    assert "couldn't find clear requirements" in text


def test_retrieval_failure_still_gives_advice():
    def broken(texts):
        raise LLMError("overloaded", try_other_model=True)
    reply = {"ratings": [{"id": 1, "rating": "not_shown"}], "gap_id": 1}
    llm = FakeLLM({"items": [{"text": "Lead a team"}]}, reply)
    text = generate_advice(llm, CV, profile(), result(), broken)
    assert "(none retrieved)" in llm.prompts[1]
    assert "**Main gap:** Lead a team" in text


def test_one_rating_that_is_not_different_is_not_a_poor_fit():
    reply = {"ratings": [{"id": 1, "rating": "not_shown"}], "gap_id": 1, "next_step": "Practise leading a team."}
    llm = FakeLLM({"items": [{"text": "Lead a team"}]}, reply)
    text = generate_advice(llm, CV, profile(), result(field_match="different", work_match="related"), by_words)
    assert "Practise leading a team." in text
