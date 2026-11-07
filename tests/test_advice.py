import json
from pathlib import Path

import numpy as np

from core.advice import (AdviceReply, ListedRequirement, RequirementRating, build_advice, chunk_cv, generate_advice,
                         list_requirements, retrieve_evidence)
from core.llm.base import LLMError, LLMProvider
from core.schemas import CandidateProfile, Job, MatchResult, Role

CV = (Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_cv.txt").read_text(encoding="utf-8")
ROLES = [Role(title="Corporate Planning Analyst"), Role(title="Business Analyst Intern")]
PLANNING = "Corporate Planning Analyst, Example Energy Co. (2021 - 2025)"


def reqs(*texts, chip=""):
    return [ListedRequirement(text=t, chip=chip) for t in texts]


def rate(id, rating, quote=""):
    return RequirementRating(id=id, rating=rating, quote=quote)


def result(missing=()):
    job = Job(id="j1", title="Data Analyst", company="Acme", location="", description="Needs SQL and a team lead.")
    return MatchResult(job=job, score=50, skill_coverage=0.5, semantic_similarity=0.5, seniority_fit=0.5,
                       matched_skills=[], missing_skills=list(missing))


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
    assert intern.startswith("Business Analyst Intern, Sample Consulting (2020) | ")
    assert "SQL" not in intern and "MBA" not in intern
    assert not any("EDUCATION" in c and "Wrote" in c for c in chunks)
    assert not any(c.endswith("Wrote SQL queries") or c.endswith("Built") for c in chunks)


def test_every_chunk_of_a_split_role_names_the_role_and_repeats_one_line():
    lines = [f"- Bullet {name} " + "word " * 7 for name in "ABCDE"]  # 10 words each
    cv = "Data Analyst, Acme (2020 - 2024)\n" + "\n".join(lines)
    chunks = chunk_cv(cv, [Role(title="Data Analyst")])
    assert len(chunks) == 2
    assert all(c.startswith("Data Analyst, Acme (2020 - 2024) | ") for c in chunks)
    assert "Bullet D" in chunks[0] and "Bullet D" in chunks[1]  # one line of overlap
    assert "Bullet E" not in chunks[0] and "Bullet C" not in chunks[1]


def test_a_role_that_fits_is_not_split():
    chunks = [c for c in chunk_cv(CV, ROLES) if c.startswith(PLANNING)]
    assert len(chunks) == 1
    assert "Built financial models" in chunks[0] and "Wrote SQL queries" in chunks[0]


def test_word_limit_does_not_count_the_role_line():
    for chunk in chunk_cv(CV, ROLES):
        assert len(chunk.split(" | ")[-1].split()) <= 40


def test_without_roles_it_falls_back_to_plain_lines():
    assert any(c.startswith("- Conducted market research") or "Conducted market research" in c
               for c in chunk_cv(CV))
    assert not any("|" in c for c in chunk_cv(CV))


def test_a_cv_with_no_line_breaks_still_gets_40_word_chunks():
    chunks = chunk_cv(" ".join(CV.split()))
    assert len(chunks) > 1
    assert all(len(c.split()) <= 40 for c in chunks)


def test_a_role_with_no_lines_of_its_own_is_kept():
    assert "Analyst, Acme (2020)" in chunk_cv("Analyst, Acme (2020)\n\nEDUCATION\nMBA", [Role(title="Analyst")])


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
    items = reqs("Write Rust")
    reply = AdviceReply(ratings=[rate(1, "strong", "Built compilers in Rust for 10 years")], match_id=1)
    text = build_advice(items, reply, CV)
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
    assert "If you have done this, add a specific example" in text


def test_nothing_to_show_says_so_instead_of_inventing():
    text = build_advice(reqs("Write SQL"), AdviceReply(ratings=[]), CV)
    assert "No strong match found" in text and "No clear gap found" in text


def test_a_weak_link_is_worded_as_weak():
    reply = AdviceReply(ratings=[rate(1, "weak", "Python (basic)")], gap_id=1, next_step="Do a project.")
    assert 'Your CV only shows a weak link: "Python (basic)"' in build_advice(reqs("Python for analysis"), reply, CV)


def test_dollar_signs_can_not_turn_into_a_math_formula():
    reply = AdviceReply(ratings=[rate(1, "strong", "$2B downstream investment portfolio")], match_id=1,
                        next_step="Aim for $5B.")
    text = build_advice(reqs("Manage budgets"), reply, CV)
    assert r"\$2B" in text and "$2B" not in text.replace(r"\$", "")


# ---------- option C: "To build" chips ----------
def test_a_chip_the_cv_contradicts_is_explained():
    items = reqs("Write queries for reporting", chip="Reporting")
    reply = AdviceReply(ratings=[rate(1, "strong", "Wrote SQL queries")], match_id=1)
    assert '"To build" lists Reporting, but your CV shows "Wrote SQL queries", so this may not be a real gap.' in (
        build_advice(items, reply, CV))


def test_a_confirmed_chip_gets_no_note():
    items = reqs("Use Tableau", chip="Tableau")
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1)
    text = build_advice(items, reply, CV)
    assert "**Main gap:** Use Tableau" in text
    assert "may not be a real gap" not in text


def test_a_chip_named_in_the_cv_can_not_be_called_not_shown():
    items = reqs("Know SAP well", chip="SAP") + reqs("Lead a team")
    reply = AdviceReply(ratings=[rate(1, "not_shown"), rate(2, "not_shown")], gap_id=2)
    text = build_advice(items, reply, CV)
    assert "**Main gap:** Lead a team" in text
    assert '"To build" lists SAP, but your CV shows "SAP"' in text


def test_chip_check_needs_a_whole_word():
    items = reqs("Know R", chip="R")
    reply = AdviceReply(ratings=[rate(1, "not_shown")], gap_id=1)
    assert "**Main gap:** Know R. Your CV doesn't show this." in build_advice(items, reply, "Worked as a RESEARCHER.")


# ---------- end to end ----------
def test_generate_advice_sends_requirements_evidence_and_the_full_cv():
    listed = {"items": [{"text": "Write SQL", "chip": ""}]}
    reply = {"ratings": [{"id": 1, "rating": "strong", "quote": "Wrote SQL queries"}], "match_id": 1}
    llm = FakeLLM(listed, reply)
    text = generate_advice(llm, CV, CandidateProfile(summary="", years_experience=4, seniority="mid", skills=[],
                                                     job_titles=[], roles=ROLES), result(), by_words)
    assert len(llm.prompts) == 2
    assert f"- {PLANNING} | " in llm.prompts[1]
    assert "B.Eng. Industrial Engineering" in llm.prompts[1]  # the full CV, to check before calling something a gap
    assert 'Your CV shows: "Wrote SQL queries"' in text


def test_no_requirements_makes_no_second_call():
    llm = FakeLLM({"items": []})
    text = generate_advice(llm, CV, CandidateProfile(summary="", years_experience=4, seniority="mid", skills=[],
                                                     job_titles=[]), result(), by_words)
    assert len(llm.prompts) == 1
    assert "couldn't find clear requirements" in text


def test_retrieval_failure_still_gives_advice():
    def broken(texts):
        raise LLMError("overloaded", try_other_model=True)
    reply = {"ratings": [{"id": 1, "rating": "not_shown"}], "gap_id": 1}
    llm = FakeLLM({"items": [{"text": "Lead a team"}]}, reply)
    text = generate_advice(llm, CV, CandidateProfile(summary="", years_experience=4, seniority="mid", skills=[],
                                                     job_titles=[]), result(), broken)
    assert "(none retrieved)" in llm.prompts[1]
    assert "**Main gap:** Lead a team" in text
