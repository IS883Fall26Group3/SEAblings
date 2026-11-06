from core.job_sources.manual import SOURCE, jobs_from_descriptions, name_from_requirements
from core.schemas import JobRequirements

JD = "Senior Data Analyst\n\nWe need SQL, Python and Tableau. 3+ years of experience."


def test_builds_job_with_title_from_first_line():
    [job] = jobs_from_descriptions([("jd.txt", JD)])
    assert job.title == "Senior Data Analyst"
    assert job.description == JD
    assert job.source == SOURCE


def test_skips_empty_and_duplicates():
    jobs = jobs_from_descriptions([("a", JD), ("b", "  "), ("c", JD)])
    assert len(jobs) == 1


def test_long_first_line_falls_back_to_name():
    [job] = jobs_from_descriptions([("pasted job", "x" * 200)])
    assert job.title == "pasted job"


def test_skips_linkedin_logo_caption_and_company():
    text = "Company logo for, ATC.\nATC\n\nData Analyst \nMassachusetts, United States\n" + "x" * 100
    [job] = jobs_from_descriptions([("jd", text)])
    assert job.title == "Data Analyst"


def test_llm_title_and_company_replace_first_line():
    [job] = jobs_from_descriptions([("jd", JD)])
    job.requirements = JobRequirements(required_skills=[], nice_to_have_skills=[],
                                       job_title="Lead Specialist - Area Account Mgt", company="Sunoco LP")
    [job] = name_from_requirements([job])
    assert (job.title, job.company) == ("Lead Specialist - Area Account Mgt", "Sunoco LP")


def test_keeps_first_line_when_llm_finds_no_title():
    [job] = jobs_from_descriptions([("jd", JD)])
    job.requirements = JobRequirements(required_skills=[], nice_to_have_skills=[])
    [job] = name_from_requirements([job])
    assert (job.title, job.company) == ("Senior Data Analyst", "")
