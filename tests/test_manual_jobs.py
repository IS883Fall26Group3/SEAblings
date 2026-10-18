from core.job_sources.manual import SOURCE, jobs_from_descriptions

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
