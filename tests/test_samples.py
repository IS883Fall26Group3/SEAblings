"""The fictional sample data the app offers when the user has no CV or job descriptions."""
from core.samples import SAMPLE_CVS, sample_cv, sample_jobs


def test_every_sample_cv_is_there_and_marked_fictional():
    for label, path in SAMPLE_CVS.items():
        assert path.exists(), label
        assert "fictional sample CV" in sample_cv(label)
    assert sample_cv("unknown") == sample_cv(next(iter(SAMPLE_CVS)))  # falls back to the default


def test_sample_postings_have_unique_ids_and_requirements():
    jobs = sample_jobs()
    assert len(jobs) == 10 and len({j.id for j in jobs}) == 10
    assert all("Required:" in j.description and "(sample)" in j.company for j in jobs)
