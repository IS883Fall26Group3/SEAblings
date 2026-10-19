from ..config import Settings
from .base import JobSource
from .jsearch import JSearchSource
from .manual import jobs_from_descriptions  # noqa: F401
from .sample import SampleSource


def get_job_source(s: Settings) -> JobSource:
    if s.job_source == "jsearch":
        return JSearchSource(s.rapidapi_key, url=s.jsearch_url)
    return SampleSource()
