from ..config import Settings
from .base import JobSource
from .jsearch import JSearchSource
from .sample import SampleSource


def get_job_source(s: Settings) -> JobSource:
    if s.job_source == "jsearch":
        return JSearchSource(s.rapidapi_key)
    return SampleSource()
