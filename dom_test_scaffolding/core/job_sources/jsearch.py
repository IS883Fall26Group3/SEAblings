"""JSearch (RapidAPI): aggregates Google for Jobs, including Indeed and LinkedIn listings.
Free BASIC plan is hard-capped monthly, so cache results and use the sample source in dev."""
import requests

from ..schemas import Job
from .base import JobSource

URL = "https://jsearch.p.rapidapi.com/search"


class JSearchSource(JobSource):
    name = "jsearch"

    def __init__(self, api_key: str, country: str = "us", timeout: int = 30):
        if not api_key:
            raise ValueError("RAPIDAPI_KEY is missing. Add it to Streamlit secrets.")
        self.headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
        self.country = country
        self.timeout = timeout

    def search(self, query: str, location: str, limit: int = 10) -> list[Job]:
        params = {
            "query": f"{query} in {location}" if location else query,
            "page": 1,
            "num_pages": 1,          # ~10 results per page; each page counts against quota
            "country": self.country,
            "date_posted": "month",
        }
        r = requests.get(URL, headers=self.headers, params=params, timeout=self.timeout)
        if r.status_code == 429:
            raise RuntimeError("Job search quota reached for this month. "
                               "Switch JOB_SOURCE to 'sample' to keep demoing.")
        r.raise_for_status()
        return [self._to_job(d) for d in r.json().get("data", [])[:limit]]

    @staticmethod
    def _to_job(d: dict) -> Job:
        loc = ", ".join(x for x in (d.get("job_city"), d.get("job_state"), d.get("job_country")) if x)
        return Job(
            id=str(d.get("job_id", "")),
            title=d.get("job_title") or "",
            company=d.get("employer_name") or "",
            location=loc or ("Remote" if d.get("job_is_remote") else ""),
            description=d.get("job_description") or "",
            url=d.get("job_apply_link") or "",
            source=d.get("job_publisher") or "",
            employment_type=d.get("job_employment_type") or "",
            posted_at=d.get("job_posted_at_datetime_utc") or "",
        )
