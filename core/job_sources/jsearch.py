"""JSearch (RapidAPI): aggregates Google for Jobs, including Indeed and LinkedIn listings.
Free BASIC plan is hard-capped monthly, so cache results and use the sample source in dev."""
from urllib.parse import urlparse

import requests

from ..schemas import Job
from .base import JobSource

URL = "https://jsearch.p.rapidapi.com/search-v2"


class JSearchSource(JobSource):
    name = "jsearch"

    def __init__(self, api_key: str, country: str = "us", timeout: int = 30, url: str = URL):
        if not api_key:
            raise ValueError("RAPIDAPI_KEY is missing. Add it to Streamlit secrets.")
        self.url = url
        self.headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": urlparse(url).netloc}
        self.country = country
        self.timeout = timeout

    def search(self, query: str, location: str, limit: int = 10) -> list[Job]:
        params = {
            "query": f"{query} in {location}" if location else query,
            "num_pages": 1,          # ~10 results per page; each page counts against quota
            "country": self.country,
            "date_posted": "month",
        }
        r = requests.get(self.url, headers=self.headers, params=params, timeout=self.timeout)
        if r.status_code == 429:
            raise RuntimeError("Job search quota reached for this month. "
                               "Switch JOB_SOURCE to 'sample' to keep demoing.")
        if not r.ok:
            # Show RapidAPI's own message (e.g. "not subscribed", "API doesn't exist"),
            # which says far more than the status code alone.
            raise RuntimeError(
                f"Job search failed (HTTP {r.status_code} from {self.url}): {r.text[:300]}. "
                "Check your JSearch subscription and RAPIDAPI_KEY in the RapidAPI playground, "
                "or set JOB_SOURCE to 'sample' to keep demoing.")
        data = r.json().get("data") or []
        if isinstance(data, dict):  # newer API version nests the list: {"data": {"jobs": [...]}}
            data = data.get("jobs") or []
        return [self._to_job(d) for d in data[:limit]]

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
