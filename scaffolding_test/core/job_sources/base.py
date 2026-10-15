from abc import ABC, abstractmethod

from ..schemas import Job


class JobSource(ABC):
    """Every job provider implements this one method, so sources are swappable."""
    name: str = "base"

    @abstractmethod
    def search(self, query: str, location: str, limit: int = 10) -> list[Job]:
        ...
