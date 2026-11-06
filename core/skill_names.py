"""Standard skill names, so the same skill written two ways ("MS Excel", "Excel", "Phyton") compares
as one. data/skill_aliases.csv maps a cleaned variant to its standard name; it is built from a short
hand-made list and O*NET 31.0 software names by scripts/build_skill_aliases.py.
A plain lookup: no LLM or embedding call. Names not in the table are only cleaned."""
import csv
import re
from functools import lru_cache
from pathlib import Path

ALIASES = Path(__file__).resolve().parent.parent / "data" / "skill_aliases.csv"


US_SPELLING = {"modelling": "modeling", "modelled": "modeled", "labelling": "labeling",
               "travelling": "traveling", "cancelling": "canceling", "analyse": "analyze",
               "analysing": "analyzing", "behaviour": "behavior", "behavioural": "behavioral",
               "colour": "color", "centre": "center", "licence": "license", "programme": "program",
               "programmes": "programs", "catalogue": "catalog", "defence": "defense"}


# Words ending in -ise that are not the British spelling of an -ize word (also covers -ised, -ising...)
KEEP_ISE = {"expertise", "enterprise", "exercise", "advertise", "advise", "merchandise", "franchise",
            "supervise", "compromise", "premise", "promise", "revise", "devise", "precise", "concise",
            "otherwise", "improvise", "televise", "comprise", "surprise", "disguise", "noise", "praise",
            "raise", "cruise", "poise", "treatise", "wise", "likewise", "excise", "incise", "despise",
            "demise", "paradise", "chastise", "arise", "rise", "appraise", "apprise"}


def _us(word: str) -> str:
    """British to US spelling: the word list above, and -ise/-isation words ('optimise', 'visualisation')."""
    word = US_SPELLING.get(word, word)
    m = re.fullmatch(r"([a-z]{3,})is(e|es|ed|ing|ation|ations|ational|er|ers)", word)
    return f"{m[1]}iz{m[2]}" if m and f"{m[1]}ise" not in KEEP_ISE else word


# "Soft skills" is a skill of its own, not "soft"
KEEP_SKILLS = {"soft", "hard", "people", "life", "core", "key", "transferable", "technical", "social"}


def normalize(name: str) -> str:
    """Lowercase, '&' as 'and', punctuation as spaces (keeping + # . for C++, C#, ASP.NET),
    US spelling, and no trailing 'skills' ('Communication Skills' compares as 'communication')."""
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9+#.]+", " ", s)
    s = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", s)  # a dot only inside a name (ASP.NET)
    words = [_us(w) for w in s.split()]
    if len(words) > 1 and words[-1] in ("skills", "skill") and " ".join(words[:-1]) not in KEEP_SKILLS:
        words = words[:-1]
    return " ".join(words)


@lru_cache(maxsize=4)
def load_aliases(path: Path = ALIASES) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["variant"]: row["canonical"] for row in csv.DictReader(f)}


def standard_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """The standard name for a skill, or the name itself if the table doesn't know it."""
    aliases = load_aliases() if aliases is None else aliases
    return aliases.get(normalize(name), name.strip())


def skill_key(name: str, aliases: dict[str, str] | None = None) -> str:
    """What two skills are compared by: the cleaned standard name."""
    return normalize(standard_name(name, aliases))
