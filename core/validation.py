"""Check that an input looks like a resume or a job description, using keywords only (no LLM call),
so the wrong file is caught before any model quota is spent."""
import re

# Section headings: a short line holding just the heading, optionally followed by a colon.
_CV_HEADINGS = (r"(professional |career )?summary|profile|objective|(work |professional )?experience"
                r"|employment( history)?|work history|education|(technical |core )?skills"
                r"|projects|certifications?|languages|awards|publications|volunteer(ing)?"
                r"|leadership|interests|references|achievements")
_HEADING = re.compile(rf"^\s*(?:{_CV_HEADINGS})\s*:?\s*$", re.I | re.M)
_DATE_RANGE = re.compile(r"\b(?:19|20)\d{2}\s*(?:-|–|—|to)\s*(?:(?:19|20)\d{2}|present|current|now)\b", re.I)
_CONTACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|linkedin\.com/in/|\+?\d[\d ()-]{8,}\d")
_DEGREE = re.compile(r"\b(?:bachelor|master|mba|ph\.?d|b\.?sc|m\.?sc|b\.?eng|b\.?a\.|m\.?a\.|gpa)\b", re.I)

_JD_PHRASES = re.compile(
    r"\b(?:responsibilities|requirements|qualifications|we are (?:looking for|hiring|seeking)"
    r"|we're (?:looking for|hiring|seeking)|you will|you'll|the ideal candidate|nice to have"
    r"|must have|required|preferred|about the role|about (?:the|this) (?:job|position)"
    r"|job description|what you'll do|who you are|benefits|salary|compensation|apply"
    r"|equal opportunity|years of (?:relevant |professional )?experience|is (?:seeking|hiring|looking for)"
    r"|seeks|(?:the|this) role|join (?:our|the) team|candidates?|recruiting|hiring|full-time|part-time"
    r"|the position|this position|what you'll be doing|what you will do|(?:hybrid|remote|on-?site) role"
    r"|requires|opportunity)\b", re.I)

MIN_CV_SCORE = 3
MIN_JD_SCORE = 2


def resume_score(text: str) -> int:
    """One point per distinct section heading, plus one each for dates, contact details and a degree."""
    headings = {m.group(0).strip(" :\t\n").lower() for m in _HEADING.finditer(text)}
    return (len(headings) + bool(_DATE_RANGE.search(text)) + bool(_CONTACT.search(text))
            + bool(_DEGREE.search(text)))


def job_description_score(text: str) -> int:
    """One point per distinct job-posting phrase."""
    return len({m.lower() for m in _JD_PHRASES.findall(text)})


# Postings often have Education/Skills headings too, so compare the scores: a resume should read
# more like a resume than a posting, and the other way round.
def looks_like_resume(text: str) -> bool:
    r = resume_score(text)
    return r >= MIN_CV_SCORE and r > job_description_score(text)


def looks_like_job_description(text: str) -> bool:
    j = job_description_score(text)
    return j >= MIN_JD_SCORE and j >= resume_score(text)
