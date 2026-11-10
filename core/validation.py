"""Rule-based checks on what the user typed, before any LLM call or job search, so obvious junk
costs no quota."""

MAX_ROLE_CHARS = 80


# A role is too short to judge by keywords, so this only rejects obvious junk: no or one distinct
# letter ("123", "???", "aaaa") or a pasted paragraph. "teddy bear" passes; the search decides.
def looks_like_role(text: str) -> bool:
    role = text.strip()
    return len(role) <= MAX_ROLE_CHARS and len({c.lower() for c in role if c.isalpha()}) >= 2
