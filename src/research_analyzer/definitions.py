"""Detect explicit abbreviation definitions for targeted local retrieval."""

import re

_DEFINITION_QUESTION = re.compile(
    r"\bwhat\s+does\s+(?P<before>[A-Z][A-Z0-9]{1,})s?\s+"
    r"(?:mean|stand\s+for)\b|\b(?P<after>[A-Z][A-Z0-9]{1,})s?\s+"
    r"(?:mean|stands?\s+for)\b",
    re.IGNORECASE,
)
_PARENTHETICAL_ABBREVIATION = re.compile(
    r"(?P<phrase>(?:[A-Za-z]+(?:-[A-Za-z]+)*\s+){1,8})"
    r"\(\s*(?P<acronym>[A-Z][A-Z0-9]{1,})s?\s*\)"
)
_EXPLICIT_DEFINITION = re.compile(
    r"\b(?P<acronym>[A-Z][A-Z0-9]{1,})s?\s+"
    r"(?:means?|stands?\s+for)\s+"
    r"(?P<phrase>[A-Za-z][A-Za-z -]{1,80})",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z]+")


def requested_acronym(question: str) -> str | None:
    match = _DEFINITION_QUESTION.search(question)
    if not match:
        return None
    acronym = match.group("before") or match.group("after")
    return acronym if acronym.isupper() else None


def _expansion_for_acronym(phrase: str, acronym: str) -> str | None:
    words = _WORD.findall(phrase)
    letters = acronym.upper()
    for start in range(len(words)):
        suffix = words[start:]
        if "".join(word[0].upper() for word in suffix) != letters:
            continue
        if acronym.isupper() and acronym.endswith("S") is False:
            suffix[-1] = re.sub(r"s$", "", suffix[-1], flags=re.IGNORECASE)
        expansion = " ".join(suffix).strip()
        return " ".join(word[:1].upper() + word[1:] for word in expansion.split())
    return None


def abbreviation_definitions(text: str, acronym: str) -> set[str]:
    """Extract explicit long forms whose initials match the requested acronym."""
    definitions: set[str] = set()
    for match in _PARENTHETICAL_ABBREVIATION.finditer(text):
        if match.group("acronym").upper().rstrip("S") != acronym.upper().rstrip("S"):
            continue
        expansion = _expansion_for_acronym(match.group("phrase"), acronym)
        if expansion:
            definitions.add(expansion)
    for match in _EXPLICIT_DEFINITION.finditer(text):
        if match.group("acronym").upper().rstrip("S") != acronym.upper().rstrip("S"):
            continue
        phrase = re.split(r"[.;,\n]", match.group("phrase"), maxsplit=1)[0].strip()
        if phrase:
            definitions.add(phrase.rstrip())
    return definitions
