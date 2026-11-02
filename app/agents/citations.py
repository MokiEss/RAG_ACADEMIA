import re
from typing import List

TAG_PATTERN = re.compile(r"\[([^\[\]]+)\]")
DOC_TAG = re.compile(r"^.+\.(pdf|md|txt), p\.\d+$")


def extract_citations(text: str) -> List[str]:
    """Return unique citation tags in order of appearance."""
    seen: List[str] = []
    for tag in TAG_PATTERN.findall(text):
        tag = tag.strip()
        is_citation = tag.startswith("web:") or DOC_TAG.match(tag)
        if is_citation and tag not in seen:
            seen.append(tag)
    return seen


def invalid_citations(answer: str, evidence: str) -> List[str]:
    """Citations used in the answer that never appeared in the evidence."""
    return [tag for tag in extract_citations(answer) if f"[{tag}]" not in evidence]