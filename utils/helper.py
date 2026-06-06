from __future__ import annotations
import re

_LEADING_ENUM = re.compile(r"^\s*(\d+[\).\-\s]+|[•\-\*]\s*)")


def clean_topics(text: str) -> list[str]:
    topics: list[str] = []
    seen:   set[str]  = set()
    current_parent    = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _LEADING_ENUM.sub("", line).strip()
        if not line:
            continue
        is_indented = raw_line != raw_line.lstrip() and current_parent
        if is_indented:
            for sub in [s.strip() for s in line.split(",") if s.strip()]:
                _add(topics, seen, f"{current_parent} - {sub}")
            continue
        if ":" in line:
            main, rest = line.split(":", 1)
            main = main.strip()
            current_parent = main
            subtopics = [s.strip() for s in rest.split(",") if s.strip()]
            if subtopics:
                for sub in subtopics:
                    _add(topics, seen, f"{main} - {sub}")
            else:
                _add(topics, seen, main)
        else:
            current_parent = line
            _add(topics, seen, line)

    return topics


def _add(topics: list[str], seen: set[str], entry: str) -> None:
    entry = entry.strip()[:120]
    if entry and entry not in seen:
        seen.add(entry)
        topics.append(entry)
