#!/usr/bin/env python3
"""Parse final answers with conservative, exact-match format recovery.

The normal protocol remains a single ``<result>...</result>`` JSON string
array.  If that wrapper is absent, scoring may recover one unambiguous fenced
code block, but only when its complete contents exactly match one accepted
reference answer.  Mentions in prose and ambiguous/conflicting code blocks are
never recovered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


RESULT_RE = re.compile(r"<result>\s*([\s\S]*?)\s*</result>")
FENCE_RE = re.compile(r"```(?:json|text)?\s*\n?(.*?)```", re.I | re.S)
LIST_PREFIX_RE = re.compile(r"^(?:[-*]\s+|\d+[.)]\s*)")


@dataclass(frozen=True)
class ParsedFinalAnswer:
    value: list[str] | None
    source: str
    recovered: bool = False


def expected_options(expected: Any) -> list[list[str]]:
    if isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        return [expected]
    if (
        isinstance(expected, list)
        and expected
        and all(isinstance(option, list) for option in expected)
        and all(all(isinstance(item, str) for item in option) for option in expected)
    ):
        return [list(option) for option in expected]
    raise TypeError("expected answer must be a JSON list of strings or alternatives")


def _parse_json_list(text: str) -> list[str] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return value


def _plain_fault_lines(text: str) -> list[str] | None:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = LIST_PREFIX_RE.sub("", raw_line.strip().strip("`").strip())
        line = line.strip().strip('"').strip("'").rstrip(",").strip()
        if line:
            lines.append(line)
    if not lines or not all(line.count(";") == 1 for line in lines):
        return None
    return lines


def fenced_candidates(text: str) -> list[list[str]]:
    candidates: list[list[str]] = []
    for block in FENCE_RE.findall(text):
        candidate = _parse_json_list(block.strip())
        if candidate is None:
            candidate = _plain_fault_lines(block)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def parse_final_answer(text: str, expected: Any | None = None) -> ParsedFinalAnswer:
    """Parse a strict result, or conservatively recover an exact fenced answer.

    Recovery is deliberately asymmetric: it is enabled only with an expected
    answer and only when exactly one unique fenced candidate matches an accepted
    option while no conflicting fault-list candidate is present.
    """

    matches = RESULT_RE.findall(text)
    if len(matches) == 1:
        value = _parse_json_list(matches[0])
        return ParsedFinalAnswer(
            value=value,
            source="result_tag" if value is not None else "invalid_result_json",
        )
    if len(matches) > 1:
        return ParsedFinalAnswer(value=None, source=f"ambiguous_result_tags:{len(matches)}")
    if expected is None:
        return ParsedFinalAnswer(value=None, source="missing_result_tag")

    options = expected_options(expected)
    candidates = fenced_candidates(text)
    matching = [candidate for candidate in candidates if candidate in options]
    conflicting = [candidate for candidate in candidates if candidate not in options]
    unique_matching = {json.dumps(candidate, ensure_ascii=False) for candidate in matching}
    if len(unique_matching) == 1 and not conflicting:
        return ParsedFinalAnswer(
            value=matching[0],
            source="recovered_fenced_exact_match",
            recovered=True,
        )
    if conflicting:
        return ParsedFinalAnswer(value=None, source="conflicting_fenced_candidates")
    return ParsedFinalAnswer(value=None, source="missing_result_tag")
