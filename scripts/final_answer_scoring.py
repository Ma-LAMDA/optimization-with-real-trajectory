#!/usr/bin/env python3
"""Canonical, versioned scoring for Agent final answers.

The primary protocol is exactly one valid ``<result>...</result>`` JSON string
list.  Only when the result wrapper is wholly absent may one fenced block be
recovered, and then only when the block's complete parsed answer exactly equals
one accepted option.  Prose mentions, malformed or multiple result wrappers,
and multiple/conflicting fenced blocks are never recovered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


SCORING_POLICY_VERSION = "agent-final-answer.v3.2026-08-10-final-answer-only"

RESULT_RE = re.compile(r"<result>\s*([\s\S]*?)\s*</result>", re.I)
RESULT_MARKER_RE = re.compile(r"</?result\b", re.I)
FENCE_RE = re.compile(r"```[^\r\n`]*\r?\n?([\s\S]*?)```", re.I)
FENCE_MARKER_RE = re.compile(r"```")
LIST_PREFIX_RE = re.compile(r"^(?:[-*]\s+|\d+[.)]\s*)")

VRRP_MASTER_01 = "Core_SW_01;VRRP Master角色规划不合理"
VRRP_MASTER_02 = "Core_SW_02;VRRP Master角色规划不合理"
VRRP_INCLUSIVE_OR_OPTIONS = (
    [VRRP_MASTER_01],
    [VRRP_MASTER_02],
    [VRRP_MASTER_01, VRRP_MASTER_02],
    [VRRP_MASTER_02, VRRP_MASTER_01],
)


@dataclass(frozen=True)
class ParsedFinalAnswer:
    value: list[str] | None
    source: str
    recovered: bool = False


@dataclass(frozen=True)
class ScoredFinalAnswer:
    prediction: list[str] | None
    correct: bool
    source: str
    recovered: bool
    accepted_options: tuple[tuple[str, ...], ...]
    scoring_policy_version: str = SCORING_POLICY_VERSION


@dataclass(frozen=True)
class EarlyStopCeiling:
    triggered: bool
    completed_full_effective_rounds: int
    effective_model_wrong: int
    planned_total: int
    maximum_possible_final_correct: int
    maximum_possible_final_accuracy_percent: float
    minimum_full_effective_rounds: int
    minimum_effective_model_wrong: int


def expected_options(expected: Any) -> list[list[str]]:
    if isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        return [list(expected)]
    if (
        isinstance(expected, list)
        and expected
        and all(isinstance(option, list) for option in expected)
        and all(all(isinstance(item, str) for item in option) for option in expected)
    ):
        return [list(option) for option in expected]
    raise TypeError("expected answer must be a JSON list of strings or alternatives")


def accepted_options(expected: Any, case_id: int | None = None) -> list[list[str]]:
    """Return exact accepted options, including the q73-q86 VRRP inclusive OR."""

    options = expected_options(expected)
    canonical_vrrp = {tuple(option) for option in VRRP_INCLUSIVE_OR_OPTIONS}
    if 73 <= int(case_id or -1) <= 86 and any(
        tuple(option) in canonical_vrrp for option in options
    ):
        options.extend(list(option) for option in VRRP_INCLUSIVE_OR_OPTIONS)
    deduplicated: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for option in options:
        key = tuple(option)
        if key not in seen:
            seen.add(key)
            deduplicated.append(option)
    return deduplicated


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


def _parse_fenced_block(block: str) -> list[str] | None:
    candidate = _parse_json_list(block.strip())
    return candidate if candidate is not None else _plain_fault_lines(block)


def parse_final_answer(text: str, expected: Any | None = None) -> ParsedFinalAnswer:
    """Parse one strict result or conservatively recover one exact fenced block."""

    matches = RESULT_RE.findall(text)
    markers = RESULT_MARKER_RE.findall(text)
    if len(matches) == 1 and len(markers) == 2:
        value = _parse_json_list(matches[0])
        return ParsedFinalAnswer(
            value=value,
            source="result_tag" if value is not None else "invalid_result_json",
        )
    if len(matches) > 1:
        return ParsedFinalAnswer(value=None, source=f"ambiguous_result_tags:{len(matches)}")
    if markers:
        return ParsedFinalAnswer(value=None, source="invalid_result_markup")
    if expected is None:
        return ParsedFinalAnswer(value=None, source="missing_result_tag")

    blocks = FENCE_RE.findall(text)
    fence_markers = len(FENCE_MARKER_RE.findall(text))
    if not blocks:
        return ParsedFinalAnswer(value=None, source="missing_result_tag")
    if len(blocks) != 1 or fence_markers != 2:
        return ParsedFinalAnswer(
            value=None,
            source=f"ambiguous_fenced_blocks:{len(blocks)}:{fence_markers}_markers",
        )
    candidate = _parse_fenced_block(blocks[0])
    if candidate is None:
        return ParsedFinalAnswer(value=None, source="invalid_fenced_answer")
    if candidate not in expected_options(expected):
        return ParsedFinalAnswer(value=None, source="conflicting_fenced_candidate")
    return ParsedFinalAnswer(
        value=candidate,
        source="recovered_fenced_exact_match",
        recovered=True,
    )


def score_final_answer(
    text: str,
    expected: Any,
    *,
    case_id: int | None = None,
) -> ScoredFinalAnswer:
    options = accepted_options(expected, case_id)
    parsed = parse_final_answer(text, options)
    return ScoredFinalAnswer(
        prediction=parsed.value,
        correct=parsed.value in options if parsed.value is not None else False,
        source=parsed.source,
        recovered=parsed.recovered,
        accepted_options=tuple(tuple(option) for option in options),
    )


def require_scoring_policy_version(payload_or_version: Mapping[str, Any] | str) -> None:
    version = (
        payload_or_version.get("scoring_policy_version")
        if isinstance(payload_or_version, Mapping)
        else payload_or_version
    )
    if version != SCORING_POLICY_VERSION:
        raise ValueError(
            "unversioned or stale correctness is forbidden: "
            f"expected {SCORING_POLICY_VERSION!r}, found {version!r}"
        )


def evaluate_early_stop_ceiling(
    *,
    completed_full_effective_rounds: int,
    effective_model_wrong: int,
    planned_total: int = 60,
    minimum_full_effective_rounds: int = 3,
    minimum_effective_model_wrong: int = 30,
) -> EarlyStopCeiling:
    for name, value in {
        "completed_full_effective_rounds": completed_full_effective_rounds,
        "effective_model_wrong": effective_model_wrong,
        "planned_total": planned_total,
        "minimum_full_effective_rounds": minimum_full_effective_rounds,
        "minimum_effective_model_wrong": minimum_effective_model_wrong,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    maximum_correct = max(0, planned_total - effective_model_wrong)
    maximum_percent = 100.0 * maximum_correct / planned_total if planned_total else 0.0
    triggered = (
        completed_full_effective_rounds >= minimum_full_effective_rounds
        and effective_model_wrong >= minimum_effective_model_wrong
    )
    return EarlyStopCeiling(
        triggered=triggered,
        completed_full_effective_rounds=completed_full_effective_rounds,
        effective_model_wrong=effective_model_wrong,
        planned_total=planned_total,
        maximum_possible_final_correct=maximum_correct,
        maximum_possible_final_accuracy_percent=maximum_percent,
        minimum_full_effective_rounds=minimum_full_effective_rounds,
        minimum_effective_model_wrong=minimum_effective_model_wrong,
    )
