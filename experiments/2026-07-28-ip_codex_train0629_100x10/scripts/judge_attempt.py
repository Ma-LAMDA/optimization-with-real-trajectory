#!/usr/bin/env python3
"""Post-generation judge.  Its output intentionally contains no ground answer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from final_answer_scoring import parse_final_answer



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReferenceOptions(list):
    pass


class FaultSet(list):
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReferenceOptions):
            return any(list.__eq__(self, option) for option in other)
        return list.__eq__(self, other)


def canonical_fault_set(value: Any) -> list[str]:
    if (
        isinstance(value, list)
        and value
        and all(isinstance(option, list) for option in value)
        and all(
            all(isinstance(item, str) for item in option)
            for option in value
        )
    ):
        return ReferenceOptions(
            [FaultSet(sorted(set(option))) for option in value]
        )
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("result must be a JSON array of strings")
    return FaultSet(sorted(set(value)))


def parse_prediction(
    text: str, gold: FaultSet | ReferenceOptions
) -> tuple[bool, str, list[str] | None]:
    parsed = parse_final_answer(text, gold)
    if parsed.value is None:
        return False, parsed.source, None
    try:
        return True, parsed.source, canonical_fault_set(parsed.value)
    except TypeError as exc:
        return False, f"result_error:{type(exc).__name__}", None


def read_row(dataset: Path, row_index: int) -> dict[str, Any]:
    with dataset.open("r", encoding="utf-8-sig") as handle:
        for current, line in enumerate(handle, start=1):
            if current == row_index:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("source record is not an object")
                return row
    raise IndexError(f"row_index {row_index} is outside the dataset")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--original-id", required=True)
    parser.add_argument("--final-answer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    final_text = args.final_answer.read_text(encoding="utf-8", errors="replace")
    row = read_row(args.dataset, args.row_index)
    if str(row.get("id")) != args.original_id:
        raise ValueError("row identity mismatch")
    try:
        gold = canonical_fault_set(json.loads(row["answer"]))
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("source answer is invalid for the established comparator") from exc
    parsed_ok, parse_status, predicted = parse_prediction(final_text, gold)
    correct = bool(parsed_ok and predicted == gold)
    payload = {
        "schema_version": "ip-distill-judgment.v1",
        "judge_status": "completed",
        "parsed": parsed_ok,
        "parse_status": parse_status,
        "correct": correct,
        "format_recovered": parse_status == "recovered_fenced_exact_match",
        "comparator": "established_fault_set_exact_v1",
        "comparison_rule": "exact set equality for JSON string arrays; order ignored; missing/extra/different items fail",
        "final_answer_sha256": sha256(args.final_answer),
    }
    payload['comparator'] = 'established_fault_set_exact_with_alternatives_v2'
    payload['comparison_rule'] = (
        'prediction must be a JSON string array exactly equal to the flat '
        'reference or to any explicitly listed nested reference alternative; '
        'order ignored; missing, extra, or different items fail'
    )
    payload['reference_shape'] = (
        'alternatives' if isinstance(gold, ReferenceOptions) else 'flat'
    )
    payload['acceptable_alternative_count'] = (
        len(gold) if isinstance(gold, ReferenceOptions) else 1
    )
    atomic_json(args.output, payload)
    print(json.dumps({"parsed": parsed_ok, "parse_status": parse_status, "correct": correct}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
