#!/usr/bin/env python3
"""Expand q73-q86 VRRP answers to inclusive-OR alternatives."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


QUESTION_IDS = set(range(73, 87))
ANSWER_A = "Core_SW_01;VRRP Master角色规划不合理"
ANSWER_B = "Core_SW_02;VRRP Master角色规划不合理"
EXPECTED_PREFIX = [[ANSWER_A], [ANSWER_B]]
INCLUSIVE_OR_OPTIONS = [
    [ANSWER_A],
    [ANSWER_B],
    [ANSWER_A, ANSWER_B],
    [ANSWER_B, ANSWER_A],
]
DEFAULT_PATHS = (
    Path("data/simulation/train_0629.jsonl"),
    Path("experiments/2026-07-27-ip_codex_train0629_14x10/inputs/train_0629.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--from-git", metavar="REV", help="read each input from REV before updating")
    return parser.parse_args()


def update_file(path: Path, source_revision: str | None = None) -> int:
    if source_revision:
        source = subprocess.check_output(
            ["git", "show", f"{source_revision}:{path.as_posix()}"]
        ).decode("utf-8")
    else:
        source = path.read_bytes().decode("utf-8")
    lines = source.splitlines(keepends=True)
    output: list[str] = []
    updated = 0
    seen: set[int] = set()
    for line in lines:
        row = json.loads(line)
        question_id = int(row["id"])
        if question_id not in QUESTION_IDS:
            output.append(line)
            continue
        current = json.loads(row["answer"])
        if current[:2] != EXPECTED_PREFIX:
            raise ValueError(f"{path}: q{question_id} has unexpected A/B answers: {current!r}")
        row["answer"] = json.dumps(INCLUSIVE_OR_OPTIONS, ensure_ascii=False, indent=2)
        newline = "\r\n" if line.endswith("\r\n") else "\n"
        output.append(json.dumps(row, ensure_ascii=False) + newline)
        seen.add(question_id)
        updated += current != INCLUSIVE_OR_OPTIONS
    missing = QUESTION_IDS - seen
    if missing:
        raise ValueError(f"{path}: missing question ids {sorted(missing)}")
    path.write_bytes("".join(output).encode("utf-8"))
    return updated


def main() -> None:
    args = parse_args()
    for path in args.paths:
        print(f"{path}: updated {update_file(path, args.from_git)} rows")


if __name__ == "__main__":
    main()
