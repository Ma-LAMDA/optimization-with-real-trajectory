#!/usr/bin/env python3
"""Summarize per-question Codex trajectory durations into a UTF-8 CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = EXPERIMENT_ROOT / "results" / "runs"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "results" / "reports" / "各题耗时统计.csv"
EXPECTED_CASE_IDS = (13, 14, 17, 18, 25, 26, 27, 28, 87, 88, 91, 92, 93, 94)
EXPECTED_REPEATS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Completed run directory. Defaults to the newest succeeded run.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def newest_succeeded_run(runs_root: Path) -> Path:
    candidates: list[Path] = []
    for manifest_path in runs_root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "succeeded":
            candidates.append(manifest_path.parent)
    if not candidates:
        raise RuntimeError(f"No succeeded run found in {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_successful_metadata(run_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for metadata_path in run_dir.glob("q*_r*/attempt_*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "succeeded":
            continue
        key = (int(metadata["case_id"]), int(metadata["repeat_index"]))
        previous = selected.get(key)
        if previous is None or int(metadata["attempt_index"]) > int(
            previous["attempt_index"]
        ):
            selected[key] = metadata
    return selected


def rounded(value: float) -> float:
    return round(value, 3)


def build_rows(
    selected: dict[tuple[int, int], dict[str, Any]],
) -> list[list[int | float]]:
    rows: list[list[int | float]] = []
    for case_id in EXPECTED_CASE_IDS:
        durations: list[float] = []
        for repeat_index in range(1, EXPECTED_REPEATS + 1):
            metadata = selected.get((case_id, repeat_index))
            if metadata is None:
                raise RuntimeError(
                    f"Missing successful trajectory: case {case_id}, run {repeat_index}"
                )
            durations.append(float(metadata["duration_seconds"]))
        rows.append(
            [
                case_id,
                *(rounded(value) for value in durations),
                rounded(min(durations)),
                rounded(statistics.mean(durations)),
                rounded(statistics.median(durations)),
                rounded(max(durations)),
                rounded(sum(durations)),
            ]
        )
    expected = len(EXPECTED_CASE_IDS) * EXPECTED_REPEATS
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} successful trajectories, found {len(selected)}"
        )
    return rows


def main() -> int:
    arguments = parse_args()
    run_dir = (
        arguments.run_dir.resolve()
        if arguments.run_dir
        else newest_succeeded_run(DEFAULT_RUNS_ROOT)
    )
    output = arguments.output.resolve()
    selected = load_successful_metadata(run_dir)
    rows = build_rows(selected)
    headers = [
        "题号",
        *(f"第{index}次耗时(秒)" for index in range(1, EXPECTED_REPEATS + 1)),
        "最短(秒)",
        "平均(秒)",
        "中位数(秒)",
        "最长(秒)",
        "总耗时(秒)",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    total_seconds = rounded(sum(float(row[-1]) for row in rows))
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output": str(output),
                "questions": len(rows),
                "trajectories": len(selected),
                "total_duration_seconds": total_seconds,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
