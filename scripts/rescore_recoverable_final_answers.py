#!/usr/bin/env python3
"""Recompute archived accuracy with inclusive-OR and format recovery rules.

Original trajectories and reports are read-only.  The script writes one new
machine-readable report covering Agent ``attempts.csv`` files, SFT inference
``validation_predictions.jsonl`` files, and distillation ``judgment.json``
records found beneath the requested roots.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from final_answer_scoring import expected_options, parse_final_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("roots", nargs="*", type=Path)
    return parser.parse_args()


def json_cell(value: str | None) -> Any:
    try:
        return json.loads(value or "null")
    except json.JSONDecodeError:
        return None


def scoring_options(case_id: int, expected: Any) -> list[list[str]]:
    options = expected_options(expected)
    if (
        73 <= case_id <= 86
        and len(options) >= 2
        and len(options[0]) == 1
        and len(options[1]) == 1
    ):
        answer_a, answer_b = options[0][0], options[1][0]
        for option in ([answer_a, answer_b], [answer_b, answer_a]):
            if option not in options:
                options.append(option)
    return options


def locate_final_answer(repo: Path, csv_path: Path, artifact: str) -> Path | None:
    raw = Path(artifact)
    candidates = [raw] if raw.is_absolute() else [repo / raw, csv_path.parent / raw]
    if raw.is_absolute() and not raw.exists():
        normalized = raw.as_posix()
        for marker in ("/output/", "/experiments/"):
            if marker in normalized:
                candidates.append(
                    repo / marker.strip("/") / normalized.split(marker, 1)[1]
                )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "final_answer.txt":
            return candidate
        if candidate.is_dir():
            matches = sorted(candidate.glob("q*_r*/attempt_*/final_answer.txt"))
            if not matches:
                matches = sorted(candidate.glob("attempt_*/final_answer.txt"))
            if matches:
                return matches[-1]
    return None


def bool_cell(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(original: list[bool], rescored: list[bool], effective: list[bool]) -> dict[str, Any]:
    total = len(original)
    effective_indexes = [index for index, keep in enumerate(effective) if keep]
    original_correct = sum(original)
    rescored_correct = sum(rescored)
    effective_original = sum(original[index] for index in effective_indexes)
    effective_rescored = sum(rescored[index] for index in effective_indexes)
    return {
        "rows": total,
        "original_correct": original_correct,
        "rescored_correct": rescored_correct,
        "accuracy_percent": 100 * rescored_correct / total if total else None,
        "delta_correct": rescored_correct - original_correct,
        "effective_terminals": len(effective_indexes),
        "effective_original_correct": effective_original,
        "effective_rescored_correct": effective_rescored,
        "effective_accuracy_percent": (
            100 * effective_rescored / len(effective_indexes)
            if effective_indexes
            else None
        ),
    }


def rescore_attempts(repo: Path, path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    original: list[bool] = []
    rescored: list[bool] = []
    effective: list[bool] = []
    recovered: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for row in rows:
        case_id = int(row.get("case_id") or 0)
        expected = json_cell(row.get("expected"))
        options = scoring_options(case_id, expected)
        prediction = json_cell(row.get("prediction"))
        source = "archived_prediction"
        if prediction is None:
            final_path = locate_final_answer(repo, path, row.get("artifact_dir") or "")
            if final_path:
                parsed = parse_final_answer(
                    final_path.read_text(encoding="utf-8", errors="replace"), options
                )
                prediction = parsed.value
                source = parsed.source
                if parsed.recovered:
                    recovered.append(
                        {
                            "case_id": case_id,
                            "repeat": int(row.get("repeat") or 0),
                            "source": source,
                            "artifact": final_path.as_posix(),
                        }
                    )
        was_correct = bool_cell(row.get("correct"))
        is_correct = isinstance(prediction, list) and prediction in options
        original.append(was_correct)
        rescored.append(is_correct)
        status = str(row.get("runner_status") or row.get("status") or "").lower()
        timeout = bool_cell(row.get("timeout")) or status == "timeout"
        infrastructure = bool_cell(row.get("infrastructure_failure")) or status in {
            "failed_before_manifest",
            "infrastructure_failure",
            "interrupted",
            "request_failed",
        }
        keep = not timeout and not infrastructure
        effective.append(keep)
        if was_correct != is_correct:
            changes.append(
                {
                    "case_id": case_id,
                    "repeat": int(row.get("repeat") or 0),
                    "from": was_correct,
                    "to": is_correct,
                    "prediction_source": source,
                }
            )
    return {
        "kind": "agent_attempts",
        **summarize(original, rescored, effective),
        "format_recovered": recovered,
        "changes": changes,
    }


def rescore_predictions(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    original: list[bool] = []
    rescored: list[bool] = []
    effective: list[bool] = []
    recovered: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected_result_items")
        prediction = row.get("actual_result_items")
        if prediction is None and isinstance(row.get("response_text"), str):
            parsed = parse_final_answer(row["response_text"], expected)
            prediction = sorted(set(parsed.value)) if parsed.value is not None else None
            if parsed.recovered:
                recovered.append({"id": row.get("id"), "source": parsed.source})
        original.append(bool(row.get("exact_match")))
        rescored.append(prediction == expected)
        effective.append(row.get("status") == "completed")
    return {
        "kind": "sft_validation_predictions",
        **summarize(original, rescored, effective),
        "format_recovered": recovered,
    }


def rescore_judgment(repo: Path, path: Path) -> tuple[bool, bool, bool]:
    judgment = json.loads(path.read_text(encoding="utf-8"))
    match = re.search(r"q(\d+)_r", path.as_posix())
    if not match:
        return bool(judgment.get("correct")), bool(judgment.get("correct")), False
    case_id = int(match.group(1))
    experiment = path
    while experiment.parent != experiment and experiment.parent.name != "experiments":
        experiment = experiment.parent
    source = experiment / f"results/questions/q{case_id:04d}/source_record.json"
    answer = path.parent / "final_answer.txt"
    if not source.is_file() or not answer.is_file():
        return bool(judgment.get("correct")), bool(judgment.get("correct")), False
    source_row = json.loads(source.read_text(encoding="utf-8"))
    expected = source_row.get("answer")
    if isinstance(expected, str):
        expected = json_cell(expected)
    try:
        options = scoring_options(case_id, expected)
    except TypeError:
        original = bool(judgment.get("correct"))
        return original, original, False
    parsed = parse_final_answer(answer.read_text(encoding="utf-8", errors="replace"), options)
    rescored = parsed.value in options if parsed.value is not None else False
    return bool(judgment.get("correct")), rescored, parsed.recovered


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    roots = [path.resolve() for path in args.roots] or [repo / "experiments", repo / "output"]
    attempt_files: set[Path] = set()
    prediction_files: set[Path] = set()
    judgment_files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        attempt_files.update(root.glob("**/attempts.csv"))
        prediction_files.update(root.glob("**/validation_predictions.jsonl"))
        judgment_files.update(root.glob("**/judgment.json"))

    files: dict[str, Any] = {}
    for path in sorted(attempt_files):
        files[path.relative_to(repo).as_posix()] = rescore_attempts(repo, path)
    for path in sorted(prediction_files):
        files[path.relative_to(repo).as_posix()] = rescore_predictions(path)

    grouped_judgments: dict[str, list[Path]] = {}
    for path in judgment_files:
        relative = path.relative_to(repo)
        key = "/".join(relative.parts[:2])
        grouped_judgments.setdefault(key, []).append(path)
    for key, paths in sorted(grouped_judgments.items()):
        values = [rescore_judgment(repo, path) for path in paths]
        original = [item[0] for item in values]
        rescored = [item[1] for item in values]
        files[key + "/results/runs/**/judgment.json"] = {
            "kind": "distillation_judgments",
            **summarize(original, rescored, [True] * len(values)),
            "format_recovered_count": sum(item[2] for item in values),
        }

    recovered_keys: set[str] = set()
    recovered_occurrences = 0
    for file_name, item in files.items():
        for recovered in item.get("format_recovered", []):
            recovered_occurrences += 1
            recovered_keys.add(
                str(
                    recovered.get("artifact")
                    or f"{file_name}:{recovered.get('id')}:{recovered.get('case_id')}:{recovered.get('repeat')}"
                )
            )
        recovered_count = int(item.get("format_recovered_count", 0))
        recovered_occurrences += recovered_count
        for index in range(recovered_count):
            recovered_keys.add(f"{file_name}:judgment:{index}")

    payload = {
        "schema_version": "recoverable-final-answer-rescore.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scorer_sha256": sha256(Path(__file__).with_name("final_answer_scoring.py")),
        "rescore_script_sha256": sha256(Path(__file__)),
        "rule": {
            "strict": "single <result> JSON string list",
            "recovery": "unique non-conflicting fenced exact match to an accepted answer",
            "prose_mentions_are_not_recovered": True,
            "q73_q86_inclusive_or": True,
        },
        "files": files,
        "totals": {
            "files": len(files),
            "rows": sum(item["rows"] for item in files.values()),
            "original_correct": sum(item["original_correct"] for item in files.values()),
            "rescored_correct": sum(item["rescored_correct"] for item in files.values()),
            "format_recovered_occurrences": recovered_occurrences,
            "format_recovered_unique_trajectories": len(recovered_keys),
        },
    }
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
