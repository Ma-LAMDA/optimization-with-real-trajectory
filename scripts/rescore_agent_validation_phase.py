#!/usr/bin/env python3
"""Version-rescore a complete, partial, or early-stopped Agent phase in place.

The phase's original manifests, events, attempts, and reports are read-only.
This command emits a new audit report from raw final answers.  Historical
model-failure ledgers remain diagnostic evidence but never override a final
answer, while timeout/infrastructure/interrupted cells remain excluded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from final_answer_scoring import (
    SCORING_POLICY_VERSION,
    evaluate_early_stop_ceiling,
    score_final_answer,
)
from summarize_agent_validation import load_expected, parse_attempt


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--model", default="")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_failures(phase: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = phase / "control" / "canonical_model_failures.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in payload.get("cells", []):
        key = (int(cell["case_id"]), int(cell["repeat_index"]))
        if key in result:
            raise ValueError(f"duplicate canonical model-failure cell: {key}")
        if cell.get("state") != "model_wrong":
            raise ValueError(f"unsupported canonical model-failure state: {cell}")
        result[key] = cell
    return result


def score_canonical_artifact(
    record: dict[str, Any], expected: Any, case_id: int, repeat: int
) -> dict[str, Any]:
    """Score a preserved first model terminal by its final answer only."""

    artifact = Path(str(record.get("artifact") or ""))
    manifest_path = artifact / "manifest.json"
    selected: int | None = None
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        run = (manifest.get("runs") or [{}])[0]
        attempts = run.get("attempts") or []
        selected = run.get("successful_attempt")
        if selected is None and attempts:
            selected = attempts[-1].get("attempt_index")
    final_paths = (
        list(artifact.glob(f"q{case_id:04d}_r*/attempt_{int(selected):03d}/final_answer.txt"))
        if selected is not None
        else []
    )
    final_text = (
        final_paths[0].read_text(encoding="utf-8", errors="replace")
        if len(final_paths) == 1
        else ""
    )
    scored = score_final_answer(final_text, expected, case_id=case_id)
    return {
        "case_id": case_id,
        "repeat": repeat,
        "effective_terminal": True,
        "correct": scored.correct,
        "result": "correct" if scored.correct else "wrong",
        "prediction": scored.prediction,
        "prediction_source": scored.source,
        "format_recovered": scored.recovered,
        "model_failure_reason": record.get("reason"),
        "model_failure_diagnostic_only": True,
        "artifact_dir": str(artifact),
        "scoring_policy_version": SCORING_POLICY_VERSION,
    }


def pending_state(root: Path, timeout_seconds: int) -> str:
    if (root / f".timeout_{timeout_seconds}s").is_file():
        return "timeout"
    if (root / ".runner_exit_code").is_file() or (root / "manifest.json").is_file():
        return "infrastructure_failure"
    return "not_run_or_archived"


def main() -> None:
    args = arguments()
    phase = args.phase.resolve()
    runs_root = phase / "runs"
    expected = load_expected(args.dataset.resolve())
    canonical = canonical_failures(phase)
    cells: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for case_id in args.case_ids:
            key = (case_id, repeat)
            if key in canonical:
                cells.append(score_canonical_artifact(
                    canonical[key], expected[case_id], case_id, repeat
                ))
                continue
            root = runs_root / f"{args.run_prefix}-q{case_id}-r{repeat:02d}"
            try:
                row = parse_attempt(
                    runs_root,
                    args.run_prefix,
                    case_id,
                    repeat,
                    expected[case_id],
                    args.timeout_seconds,
                    args.model,
                )
            except FileNotFoundError:
                row = {
                    "case_id": case_id,
                    "repeat": repeat,
                    "effective_terminal": False,
                    "correct": False,
                    "result": pending_state(root, args.timeout_seconds),
                    "artifact_dir": str(root),
                    "scoring_policy_version": SCORING_POLICY_VERSION,
                }
            cells.append(row)

    effective = [cell for cell in cells if cell["effective_terminal"]]
    wrong = [cell for cell in effective if not cell["correct"]]
    correct = len(effective) - len(wrong)
    completed_rounds = sum(
        all(
            next(
                cell for cell in cells
                if cell["case_id"] == case_id and cell["repeat"] == repeat
            )["effective_terminal"]
            for case_id in args.case_ids
        )
        for repeat in range(1, args.repeats + 1)
    )
    ceiling = evaluate_early_stop_ceiling(
        completed_full_effective_rounds=completed_rounds,
        effective_model_wrong=len(wrong),
        planned_total=len(args.case_ids) * args.repeats,
    )
    prior_path = phase / "reports" / "early_stop_summary.json"
    prior = load_json(prior_path) if prior_path.is_file() else None
    prior_check = None
    if prior is not None:
        prior_check = {
            "path": str(prior_path),
            "effective_terminals_match": prior.get("effective_terminals") == len(effective),
            "correct_match": prior.get("correct") == correct,
            "effective_model_wrong_match": prior.get("effective_model_wrong") == len(wrong),
        }
    per_case: dict[str, Any] = {}
    for case_id in args.case_ids:
        selected = [cell for cell in cells if cell["case_id"] == case_id]
        case_effective = [cell for cell in selected if cell["effective_terminal"]]
        case_correct = sum(bool(cell["correct"]) for cell in case_effective)
        per_case[str(case_id)] = {
            "results": [cell["result"] for cell in selected],
            "effective_terminals": len(case_effective),
            "correct": case_correct,
            "accuracy_percent": (
                100 * case_correct / len(case_effective) if case_effective else None
            ),
        }
    payload = {
        "schema_version": "agent-validation-phase-rescore.v1",
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": str(phase),
        "dataset": str(args.dataset.resolve()),
        "run_prefix": args.run_prefix,
        "case_ids": args.case_ids,
        "repeats": args.repeats,
        "planned_total": len(cells),
        "effective_terminals": len(effective),
        "correct": correct,
        "effective_model_wrong": len(wrong),
        "observed_accuracy_percent": 100 * correct / len(effective) if effective else None,
        "completed_full_effective_rounds": completed_rounds,
        "early_stop": ceiling.__dict__,
        "prior_summary_check": prior_check,
        "canonical_model_failure_cells": [list(key) for key in sorted(canonical)],
        "per_case": per_case,
        "cells": cells,
    }
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Agent phase rescore: {correct}/{len(effective)} observed; "
        f"wrong={len(wrong)}; ceiling={ceiling.maximum_possible_final_accuracy_percent:.2f}%"
    )


if __name__ == "__main__":
    main()
