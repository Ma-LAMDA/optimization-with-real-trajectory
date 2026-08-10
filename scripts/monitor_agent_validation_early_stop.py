#!/usr/bin/env python3
"""Stop a 12x5 Agent validation once its final 50% ceiling is impossible."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--launcher-pid-file", type=Path, required=True)
    parser.add_argument("--repo-scripts", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--minimum-full-rounds", type=int, default=3)
    parser.add_argument("--minimum-wrong-count", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--expected-launcher-fragment", required=True)
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_expected(path: Path) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[int(row["id"])] = json.loads(row["answer"])
    return result


def cell_status(
    *,
    phase: Path,
    prefix: str,
    case_id: int,
    repeat: int,
    expected: Any,
    score_final_answer: Any,
    timeout_seconds: int,
) -> dict[str, Any]:
    root = phase / "runs" / f"{prefix}-q{case_id}-r{repeat:02d}"
    manifest_path = root / "manifest.json"
    timeout = (root / f".timeout_{timeout_seconds}s").exists()
    if not manifest_path.is_file():
        return {"effective": False, "state": "timeout" if timeout else "pending"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = (manifest.get("runs") or [{}])[0]
    attempts = run.get("attempts") or []
    duration = sum(float(item.get("duration_seconds") or 0) for item in attempts)
    timeout = timeout or duration > timeout_seconds
    if timeout:
        return {"effective": False, "state": "timeout"}
    selected = run.get("successful_attempt")
    if selected is None and attempts:
        selected = attempts[-1].get("attempt_index")
    if selected is None:
        return {"effective": False, "state": "pending"}
    attempt = root / f"q{case_id:04d}_r01" / f"attempt_{int(selected):03d}"
    metadata_path = attempt / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    runner_status = str(manifest.get("status") or "failed_before_manifest")
    event_counts = metadata.get("event_type_counts") or {}
    completed_model_turn = (
        runner_status == "failed"
        and metadata.get("exit_code") == 0
        and int(event_counts.get("turn.completed") or 0) > 0
        and not metadata.get("error_events")
        and not metadata.get("invalid_jsonl_events")
        and not metadata.get("launch_error")
    )
    model_protocol_failure = (root / ".model_protocol_failure").is_file()
    final_answer = attempt / "final_answer.txt"
    text = final_answer.read_text(encoding="utf-8", errors="replace") if final_answer.is_file() else ""
    scored = score_final_answer(text, expected, case_id=case_id)
    has_valid_final_answer = scored.prediction is not None
    effective = (
        has_valid_final_answer
        or runner_status == "succeeded"
        or completed_model_turn
        or model_protocol_failure
    )
    if not effective:
        return {"effective": False, "state": "infrastructure_failure"}
    # Accuracy is based only on the final answer.  Protocol/tool diagnostics
    # may explain the run but cannot flip a matching answer to wrong.
    correct = scored.correct
    return {
        "effective": True,
        "state": "correct" if correct else "model_wrong",
        "correct": correct,
        "parse_source": scored.source,
        "model_protocol_failure": model_protocol_failure,
        "artifact": str(root),
    }


def acquire_lock(path: Path) -> None:
    payload = {"pid": os.getpid(), "started_at": now()}
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_pid = int(old.get("pid") or 0)
        except (OSError, ValueError, json.JSONDecodeError):
            old_pid = 0
        if old_pid and process_alive(old_pid):
            raise SystemExit(f"early-stop monitor already active as pid {old_pid}")
        archived = path.with_name(f"{path.name}.stale.{int(time.time())}")
        path.replace(archived)
        return acquire_lock(path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def terminate_launcher(args: argparse.Namespace, decision: dict[str, Any]) -> None:
    launcher_pid = int(args.launcher_pid_file.read_text(encoding="utf-8").strip())
    command_path = Path(f"/proc/{launcher_pid}/cmdline")
    command = command_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    if args.expected_launcher_fragment not in command:
        raise RuntimeError(
            f"refusing to terminate pid {launcher_pid}: unexpected command {command!r}"
        )
    process_group = os.getpgid(launcher_pid)
    if process_group == os.getpgrp():
        raise RuntimeError("refusing to terminate the monitor process group")
    decision["termination"] = {
        "launcher_pid": launcher_pid,
        "process_group": process_group,
        "signal": "SIGTERM",
        "requested_at": now(),
    }
    atomic_json(args.phase / "control" / "early_stop_decision.json", decision)
    os.killpg(process_group, signal.SIGTERM)
    deadline = time.time() + 60
    while time.time() < deadline and process_alive(launcher_pid):
        time.sleep(2)
    if process_alive(launcher_pid):
        decision["termination"]["escalated_signal"] = "SIGKILL"
        decision["termination"]["escalated_at"] = now()
        atomic_json(args.phase / "control" / "early_stop_decision.json", decision)
        os.killpg(process_group, signal.SIGKILL)
    decision["termination"]["launcher_exited"] = not process_alive(launcher_pid)
    decision["termination"]["checked_at"] = now()
    atomic_json(args.phase / "control" / "early_stop_decision.json", decision)


def main() -> int:
    args = arguments()
    control = args.phase / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock = control / "early_stop_monitor.lock"
    acquire_lock(lock)
    sys.path.insert(0, str(args.repo_scripts))
    from final_answer_scoring import (
        SCORING_POLICY_VERSION,
        evaluate_early_stop_ceiling,
        score_final_answer,
    )

    expected = load_expected(args.dataset)
    state_path = control / "early_stop_monitor_state.json"
    total = len(args.case_ids) * args.repeats
    try:
        while True:
            cells: dict[tuple[int, int], dict[str, Any]] = {}
            for repeat in range(1, args.repeats + 1):
                for case_id in args.case_ids:
                    cells[(case_id, repeat)] = cell_status(
                        phase=args.phase,
                        prefix=args.run_prefix,
                        case_id=case_id,
                        repeat=repeat,
                        expected=expected[case_id],
                        score_final_answer=score_final_answer,
                        timeout_seconds=args.timeout_seconds,
                    )
            completed_rounds = sum(
                all(cells[(case_id, repeat)]["effective"] for case_id in args.case_ids)
                for repeat in range(1, args.repeats + 1)
            )
            effective = [cell for cell in cells.values() if cell["effective"]]
            wrong = [cell for cell in effective if not cell.get("correct")]
            ceiling = evaluate_early_stop_ceiling(
                completed_full_effective_rounds=completed_rounds,
                effective_model_wrong=len(wrong),
                planned_total=total,
                minimum_full_effective_rounds=args.minimum_full_rounds,
                minimum_effective_model_wrong=args.minimum_wrong_count,
            )
            state = {
                "schema_version": "agent-validation-early-stop-monitor.v1",
                "scoring_policy_version": SCORING_POLICY_VERSION,
                "updated_at": now(),
                "phase": str(args.phase),
                "run_prefix": args.run_prefix,
                "completed_full_effective_rounds": completed_rounds,
                "effective_terminals": len(effective),
                "effective_model_wrong": len(wrong),
                "planned_total": total,
                "maximum_possible_final_correct": ceiling.maximum_possible_final_correct,
                "maximum_possible_final_accuracy_percent": ceiling.maximum_possible_final_accuracy_percent,
                "trigger": {
                    "minimum_full_rounds": args.minimum_full_rounds,
                    "minimum_wrong_count": args.minimum_wrong_count,
                },
                "triggered": ceiling.triggered,
            }
            atomic_json(state_path, state)
            if state["triggered"]:
                decision = {
                    **state,
                    "status": "early_stopped_accuracy_ceiling_not_above_50_percent",
                    "wrong_artifacts": [cell.get("artifact") for cell in wrong],
                }
                terminate_launcher(args, decision)
                return 0
            launcher_pid = int(args.launcher_pid_file.read_text(encoding="utf-8").strip())
            if not process_alive(launcher_pid):
                state["status"] = "launcher_exited_without_early_stop_trigger"
                atomic_json(state_path, state)
                return 0
            time.sleep(args.poll_seconds)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
