#!/usr/bin/env python3
"""Top up partially successful questions without rerunning zero-success cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import run_experiment as base


SELECTED_ROWS = (3, 7, 21, 22, 23)
PRIORITY_ROWS = (3, 7, 21, 22, 23)
CONCURRENCY = 4
MAX_NEW_FAILURES = 50
MIN_REMAINING_PERCENT = 30.0
QUOTA_CHECK_INTERVAL_SECONDS = 60
QUOTA_QUERY_TIMEOUT_SECONDS = 30
MAX_CONSECUTIVE_QUOTA_ERRORS = 3
HOURLY_REPORT_SECONDS = 60 * 60
SUPPLEMENT_STATE_PATH = base.REPORT_DIR / "partial_success_supplement.json"
SUPPLEMENT_REPORT_PATH = base.REPORT_DIR / "PARTIAL_SUCCESS_SUPPLEMENT.md"
HOURLY_REPORT_PATH = base.REPORT_DIR / "partial_success_supplement_hourly.jsonl"
FINAL_AUDIT_PATH = base.REPORT_DIR / "final_audit.json"


class QuotaClient:
    """Small stdio client for the documented Codex app-server rate-limit API."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr_tail: list[str] = []
        self.next_id = 10
        self.reader: threading.Thread | None = None
        self.stderr_reader: threading.Thread | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [str(base.CLI_PATH), "app-server", "--listen", "stdio://"],
            cwd=base.EXPERIMENT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.reader.start()
        self.stderr_reader.start()
        self._send(
            {
                "method": "initialize",
                "id": 1,
                "params": {
                    "clientInfo": {
                        "name": "ip_distill_quota_monitor",
                        "title": "IP Distillation Quota Monitor",
                        "version": "1.0.0",
                    }
                },
            }
        )
        response = self._wait_for(1, QUOTA_QUERY_TIMEOUT_SECONDS)
        if response.get("error"):
            raise RuntimeError(f"app-server initialize failed: {response['error']}")
        self._send({"method": "initialized", "params": {}})

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self.messages.put(message)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_tail.append(line.rstrip())
            del self.stderr_tail[:-50]

    def _send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("quota app-server is not running")
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _wait_for(self, request_id: int, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                detail = " | ".join(self.stderr_tail[-10:])
                raise RuntimeError(
                    f"quota app-server exited with {self.process.returncode}: {detail}"
                )
            try:
                message = self.messages.get(timeout=min(1.0, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if message.get("id") == request_id:
                return message
        raise TimeoutError(f"quota request {request_id} timed out")

    def read(self) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        self._send({"method": "account/rateLimits/read", "id": request_id})
        response = self._wait_for(request_id, QUOTA_QUERY_TIMEOUT_SECONDS)
        if response.get("error"):
            raise RuntimeError(f"rate limit query failed: {response['error']}")
        result = response.get("result") or {}
        primary_view = result.get("rateLimits") or {}
        limit_id = str(primary_view.get("limitId") or "codex")
        all_buckets = result.get("rateLimitsByLimitId") or {}
        bucket = all_buckets.get(limit_id) or primary_view
        windows: list[dict[str, Any]] = []
        for label in ("primary", "secondary"):
            value = bucket.get(label) if isinstance(bucket, dict) else None
            if not isinstance(value, dict) or value.get("usedPercent") is None:
                continue
            used = float(value["usedPercent"])
            windows.append(
                {
                    "name": label,
                    "used_percent": used,
                    "remaining_percent": 100.0 - used,
                    "window_duration_minutes": value.get("windowDurationMins"),
                    "resets_at": value.get("resetsAt"),
                }
            )
        if not windows:
            raise RuntimeError("rate limit response did not contain usedPercent")
        return {
            "observed_at": base.utc_now(),
            "limit_id": limit_id,
            "remaining_percent": min(item["remaining_percent"] for item in windows),
            "windows": windows,
        }

    def close(self) -> None:
        if self.process is None:
            return
        with contextlib.suppress(Exception):
            if self.process.stdin is not None:
                self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=10)
        if self.process.poll() is None:
            self.process.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected rows and query quota without changing experiment state.",
    )
    return parser.parse_args()


def selected_samples(state: dict[str, Any]) -> list[dict[str, Any]]:
    by_row = {int(item["row_index"]): item for item in state["samples"]}
    missing = [row for row in SELECTED_ROWS if row not in by_row]
    if missing:
        raise RuntimeError(f"selected rows missing from state: {missing}")
    result = [by_row[row] for row in PRIORITY_ROWS]
    invalid = [
        int(item["row_index"])
        for item in result
        if not 0 < int(item["accepted_count"]) < base.TARGET_CORRECT
    ]
    if invalid:
        raise RuntimeError(
            "supplement rows must have 1-9 accepted trajectories before first start: "
            + ", ".join(map(str, invalid))
        )
    return result


def new_supplement_state(state: dict[str, Any], quota: dict[str, Any]) -> dict[str, Any]:
    samples = selected_samples(state)
    return {
        "schema_version": "ip-distill-partial-success-supplement.v1",
        "status": "running",
        "started_at": base.utc_now(),
        "updated_at": base.utc_now(),
        "selected_rows": list(SELECTED_ROWS),
        "priority_rows": list(PRIORITY_ROWS),
        "concurrency": CONCURRENCY,
        "target_total_accepted": base.TARGET_CORRECT,
        "max_new_failures_per_row": MAX_NEW_FAILURES,
        "minimum_remaining_quota_percent": MIN_REMAINING_PERCENT,
        "quota_check_interval_seconds": QUOTA_CHECK_INTERVAL_SECONDS,
        "stop_reason": None,
        "quota_checks": [quota],
        "quota_error_count": 0,
        "last_quota": quota,
        "samples": {
            item["sample_key"]: {
                "row_index": int(item["row_index"]),
                "initial_status": item["status"],
                "initial_accepted_count": int(item["accepted_count"]),
                "initial_total_wrong": int(item["total_wrong"]),
                "initial_consecutive_wrong": int(item["consecutive_wrong"]),
                "initial_total_attempts": int(item["total_attempts"]),
                "initial_total_model_attempts": int(item["total_model_attempts"]),
                "initial_infrastructure_failures": int(item["infrastructure_failures"]),
                "new_accepted": 0,
                "new_failures": 0,
                "new_infrastructure_failures": 0,
                "new_attempts": 0,
                "status": "pending",
            }
            for item in samples
        },
    }


def persist_supplement(value: dict[str, Any]) -> None:
    value["updated_at"] = base.utc_now()
    base.atomic_json(SUPPLEMENT_STATE_PATH, value)


def append_hourly(snapshot: dict[str, Any]) -> None:
    HOURLY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HOURLY_REPORT_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")) + "\n")


def update_progress(state: dict[str, Any], supplement: dict[str, Any]) -> None:
    by_key = {item["sample_key"]: item for item in state["samples"]}
    for key, progress in supplement["samples"].items():
        sample = by_key[key]
        progress["new_accepted"] = (
            int(sample["accepted_count"]) - int(progress["initial_accepted_count"])
        )
        if supplement.get("counters_restored"):
            progress["new_failures"] = (
                int(sample["total_wrong"])
                - int(progress["initial_total_wrong"])
            )
        else:
            progress["new_failures"] = int(sample["total_wrong"])
        progress["new_infrastructure_failures"] = (
            int(sample["infrastructure_failures"])
            - int(progress["initial_infrastructure_failures"])
        )
        progress["new_attempts"] = (
            int(sample["total_attempts"]) - int(progress["initial_total_attempts"])
        )
        if int(sample["accepted_count"]) >= base.TARGET_CORRECT:
            progress["status"] = "completed_with_10_correct"
        elif int(progress["new_failures"]) >= MAX_NEW_FAILURES:
            progress["status"] = "stopped_after_50_new_failures"
        elif sample.get("current_attempt"):
            progress["status"] = "running"
        else:
            progress["status"] = "pending"


def snapshot(state: dict[str, Any], supplement: dict[str, Any], kind: str) -> dict[str, Any]:
    update_progress(state, supplement)
    by_key = {item["sample_key"]: item for item in state["samples"]}
    return {
        "kind": kind,
        "at": base.utc_now(),
        "status": supplement["status"],
        "stop_reason": supplement.get("stop_reason"),
        "quota": supplement.get("last_quota"),
        "rows": [
            {
                **supplement["samples"][key],
                "sample_key": key,
                "current_total_accepted": int(by_key[key]["accepted_count"]),
            }
            for key in supplement["samples"]
        ],
    }


def write_report(state: dict[str, Any], supplement: dict[str, Any], kind: str) -> None:
    current = snapshot(state, supplement, kind)
    lines = [
        "# Partial-success trajectory supplement",
        "",
        f"- Status: `{supplement['status']}`",
        f"- Started: `{supplement['started_at']}`",
        f"- Updated: `{supplement['updated_at']}`",
        f"- Stop reason: `{supplement.get('stop_reason')}`",
        f"- Concurrency: {CONCURRENCY}",
        f"- Stop after new failures per row: {MAX_NEW_FAILURES}",
        f"- Stop scheduling below remaining quota: {MIN_REMAINING_PERCENT}%",
        "",
    ]
    quota = supplement.get("last_quota")
    if quota:
        lines.extend(
            [
                "## Latest quota",
                "",
                f"- Observed: `{quota.get('observed_at')}`",
                f"- Limit bucket: `{quota.get('limit_id')}`",
                f"- Remaining: {quota.get('remaining_percent')}%",
                "",
            ]
        )
    lines.extend(
        [
            "## Selected rows",
            "",
            "| row | initial accepted | current accepted | added accepted | new failures | new infrastructure failures | new attempts | status |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in current["rows"]:
        lines.append(
            "| {row_index} | {initial_accepted_count} | {current_total_accepted} | "
            "{new_accepted} | {new_failures} | {new_infrastructure_failures} | "
            "{new_attempts} | {status} |".format(**item)
        )
    base.atomic_text(SUPPLEMENT_REPORT_PATH, "\n".join(lines) + "\n")


def configure_state_for_supplement(
    manifest: dict[str, Any],
    state: dict[str, Any],
    supplement: dict[str, Any],
) -> None:
    by_key = {item["sample_key"]: item for item in state["samples"]}
    if not supplement.get("counters_prepared"):
        for key, progress in supplement["samples"].items():
            sample = by_key[key]
            sample["total_wrong"] = 0
            sample["consecutive_wrong"] = 0
            sample["status"] = "pending"
            sample["current_attempt"] = None
            sample["updated_at"] = base.utc_now()
            progress["status"] = "pending"
        supplement["counters_prepared"] = True
    else:
        for key in supplement["samples"]:
            sample = by_key[key]
            if int(sample["accepted_count"]) >= base.TARGET_CORRECT:
                sample["status"] = "completed_with_10_correct"
            elif int(sample["total_wrong"]) >= MAX_NEW_FAILURES:
                sample["status"] = "abandoned_after_20_total_wrong"
            elif not sample.get("current_attempt"):
                sample["status"] = "pending"
    state["status"] = "running_supplement"
    state["current_concurrency"] = CONCURRENCY
    state["global_pause_until"] = None
    state["pause_reason"] = None
    state["updated_at"] = base.utc_now()
    state.setdefault("supplement_runs", []).append(
        {
            "started_at": supplement["started_at"],
            "state_path": base.relative(SUPPLEMENT_STATE_PATH),
        }
    )
    manifest["status"] = "running_supplement"
    manifest["updated_at"] = base.utc_now()
    manifest.setdefault("supplement_runs", []).append(
        {
            "started_at": supplement["started_at"],
            "selected_rows": list(SELECTED_ROWS),
            "concurrency": CONCURRENCY,
            "max_new_failures_per_row": MAX_NEW_FAILURES,
            "minimum_remaining_quota_percent": MIN_REMAINING_PERCENT,
        }
    )
    base.atomic_json(base.STATE_PATH, state)
    base.atomic_json(base.MANIFEST_PATH, manifest)
    base.write_accepted_index(state)
    persist_supplement(supplement)


def fixed_concurrency_record(state: dict[str, Any], new_value: int, reason: str) -> None:
    state.setdefault("supplement_concurrency_requests", []).append(
        {
            "at": base.utc_now(),
            "requested": int(new_value),
            "kept": CONCURRENCY,
            "reason": reason,
        }
    )
    state["current_concurrency"] = CONCURRENCY


def apply_result(
    state: dict[str, Any],
    supplement: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    blocker = base.apply_outcome(state, result)
    state["current_concurrency"] = CONCURRENCY
    update_progress(state, supplement)
    persist_supplement(supplement)
    return blocker


def quota_due(last_check_mono: float) -> bool:
    return time.monotonic() - last_check_mono >= QUOTA_CHECK_INTERVAL_SECONDS


def run_scheduler(
    manifest: dict[str, Any],
    state: dict[str, Any],
    supplement: dict[str, Any],
    safe_index: dict[str, Any],
    quota_client: QuotaClient,
) -> str | None:
    records = {record["sample_key"]: record for record in safe_index["records"]}
    template = base.LOCALIZED_TEMPLATE.read_text(encoding="utf-8")
    active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    last_quota_check = time.monotonic()
    last_heartbeat = 0.0
    last_hourly = time.monotonic()
    stop_reason: str | None = None
    by_row = {int(item["row_index"]): item for item in state["samples"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        while True:
            selected = [by_row[row] for row in PRIORITY_ROWS]
            terminal = all(
                int(item["accepted_count"]) >= base.TARGET_CORRECT
                or int(item["total_wrong"]) >= MAX_NEW_FAILURES
                for item in selected
            )
            if (terminal or stop_reason) and not active:
                break

            now_mono = time.monotonic()
            if quota_due(last_quota_check):
                last_quota_check = now_mono
                try:
                    quota = quota_client.read()
                    supplement["last_quota"] = quota
                    supplement["quota_checks"].append(quota)
                    supplement["quota_error_count"] = 0
                    if float(quota["remaining_percent"]) < MIN_REMAINING_PERCENT:
                        stop_reason = "quota_remaining_below_30_percent"
                except Exception as exc:
                    supplement["quota_error_count"] = int(
                        supplement.get("quota_error_count", 0)
                    ) + 1
                    supplement.setdefault("quota_errors", []).append(
                        {
                            "at": base.utc_now(),
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    if int(supplement["quota_error_count"]) >= MAX_CONSECUTIVE_QUOTA_ERRORS:
                        stop_reason = "quota_monitor_unavailable_after_3_checks"
                persist_supplement(supplement)

            if now_mono - last_heartbeat >= 60:
                base.heartbeat(state, active)
                write_report(state, supplement, "heartbeat")
                last_heartbeat = now_mono
            if now_mono - last_hourly >= HOURLY_REPORT_SECONDS:
                hourly = snapshot(state, supplement, "hourly")
                append_hourly(hourly)
                write_report(state, supplement, "hourly")
                print("HOURLY_REPORT " + json.dumps(hourly, ensure_ascii=False), flush=True)
                last_hourly = now_mono

            pause_until = state.get("global_pause_until")
            paused = bool(
                pause_until and base.parse_time(pause_until) > datetime.now(timezone.utc)
            )
            if pause_until and not paused:
                state["global_pause_until"] = None
                state["pause_reason"] = None
                base.atomic_json(base.STATE_PATH, state)

            quota_ok = supplement.get("last_quota") and float(
                supplement["last_quota"]["remaining_percent"]
            ) >= MIN_REMAINING_PERCENT
            monitor_ok = int(supplement.get("quota_error_count", 0)) == 0
            if not stop_reason and not paused and quota_ok and monitor_ok:
                running_keys = {task["sample_key"] for task in active.values()}
                candidates = [
                    by_row[row]
                    for row in PRIORITY_ROWS
                    if by_row[row]["status"] == "pending"
                    and by_row[row]["sample_key"] not in running_keys
                    and int(by_row[row]["accepted_count"]) < base.TARGET_CORRECT
                    and int(by_row[row]["total_wrong"]) < MAX_NEW_FAILURES
                ]
                capacity = CONCURRENCY - len(active)
                for sample in candidates[: max(0, capacity)]:
                    task = base.create_attempt(
                        state,
                        sample,
                        records[sample["sample_key"]],
                        template,
                    )
                    future = executor.submit(base.execute_attempt, task)
                    active[future] = task
                    base.log(
                        "supplement scheduled "
                        f"{task['sample_key']} success_slot={task['target_success_slot']:02d} "
                        f"attempt={task['attempt_index']:03d}"
                    )

            if not active:
                time.sleep(10 if paused or not monitor_ok else 1)
                continue

            done, _ = concurrent.futures.wait(
                active,
                timeout=10,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                reserved = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **reserved,
                        "outcome": "infrastructure_failure",
                        "error_class": "runner_worker_exception",
                        "model_launched": False,
                        "thread_id": None,
                    }
                    metadata = base.load_json(reserved["attempt_dir"] / "metadata.json")
                    metadata.update(
                        {
                            "status": "infrastructure_failure",
                            "generation_status": "worker_exception",
                            "judge_status": "not_run",
                            "error_class": "runner_worker_exception",
                            "ended_at": base.utc_now(),
                            "runner_exception_type": type(exc).__name__,
                        }
                    )
                    base.atomic_json(reserved["attempt_dir"] / "metadata.json", metadata)
                if reserved["workspace"].exists():
                    shutil.rmtree(reserved["workspace"])
                base.log(
                    "supplement finished "
                    f"{result['sample_key']} attempt={result['attempt_index']:03d} "
                    f"outcome={result['outcome']} error={result.get('error_class')}"
                )
                outcome_blocker = apply_result(state, supplement, result)
                if result.get("error_class") == "quota_exhausted":
                    stop_reason = "quota_exhausted"
                elif outcome_blocker:
                    stop_reason = outcome_blocker

    return stop_reason


def restore_counters_and_finalize(
    manifest: dict[str, Any],
    state: dict[str, Any],
    supplement: dict[str, Any],
    stop_reason: str | None,
) -> int:
    update_progress(state, supplement)
    by_key = {item["sample_key"]: item for item in state["samples"]}
    all_threshold_terminal = True
    for key, progress in supplement["samples"].items():
        sample = by_key[key]
        new_failures = int(sample["total_wrong"])
        progress["new_failures"] = new_failures
        sample["total_wrong"] = int(progress["initial_total_wrong"]) + new_failures
        sample["current_attempt"] = None
        if int(sample["accepted_count"]) >= base.TARGET_CORRECT:
            sample["status"] = "completed_with_10_correct"
            progress["status"] = "completed_with_10_correct"
        elif new_failures >= MAX_NEW_FAILURES:
            sample["status"] = "abandoned_after_20_total_wrong"
            progress["status"] = "stopped_after_50_new_failures"
        else:
            sample["status"] = "stopped_by_infrastructure_blocker"
            progress["status"] = "stopped_before_target"
            all_threshold_terminal = False
        sample["updated_at"] = base.utc_now()

    completed = stop_reason is None and all_threshold_terminal
    supplement["status"] = "completed" if completed else "stopped"
    supplement["stop_reason"] = stop_reason
    supplement["ended_at"] = base.utc_now()
    supplement["counters_restored"] = True
    state["status"] = "completed" if completed else "stopped_by_infrastructure_blocker"
    state["updated_at"] = base.utc_now()
    state["supplement_ended_at"] = supplement["ended_at"]
    state["global_pause_until"] = None
    state["pause_reason"] = None
    if stop_reason:
        state["blocker_reason"] = "supplement:" + stop_reason
    else:
        state.pop("blocker_reason", None)
    manifest["status"] = "completed" if completed else "stopped"
    manifest["updated_at"] = base.utc_now()
    manifest["ended_at"] = supplement["ended_at"]
    base.atomic_json(base.STATE_PATH, state)
    base.atomic_json(base.MANIFEST_PATH, manifest)
    base.write_accepted_index(state)
    persist_supplement(supplement)
    base.write_final_report(manifest, state)

    audit = subprocess.run(
        [sys.executable, "-B", str(base.SCRIPTS_DIR / "final_audit.py")],
        cwd=base.EXPERIMENT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30 * 60,
        check=False,
    )
    supplement["final_audit"] = {
        "return_code": audit.returncode,
        "stdout": audit.stdout.strip(),
        "stderr": audit.stderr.strip(),
        "passed": bool(
            audit.returncode == 0
            and FINAL_AUDIT_PATH.exists()
            and base.load_json(FINAL_AUDIT_PATH).get("passed") is True
        ),
    }
    persist_supplement(supplement)
    final_snapshot = snapshot(state, supplement, "final")
    append_hourly(final_snapshot)
    write_report(state, supplement, "final")
    print("FINAL_REPORT " + json.dumps(final_snapshot, ensure_ascii=False), flush=True)
    lock_status = "supplement_completed" if completed else "supplement_stopped"
    base.finish_runner_lock(lock_status)
    return 0 if completed and supplement["final_audit"]["passed"] else 2


def validate_dry_run(state: dict[str, Any], quota: dict[str, Any]) -> int:
    samples = selected_samples(state)
    result = {
        "selected": [
            {
                "row": item["row_index"],
                "sample_key": item["sample_key"],
                "accepted": item["accepted_count"],
                "needed": base.TARGET_CORRECT - int(item["accepted_count"]),
                "status": item["status"],
            }
            for item in samples
        ],
        "concurrency": CONCURRENCY,
        "max_new_failures_per_row": MAX_NEW_FAILURES,
        "quota": quota,
        "may_start": float(quota["remaining_percent"]) >= MIN_REMAINING_PERCENT,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["may_start"] else 3


def main() -> int:
    args = parse_args()
    base.locate_and_copy_codex()
    quota_client = QuotaClient()
    quota_client.start()
    try:
        initial_quota = quota_client.read()
        state = base.load_json(base.STATE_PATH)
        if args.dry_run:
            return validate_dry_run(state, initial_quota)
        base.acquire_runner_lock()
        safe_index = base.run_input_broker()
        manifest, state = base.initialize_or_validate(safe_index)
        if SUPPLEMENT_STATE_PATH.exists():
            raise RuntimeError(
                "supplement state already exists; inspect it before attempting a restart"
            )
        supplement = new_supplement_state(state, initial_quota)
        if float(initial_quota["remaining_percent"]) < MIN_REMAINING_PERCENT:
            supplement["status"] = "stopped"
            supplement["stop_reason"] = "initial_quota_remaining_below_30_percent"
            persist_supplement(supplement)
            write_report(state, supplement, "final")
            base.finish_runner_lock("supplement_not_started_low_quota")
            return 3

        configure_state_for_supplement(manifest, state, supplement)
        base.MAX_CONSECUTIVE_WRONG = 10**9
        base.MAX_TOTAL_WRONG = MAX_NEW_FAILURES
        base.record_concurrency = fixed_concurrency_record
        stop_reason = run_scheduler(
            manifest, state, supplement, safe_index, quota_client
        )
        return restore_counters_and_finalize(
            manifest, state, supplement, stop_reason
        )
    except KeyboardInterrupt:
        if SUPPLEMENT_STATE_PATH.exists():
            supplement = base.load_json(SUPPLEMENT_STATE_PATH)
            supplement["status"] = "interrupted"
            supplement["stop_reason"] = "keyboard_interrupt"
            persist_supplement(supplement)
        base.finish_runner_lock("supplement_interrupted")
        return 130
    except Exception as exc:
        if SUPPLEMENT_STATE_PATH.exists():
            supplement = base.load_json(SUPPLEMENT_STATE_PATH)
            supplement["status"] = "failed"
            supplement["stop_reason"] = f"{type(exc).__name__}: {exc}"
            persist_supplement(supplement)
        with contextlib.suppress(Exception):
            base.finish_runner_lock("supplement_failed")
        raise
    finally:
        quota_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
