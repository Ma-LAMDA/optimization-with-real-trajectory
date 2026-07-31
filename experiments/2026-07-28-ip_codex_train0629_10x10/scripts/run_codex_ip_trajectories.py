#!/usr/bin/env python3
"""Instantiate selected IP questions, run Codex, and save full JSONL traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def find_repository_root(start: Path) -> Path:
    """Locate the repository root so Codex can query the shared saved_configs."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repository root from {start}")


ROOT = find_repository_root(EXPERIMENT_ROOT)
DEFAULT_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
DEFAULT_TEMPLATE = ROOT / "data" / "simulation" / "IP user prompt.txt"
LOCAL_TEMPLATE = EXPERIMENT_ROOT / "inputs" / "IP user prompt local-url-only.txt"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "runs"
QUESTIONS_DIR = EXPERIMENT_ROOT / "results" / "questions"
DEFAULT_CASE_IDS = (13, 14, 17, 18, 87, 88, 91, 92, 93, 94)
DEFAULT_REPEATS = 10
DEFAULT_CREDIT_RETRY_SECONDS = 30 * 60
DEFAULT_RUN_NAME = "fullaccess"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_CODEX = EXPERIMENT_ROOT / "runtime" / "codex.exe"
WORKSPACE = EXPERIMENT_ROOT / "runtime" / "workspace"
REMOTE_SERVICE_HOST = "10.139.194.154"
LOCAL_SERVICE_HOST = "127.0.0.1"
SERVICE_BASE_URL = "http://127.0.0.1:3080"
REQUIRED_RECORD_FIELDS = ("id", "question", "output_format")
PLACEHOLDERS = {
    "{original_query}": "question",
    "{output_format}": "output_format",
}
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CREDIT_LIMIT_PATTERNS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient credits",
    "not enough credits",
    "out of credits",
    "credit balance",
    "credit limit",
    "credits exhausted",
    "no credits",
    "add credits",
    "purchase additional credits",
    "usage limit",
    "usage_limit",
    "hit your usage limit",
    "reached your usage limit",
    "usage cap",
    "limit reached",
    "rate limit exceeded",
    "rate_limit_exceeded",
    "quota exceeded",
    "billing hard limit",
    "exceeded your current quota",
    "额度不足",
    "积分不足",
    "用量限制",
    "达到使用上限",
    "余额不足",
    "配额不足",
)


class RunError(RuntimeError):
    """Raised when a Codex case cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_CASE_IDS),
        metavar="ID",
        help="Record IDs to run, in execution order.",
    )
    parser.add_argument(
        "--codex-bin",
        default=os.environ.get("CODEX_BIN", str(DEFAULT_CODEX)),
        help="Codex CLI executable (default: runtime/codex.exe).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Codex model override (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="danger-full-access",
        help="Codex sandbox mode (default: danger-full-access for localhost HTTP).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Required successful runs per case (default: 10).",
    )
    parser.add_argument(
        "--credit-retry-seconds",
        type=int,
        default=DEFAULT_CREDIT_RETRY_SECONDS,
        help="Wait before retrying the same run after a credit/usage limit (default: 1800).",
    )
    parser.add_argument(
        "--credit-error-pattern",
        action="append",
        default=[],
        metavar="TEXT",
        help="Additional case-insensitive text that identifies a credit-limit error.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"Output subdirectory name (default: {DEFAULT_RUN_NAME}).",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume an interrupted run directory using its manifest checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only instantiate prompts and metadata; do not invoke Codex.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_dataset(path: Path) -> dict[int, tuple[int, dict[str, Any]]]:
    rows: dict[int, tuple[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            for field in REQUIRED_RECORD_FIELDS:
                if field not in row:
                    raise ValueError(
                        f"{path}:{line_number}: missing required field {field!r}"
                    )
            identifier = row["id"]
            if not isinstance(identifier, int) or isinstance(identifier, bool):
                raise ValueError(
                    f"{path}:{line_number}: id must be an integer, got {identifier!r}"
                )
            if identifier in rows:
                first_line = rows[identifier][0]
                raise ValueError(
                    f"{path}:{line_number}: duplicate id {identifier}; "
                    f"first seen on line {first_line}"
                )
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"{path}:{line_number}: question must be non-empty")
            if (
                not isinstance(row["output_format"], str)
                or not row["output_format"].strip()
            ):
                raise ValueError(
                    f"{path}:{line_number}: output_format must be non-empty"
                )
            rows[identifier] = (line_number, row)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def select_records(
    rows: dict[int, tuple[int, dict[str, Any]]],
    case_ids: list[int],
) -> list[tuple[int, dict[str, Any]]]:
    if not case_ids:
        raise ValueError("At least one case ID is required")
    duplicates = [
        identifier
        for identifier, count in Counter(case_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate requested case IDs: {duplicates}")
    missing = [identifier for identifier in case_ids if identifier not in rows]
    if missing:
        raise ValueError(f"Requested case IDs are absent from the dataset: {missing}")
    return [rows[identifier] for identifier in case_ids]


def instantiate(template: str, row: dict[str, Any]) -> str:
    prompt = template
    for placeholder, field in PLACEHOLDERS.items():
        count = prompt.count(placeholder)
        if count != 1:
            raise ValueError(
                f"Template must contain {placeholder!r} exactly once; found {count}"
            )
        prompt = prompt.replace(placeholder, row[field])
    unresolved = [placeholder for placeholder in PLACEHOLDERS if placeholder in prompt]
    if unresolved:
        raise ValueError(f"Unresolved template placeholders: {unresolved}")
    return prompt


def localize_template(source: str) -> str:
    """Change only the original remote service host to the local service host."""
    occurrences = source.count(REMOTE_SERVICE_HOST)
    if occurrences != 2:
        raise ValueError(
            "Original IP prompt must contain the remote service host exactly twice; "
            f"found {occurrences}"
        )
    localized = source.replace(REMOTE_SERVICE_HOST, LOCAL_SERVICE_HOST)
    if REMOTE_SERVICE_HOST in localized:
        raise ValueError("Remote service host remains in localized prompt")
    return localized


def save_localized_template(template: str) -> None:
    LOCAL_TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_TEMPLATE.is_file():
        if LOCAL_TEMPLATE.read_text(encoding="utf-8") != template:
            raise ValueError(f"Existing localized template differs: {LOCAL_TEMPLATE}")
        return
    LOCAL_TEMPLATE.write_text(template, encoding="utf-8", newline="\n")


def require_service() -> None:
    request = urllib.request.Request(
        f"{SERVICE_BASE_URL}/healthz",
        headers={"User-Agent": "codex-ip-trajectory-runner/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            server = response.headers.get("Server", "")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RunError(f"Saved configs service is unavailable: {exc}") from exc
    if payload != {"status": "ok"}:
        raise RunError(f"Unexpected service health response: {payload!r}")
    if "SavedConfigsService/" not in server:
        raise RunError(f"Unexpected service identity: {server!r}")


def resolve_codex(value: str) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_absolute() or supplied.parent != Path("."):
        if not supplied.is_file():
            raise FileNotFoundError(f"Codex CLI does not exist: {supplied}")
        return supplied.resolve()
    found = shutil.which(value)
    if not found:
        raise FileNotFoundError(
            "Codex CLI was not found on PATH. Install/authenticate the Codex CLI, "
            "set CODEX_BIN, or pass --codex-bin with its executable path."
        )
    return Path(found).resolve()


def codex_command(
    executable: Path,
    final_answer: Path,
    model: str | None,
    sandbox: str,
) -> list[str]:
    trust_override = (
        f"projects.'{WORKSPACE.resolve()}'.trust_level=\"trusted\""
    )
    command = [str(executable)]
    command.extend(
        [
            "exec",
            "--ephemeral",
            "--ignore-rules",
            "--dangerously-bypass-hook-trust",
            "--enable",
            "hooks",
            "--skip-git-repo-check",
            "--cd",
            str(WORKSPACE.resolve()),
            "-c",
            trust_override,
            "-c",
            'web_search="disabled"',
            "--json",
            "--sandbox",
            sandbox,
            "--output-last-message",
            str(final_answer.resolve()),
        ]
    )
    if model:
        command.extend(["--model", model])
    # The complete prompt is sent on stdin to avoid shell quoting and Windows
    # command-length limits.
    command.append("-")
    return command


def base_case_metadata(
    identifier: int,
    repeat_index: int,
    attempt_index: int,
    line_number: int,
    row: dict[str, Any],
    prompt_path: Path,
    record_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "codex-ip-trajectory-case.v1",
        "case_id": identifier,
        "repeat_index": repeat_index,
        "attempt_index": attempt_index,
        "source_line_number": line_number,
        "scenario": row.get("scenario"),
        "scenario_id": row.get("scenario_id"),
        "R3_level": row.get("R3_level"),
        "service_base_url": SERVICE_BASE_URL,
        "prompt_localization": {
            "source": label(DEFAULT_TEMPLATE),
            "replacement": {
                "from": REMOTE_SERVICE_HOST,
                "to": LOCAL_SERVICE_HOST,
            },
        },
        "files": {
            "prompt": "../prompt.txt",
            "source_record": "../source_record.json",
            "events": "events.jsonl",
            "stderr": "stderr.log",
            "final_answer": "final_answer.txt",
        },
        "started_at": now(),
    }


def prepare_slot(
    run_dir: Path,
    repeat_index: int,
    line_number: int,
    row: dict[str, Any],
    prompt: str,
) -> Path:
    identifier = row["id"]
    slot_dir = run_dir / f"q{identifier:04d}_r{repeat_index:02d}"
    slot_dir.mkdir(exist_ok=True)
    question_dir = QUESTIONS_DIR / f"q{identifier:04d}"
    question_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = question_dir / "prompt.txt"
    record_path = question_dir / "source_record.json"
    if prompt_path.is_file():
        if prompt_path.read_text(encoding="utf-8") != prompt:
            raise ValueError(f"Existing prompt differs during resume: {prompt_path}")
    else:
        prompt_path.write_text(prompt, encoding="utf-8", newline="\n")
    if record_path.is_file():
        existing_record = json.loads(record_path.read_text(encoding="utf-8"))
        if existing_record != row:
            raise ValueError(
                f"Existing source record differs during resume: {record_path}"
            )
    else:
        write_json(record_path, row)
    return slot_dir


def prepare_attempt(
    slot_dir: Path,
    repeat_index: int,
    attempt_index: int,
    line_number: int,
    row: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    attempt_dir = slot_dir / f"attempt_{attempt_index:03d}"
    attempt_dir.mkdir()
    question_dir = QUESTIONS_DIR / f"q{row['id']:04d}"
    prompt_path = question_dir / "prompt.txt"
    record_path = question_dir / "source_record.json"
    metadata = base_case_metadata(
        row["id"],
        repeat_index,
        attempt_index,
        line_number,
        row,
        prompt_path,
        record_path,
    )
    metadata["sha256"] = {
        "prompt": digest(prompt_path),
        "source_record": digest(record_path),
    }
    metadata["files"]["prompt"] = Path(
        os.path.relpath(prompt_path, attempt_dir)
    ).as_posix()
    metadata["files"]["source_record"] = Path(
        os.path.relpath(record_path, attempt_dir)
    ).as_posix()
    return attempt_dir, metadata


def credit_limit_matches(
    events_path: Path,
    stderr_path: Path,
    event_errors: list[dict[str, Any]],
    launch_error: str | None,
    extra_patterns: list[str],
) -> list[str]:
    parts = [json.dumps(event_errors, ensure_ascii=False), launch_error or ""]
    if events_path.is_file():
        parts.append(events_path.read_text(encoding="utf-8", errors="replace"))
    if stderr_path.is_file():
        parts.append(stderr_path.read_text(encoding="utf-8", errors="replace"))
    failure_text = "\n".join(parts).casefold()
    patterns = [
        pattern.casefold().strip()
        for pattern in (*CREDIT_LIMIT_PATTERNS, *extra_patterns)
        if pattern.strip()
    ]
    return sorted({pattern for pattern in patterns if pattern in failure_text})


def command_policy_violations(events_path: Path) -> list[dict[str, Any]]:
    """Find commands that directly name repository or saved-config paths."""
    if not events_path.is_file():
        return []
    violations: list[dict[str, Any]] = []
    seen: set[str] = set()
    with events_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item") if isinstance(event, dict) else None
            if not isinstance(item, dict) or item.get("type") != "command_execution":
                continue
            command = item.get("command")
            if not isinstance(command, str) or command in seen:
                continue
            seen.add(command)
            folded = command.casefold()
            reasons = []
            if "saved_configs" in folded:
                reasons.append("saved_configs path named")
            if str(ROOT).casefold() in folded:
                reasons.append("repository path named")
            if "../" in command or "..\\" in command:
                reasons.append("parent traversal")
            if reasons:
                violations.append(
                    {
                        "line": line_number,
                        "command": command,
                        "reasons": reasons,
                    }
                )
    return violations


def invalidate_existing_policy_violations(
    manifest: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Reject prior successes that bypassed the local API without altering traces."""
    invalidated: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        if run.get("status") != "succeeded":
            continue
        attempt_index = run.get("successful_attempt")
        if not isinstance(attempt_index, int):
            continue
        attempt_dir = (
            run_dir
            / str(run["directory"])
            / f"attempt_{attempt_index:03d}"
        )
        violations = command_policy_violations(attempt_dir / "events.jsonl")
        if not violations:
            continue
        audit = {
            "schema_version": "codex-ip-trajectory-policy-audit.v1",
            "audited_at": now(),
            "status": "rejected",
            "reason": "direct filesystem access bypassed the local API",
            "violations": violations,
        }
        write_json(attempt_dir / "policy_audit.json", audit)
        for attempt in run.get("attempts", []):
            if attempt.get("attempt_index") == attempt_index:
                attempt["status"] = "rejected_policy_violation"
                attempt["failure_kind"] = "prohibited_filesystem_access"
        run["status"] = "pending"
        run["successful_attempt"] = None
        write_json(
            run_dir / str(run["directory"]) / "run.json",
            {
                "schema_version": "codex-ip-trajectory-slot.v1",
                "case_id": run["case_id"],
                "repeat_index": run["repeat_index"],
                "status": "pending_after_policy_rejection",
                "successful_attempt": None,
                "attempts": run.get("attempts", []),
            },
        )
        invalidated.append(
            {
                "case_id": run["case_id"],
                "repeat_index": run["repeat_index"],
                "attempt_index": attempt_index,
                "directory": run["directory"],
            }
        )
    return invalidated


def record_orphaned_attempts(
    manifest: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Record attempt directories left active when the prior runner was stopped."""
    recovered: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        slot_dir = run_dir / str(run["directory"])
        recorded = {
            int(attempt.get("attempt_index", 0))
            for attempt in run.get("attempts", [])
        }
        for attempt_dir in sorted(slot_dir.glob("attempt_[0-9][0-9][0-9]")):
            attempt_index = int(attempt_dir.name.removeprefix("attempt_"))
            if attempt_index in recorded:
                continue
            metadata_path = attempt_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            entry = manifest_attempt(metadata)
            entry["status"] = "interrupted"
            entry["failure_kind"] = "interrupted_for_systemic_policy_audit"
            run.setdefault("attempts", []).append(entry)
            write_json(
                attempt_dir / "recovery.json",
                {
                    "schema_version": "codex-ip-trajectory-recovery.v1",
                    "recorded_at": now(),
                    "status": "interrupted",
                    "reason": "runner stopped after systemic API-only policy violation was detected",
                    "original_metadata_status": metadata.get("status"),
                },
            )
            recovered.append(
                {
                    "case_id": run["case_id"],
                    "repeat_index": run["repeat_index"],
                    "attempt_index": attempt_index,
                    "directory": run["directory"],
                }
            )
        if run.get("status") == "running":
            run["status"] = "pending"
    return recovered


def run_case(
    case_dir: Path,
    metadata: dict[str, Any],
    prompt: str,
    executable: Path,
    model: str | None,
    sandbox: str,
    extra_credit_patterns: list[str],
) -> dict[str, Any]:
    events_path = case_dir / "events.jsonl"
    stderr_path = case_dir / "stderr.log"
    final_path = case_dir / "final_answer.txt"
    metadata_path = case_dir / "metadata.json"
    command = codex_command(executable, final_path, model, sandbox)
    metadata["command"] = command
    metadata["working_directory"] = str(WORKSPACE)
    metadata["codex_executable"] = str(executable)
    metadata["model_override"] = model
    metadata["sandbox"] = sandbox
    metadata["status"] = "running"
    write_json(metadata_path, metadata)

    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    event_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    event_errors: list[dict[str, Any]] = []
    invalid_events: list[dict[str, Any]] = []
    successful_command_executions = 0
    failed_command_executions = 0
    process: subprocess.Popen[str] | None = None
    interrupted = False
    launch_error: str | None = None
    started = time.monotonic()

    try:
        with (
            events_path.open("w", encoding="utf-8", newline="\n") as events_handle,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdin is None or process.stdout is None:
                raise RunError("Failed to open Codex stdin/stdout pipes")
            process.stdin.write(prompt)
            process.stdin.close()

            for event_line_number, line in enumerate(process.stdout, start=1):
                events_handle.write(line)
                events_handle.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    invalid_events.append(
                        {"line": event_line_number, "error": str(exc)}
                    )
                    continue
                if not isinstance(event, dict):
                    invalid_events.append(
                        {
                            "line": event_line_number,
                            "error": "event is not a JSON object",
                        }
                    )
                    continue
                event_type = str(event.get("type", "unknown"))
                event_counts[event_type] += 1
                if event_type == "thread.started" and isinstance(
                    event.get("thread_id"), str
                ):
                    thread_id = event["thread_id"]
                if event_type == "turn.completed" and isinstance(
                    event.get("usage"), dict
                ):
                    usage = event["usage"]
                if event_type in {"error", "turn.failed"}:
                    event_errors.append(
                        {"line": event_line_number, "event": event}
                    )
                item = event.get("item")
                if isinstance(item, dict) and isinstance(item.get("type"), str):
                    item_counts[item["type"]] += 1
                    if (
                        event_type == "item.completed"
                        and item["type"] == "command_execution"
                    ):
                        if item.get("status") == "completed":
                            successful_command_executions += 1
                        elif item.get("status") == "failed":
                            failed_command_executions += 1
            return_code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return_code = process.returncode if process is not None else None
    except (OSError, RunError) as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        return_code = process.returncode if process is not None else None

    duration = round(time.monotonic() - started, 3)
    policy_violations = command_policy_violations(events_path)
    final_answer_valid = (
        final_path.is_file()
        and "<result>" in final_path.read_text(encoding="utf-8", errors="replace")
        and "</result>" in final_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )
    success = (
        not interrupted
        and launch_error is None
        and return_code == 0
        and events_path.is_file()
        and events_path.stat().st_size > 0
        and not invalid_events
        and not event_errors
        and not policy_violations
        and successful_command_executions > 0
        and final_answer_valid
        and thread_id is not None
    )
    credit_matches = (
        []
        if success or interrupted
        else credit_limit_matches(
            events_path,
            stderr_path,
            event_errors,
            launch_error,
            extra_credit_patterns,
        )
    )
    metadata.update(
        {
            "status": (
                "interrupted"
                if interrupted
                else "succeeded"
                if success
                else "failed"
            ),
            "ended_at": now(),
            "duration_seconds": duration,
            "exit_code": return_code,
            "thread_id": thread_id,
            "usage": usage,
            "event_type_counts": dict(sorted(event_counts.items())),
            "item_type_counts": dict(sorted(item_counts.items())),
            "successful_command_executions": successful_command_executions,
            "failed_command_executions": failed_command_executions,
            "final_answer_valid": final_answer_valid,
            "error_events": event_errors,
            "invalid_jsonl_events": invalid_events,
            "launch_error": launch_error,
            "policy_violations": policy_violations,
            "failure_kind": (
                None
                if success
                else "interrupted"
                if interrupted
                else "credits_exhausted"
                if credit_matches
                else "prohibited_filesystem_access"
                if policy_violations
                else "other"
            ),
            "credit_limit_matches": credit_matches,
        }
    )
    hashes = metadata["sha256"]
    for name, path in (
        ("events", events_path),
        ("stderr", stderr_path),
        ("final_answer", final_path),
    ):
        if path.is_file():
            hashes[name] = digest(path)
    write_json(metadata_path, metadata)

    if interrupted:
        raise KeyboardInterrupt
    return metadata


def manifest_attempt(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_index": metadata["attempt_index"],
        "status": metadata["status"],
        "failure_kind": metadata.get("failure_kind"),
        "thread_id": metadata.get("thread_id"),
        "exit_code": metadata.get("exit_code"),
        "duration_seconds": metadata.get("duration_seconds"),
        "directory": f"attempt_{metadata['attempt_index']:03d}",
    }


def slot_entry(identifier: int, repeat_index: int) -> dict[str, Any]:
    return {
        "case_id": identifier,
        "repeat_index": repeat_index,
        "status": "pending",
        "directory": f"q{identifier:04d}_r{repeat_index:02d}",
        "successful_attempt": None,
        "attempts": [],
    }


def refresh_summary(manifest: dict[str, Any]) -> None:
    statuses = Counter(run["status"] for run in manifest["runs"])
    manifest["summary"] = {
        "required": len(manifest["runs"]),
        "succeeded": statuses["succeeded"],
        "prepared": statuses["prepared"],
        "pending": statuses["pending"],
        "running": statuses["running"],
        "waiting_for_credits": statuses["waiting_for_credits"],
        "failed": statuses["failed"],
        "interrupted": statuses["interrupted"],
        "codex_attempts": sum(len(run["attempts"]) for run in manifest["runs"]),
    }


def main() -> int:
    arguments = parse_args()
    automatic_resume = DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_NAME / "manifest.json"
    if (
        not arguments.resume_run
        and not arguments.dry_run
        and arguments.output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
        and arguments.run_name == DEFAULT_RUN_NAME
        and automatic_resume.is_file()
    ):
        arguments.resume_run = automatic_resume.parent
    resumed_manifest: dict[str, Any] | None = None
    resumed_run_dir: Path | None = None
    if arguments.resume_run:
        if arguments.dry_run:
            raise ValueError("--resume-run cannot be combined with --dry-run")
        resumed_run_dir = arguments.resume_run.resolve()
        if resumed_run_dir.is_file():
            if resumed_run_dir.name != "manifest.json":
                raise ValueError("--resume-run file must be a manifest.json")
            resumed_run_dir = resumed_run_dir.parent
        resume_manifest_path = resumed_run_dir / "manifest.json"
        resumed_manifest = json.loads(
            resume_manifest_path.read_text(encoding="utf-8")
        )
        if resumed_manifest.get("schema_version") != "codex-ip-trajectory-run.v2":
            raise ValueError(f"Unsupported resume manifest: {resume_manifest_path}")
        if resumed_manifest.get("dry_run"):
            raise ValueError("A dry-run manifest cannot be resumed as a live run")
        arguments.case_ids = list(resumed_manifest["requested_case_ids"])
        arguments.repeats = int(resumed_manifest["repeats_per_case"])
        retry_policy = resumed_manifest.get("credit_retry", {})
        arguments.credit_retry_seconds = int(
            retry_policy.get(
                "wait_seconds",
                DEFAULT_CREDIT_RETRY_SECONDS,
            )
        )
        arguments.credit_error_pattern = list(
            retry_policy.get("extra_error_patterns", [])
        )
        arguments.sandbox = str(resumed_manifest.get("sandbox", "read-only"))
        arguments.model = str(resumed_manifest.get("model", DEFAULT_MODEL))

    if arguments.repeats <= 0:
        raise ValueError("--repeats must be greater than zero")
    if arguments.credit_retry_seconds < 0:
        raise ValueError("--credit-retry-seconds cannot be negative")

    dataset = arguments.dataset.resolve()
    template_path = arguments.template.resolve()
    output_root = arguments.output_root.resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(dataset)
    selected = select_records(rows, arguments.case_ids)
    source_template = template_path.read_text(encoding="utf-8")
    template = localize_template(source_template)
    save_localized_template(template)
    prompts = {
        row["id"]: (line_number, row, instantiate(template, row))
        for line_number, row in selected
    }
    schedule = [
        (repeat_index, *prompts[identifier])
        for repeat_index in range(1, arguments.repeats + 1)
        for identifier in arguments.case_ids
    ]

    executable: Path | None = None
    if not arguments.dry_run:
        require_service()
        executable = resolve_codex(arguments.codex_bin)

    if resumed_manifest is not None and resumed_run_dir is not None:
        run_dir = resumed_run_dir
        manifest_path = run_dir / "manifest.json"
        manifest = resumed_manifest
        if manifest.get("dataset_sha256") != digest(dataset):
            raise ValueError("Dataset hash differs from the resumed run")
        if manifest.get("source_template_sha256") != digest(template_path):
            raise ValueError("Source template hash differs from the resumed run")
        if manifest.get("localized_template_sha256") != digest(LOCAL_TEMPLATE):
            raise ValueError("Localized template hash differs from the resumed run")
        expected_slots = [
            (row["id"], repeat_index)
            for repeat_index, _line_number, row, _prompt in schedule
        ]
        actual_slots = [
            (run.get("case_id"), run.get("repeat_index"))
            for run in manifest.get("runs", [])
        ]
        if actual_slots != expected_slots:
            raise ValueError("Resume manifest schedule does not match its inputs")
        recovered = record_orphaned_attempts(manifest, run_dir)
        if recovered:
            manifest.setdefault("recovery_events", []).append(
                {
                    "recovered_at": now(),
                    "interrupted_attempts": recovered,
                }
            )
        invalidated = invalidate_existing_policy_violations(manifest, run_dir)
        if invalidated:
            manifest.setdefault("policy_audits", []).append(
                {
                    "audited_at": now(),
                    "invalidated_runs": invalidated,
                }
            )
            refresh_summary(manifest)
            write_json(manifest_path, manifest)
            print(
                f"Policy audit invalidated {len(invalidated)} prior success(es)",
                file=sys.stderr,
                flush=True,
            )
        manifest.update(
            {
                "ignore_user_config": False,
                "ignore_rules": True,
                "hooks_enabled": True,
                "hook_trust_bypassed_after_external_review": True,
                "api_only_hook": label(
                    WORKSPACE / ".codex" / "hooks" / "api_only_hook.py"
                ),
            }
        )
        if manifest.get("status") == "succeeded":
            print(f"Run is already complete: {run_dir}")
            return 0
        manifest.setdefault("resume_events", []).append({"resumed_at": now()})
        manifest["status"] = "running"
        manifest.pop("ended_at", None)
        refresh_summary(manifest)
        write_json(manifest_path, manifest)
    else:
        started_at = now()
        run_name = arguments.run_name
        if not RUN_NAME.fullmatch(run_name) or run_name in {".", ".."}:
            raise ValueError(
                "run name must contain only letters, digits, '.', '_' and '-', "
                "and must start with a letter or digit"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = output_root / run_name
        run_dir.mkdir()
        manifest_path = run_dir / "manifest.json"
        manifest = {
            "schema_version": "codex-ip-trajectory-run.v2",
            "status": "preparing" if arguments.dry_run else "running",
            "started_at": started_at,
            "repository_root": str(ROOT),
            "dataset": label(dataset),
            "dataset_sha256": digest(dataset),
            "source_template": label(template_path),
            "source_template_sha256": digest(template_path),
            "localized_template": label(LOCAL_TEMPLATE),
            "localized_template_sha256": digest(LOCAL_TEMPLATE),
            "prompt_localization": {
                "operation": "literal host replacement only",
                "occurrences": source_template.count(REMOTE_SERVICE_HOST),
                "from": REMOTE_SERVICE_HOST,
                "to": LOCAL_SERVICE_HOST,
            },
            "service_base_url": SERVICE_BASE_URL,
            "model": arguments.model,
            "codex_executable": label(Path(arguments.codex_bin)),
            "codex_version": "codex-cli 0.146.0-alpha.3.1",
            "working_directory": label(WORKSPACE),
            "ephemeral_sessions": True,
            "ignore_user_config": False,
            "ignore_rules": True,
            "hooks_enabled": True,
            "hook_trust_bypassed_after_external_review": True,
            "api_only_hook": label(
                WORKSPACE / ".codex" / "hooks" / "api_only_hook.py"
            ),
            "requested_case_ids": arguments.case_ids,
            "repeats_per_case": arguments.repeats,
            "required_successful_trajectories": len(schedule),
            "dry_run": arguments.dry_run,
            "sandbox": arguments.sandbox,
            "credit_retry": {
                "wait_seconds": arguments.credit_retry_seconds,
                "maximum_retries": None,
                "built_in_error_patterns": list(CREDIT_LIMIT_PATTERNS),
                "extra_error_patterns": arguments.credit_error_pattern,
            },
            "runs": [
                slot_entry(row["id"], repeat_index)
                for repeat_index, _line_number, row, _prompt in schedule
            ],
        }
        refresh_summary(manifest)
        write_json(manifest_path, manifest)

    for index, ((repeat_index, line_number, row, prompt), run) in enumerate(
        zip(schedule, manifest["runs"], strict=True),
        start=1,
    ):
        identifier = row["id"]
        if run["status"] == "succeeded":
            print(
                f"[{index}/{len(schedule)}] case {identifier}, "
                f"run {repeat_index:02d}: already succeeded",
                flush=True,
            )
            continue
        slot_dir = prepare_slot(
            run_dir,
            repeat_index,
            line_number,
            row,
            prompt,
        )
        print(
            f"[{index}/{len(schedule)}] case {identifier}, "
            f"run {repeat_index:02d}: "
            f"{'prepare' if arguments.dry_run else 'execute'}",
            flush=True,
        )

        if arguments.dry_run:
            run["status"] = "prepared"
            write_json(
                slot_dir / "run.json",
                {
                    "schema_version": "codex-ip-trajectory-slot.v1",
                    "case_id": identifier,
                    "repeat_index": repeat_index,
                    "status": "prepared",
                    "prompt_sha256": digest(
                        QUESTIONS_DIR / f"q{identifier:04d}" / "prompt.txt"
                    ),
                    "source_record_sha256": digest(
                        QUESTIONS_DIR
                        / f"q{identifier:04d}"
                        / "source_record.json"
                    ),
                },
            )
            refresh_summary(manifest)
            write_json(manifest_path, manifest)
            continue

        assert executable is not None
        next_retry_at = run.get("next_retry_at")
        if isinstance(next_retry_at, str):
            retry_time = datetime.fromisoformat(next_retry_at)
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=timezone.utc)
            remaining_wait = max(
                0,
                int((retry_time - datetime.now(timezone.utc)).total_seconds()),
            )
            if remaining_wait:
                print(
                    f"  resumed during a credit wait; sleeping "
                    f"{remaining_wait} more seconds",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    time.sleep(remaining_wait)
                except KeyboardInterrupt:
                    run["status"] = "interrupted"
                    manifest["status"] = "interrupted"
                    manifest["ended_at"] = now()
                    refresh_summary(manifest)
                    write_json(manifest_path, manifest)
                    print(
                        f"Interrupted during resumed credit wait; partial "
                        f"traces are preserved in {run_dir}",
                        file=sys.stderr,
                    )
                    return 130
            run.pop("next_retry_at", None)
            manifest.pop("next_retry_at", None)

        existing_attempts = [
            int(path.name.removeprefix("attempt_"))
            for path in slot_dir.glob("attempt_[0-9][0-9][0-9]")
            if path.is_dir()
        ]
        recorded_attempts = [
            int(attempt.get("attempt_index", 0))
            for attempt in run.get("attempts", [])
        ]
        attempt_index = max([0, *existing_attempts, *recorded_attempts]) + 1
        while True:
            attempt_dir, metadata = prepare_attempt(
                slot_dir,
                repeat_index,
                attempt_index,
                line_number,
                row,
            )
            run["status"] = "running"
            manifest["status"] = "running"
            manifest.pop("next_retry_at", None)
            refresh_summary(manifest)
            write_json(manifest_path, manifest)
            print(
                f"  attempt {attempt_index:03d}",
                flush=True,
            )

            try:
                metadata = run_case(
                    attempt_dir,
                    metadata,
                    prompt,
                    executable,
                    arguments.model,
                    arguments.sandbox,
                    arguments.credit_error_pattern,
                )
            except KeyboardInterrupt:
                metadata_path = attempt_dir / "metadata.json"
                if metadata_path.is_file():
                    interrupted_metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    run["attempts"].append(
                        manifest_attempt(interrupted_metadata)
                    )
                run["status"] = "interrupted"
                manifest["status"] = "interrupted"
                manifest["ended_at"] = now()
                refresh_summary(manifest)
                write_json(manifest_path, manifest)
                print(
                    f"Interrupted; partial traces are preserved in {run_dir}",
                    file=sys.stderr,
                )
                return 130

            run["attempts"].append(manifest_attempt(metadata))
            if metadata["status"] == "succeeded":
                run["status"] = "succeeded"
                run["successful_attempt"] = attempt_index
                write_json(
                    slot_dir / "run.json",
                    {
                        "schema_version": "codex-ip-trajectory-slot.v1",
                        "case_id": identifier,
                        "repeat_index": repeat_index,
                        "status": "succeeded",
                        "successful_attempt": attempt_index,
                        "attempts": run["attempts"],
                    },
                )
                refresh_summary(manifest)
                write_json(manifest_path, manifest)
                break

            if metadata.get("failure_kind") == "prohibited_filesystem_access":
                policy_failure_count = sum(
                    1
                    for attempt in run["attempts"]
                    if attempt.get("failure_kind")
                    == "prohibited_filesystem_access"
                )
                if policy_failure_count <= 1:
                    run["status"] = "retrying_policy_violation"
                    refresh_summary(manifest)
                    write_json(manifest_path, manifest)
                    print(
                        f"  policy violation rejected; retrying slot once "
                        f"with attempt {attempt_index + 1:03d}",
                        file=sys.stderr,
                        flush=True,
                    )
                    attempt_index += 1
                    continue

            if metadata.get("failure_kind") != "credits_exhausted":
                run["status"] = "failed"
                manifest["status"] = "failed"
                manifest["ended_at"] = now()
                refresh_summary(manifest)
                write_json(manifest_path, manifest)
                print(
                    f"ERROR: case {identifier}, run {repeat_index:02d}, "
                    f"attempt {attempt_index:03d} failed for a non-credit "
                    f"reason; inspect {attempt_dir / 'metadata.json'}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1

            retry_at = datetime.now(timezone.utc) + timedelta(
                seconds=arguments.credit_retry_seconds
            )
            run["status"] = "waiting_for_credits"
            run["next_retry_at"] = retry_at.isoformat(timespec="seconds")
            manifest["status"] = "waiting_for_credits"
            manifest["next_retry_at"] = run["next_retry_at"]
            refresh_summary(manifest)
            write_json(manifest_path, manifest)
            print(
                f"  credit/usage limit detected; retrying the same slot in "
                f"{arguments.credit_retry_seconds} seconds at "
                f"{run['next_retry_at']}",
                file=sys.stderr,
                flush=True,
            )
            try:
                time.sleep(arguments.credit_retry_seconds)
            except KeyboardInterrupt:
                run["status"] = "interrupted"
                manifest["status"] = "interrupted"
                manifest["ended_at"] = now()
                refresh_summary(manifest)
                write_json(manifest_path, manifest)
                print(
                    f"Interrupted during credit wait; partial traces are "
                    f"preserved in {run_dir}",
                    file=sys.stderr,
                )
                return 130
            run.pop("next_retry_at", None)
            attempt_index += 1

    succeeded = manifest["summary"]["succeeded"]
    prepared = manifest["summary"]["prepared"]
    required = manifest["summary"]["required"]
    if arguments.dry_run:
        manifest["status"] = "prepared" if prepared == required else "failed"
    else:
        manifest["status"] = "succeeded" if succeeded == required else "failed"
    manifest["ended_at"] = now()
    refresh_summary(manifest)
    write_json(manifest_path, manifest)
    print(f"Output: {run_dir}", flush=True)
    return 0 if manifest["status"] in {"prepared", "succeeded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
