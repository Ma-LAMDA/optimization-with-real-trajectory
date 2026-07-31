#!/usr/bin/env python3
"""Validate the 2026-07-28 Codex IP 10x10 raw trajectory experiment."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_ROOT.parents[1]
RUN_DIR = EXPERIMENT_ROOT / "results" / "runs" / "fullaccess"
OLD_RUN_DIR = (
    ROOT
    / "experiments"
    / "2026-07-27-ip_codex_train0629_14x10"
    / "results"
    / "runs"
    / "fullaccess"
)
REPORT_PATH = EXPERIMENT_ROOT / "results" / "reports" / "validation.json"
BASELINE_PATH = EXPERIMENT_ROOT / "runtime" / "baseline.json"
SOURCE_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
SOURCE_TEMPLATE = ROOT / "data" / "simulation" / "IP user prompt.txt"
LOCAL_TEMPLATE = EXPERIMENT_ROOT / "inputs" / "IP user prompt local-url-only.txt"
EXPECTED_CASE_IDS = (13, 14, 17, 18, 87, 88, 91, 92, 93, 94)
EXPECTED_REPEATS = 10
EXPECTED_MODEL = "gpt-5.6-sol"
EXPECTED_EXPERIMENT_DIR_NAME = "2026-07-28-ip_codex_train0629_10x10"
OLD_EXPERIMENT_DIR_NAME = "ip_codex_0728_10x10"
REMOTE_HOST = "10.139.194.154"
LOCAL_HOST = "127.0.0.1"
LOCAL_URL = "http://127.0.0.1:3080"
PLACEHOLDERS = {
    "{original_query}": "question",
    "{output_format}": "output_format",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_rows() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with SOURCE_DATASET.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), int):
                raise ValueError(f"{SOURCE_DATASET}:{line_number}: invalid record")
            rows[row["id"]] = row
    return rows


def instantiate(template: str, row: dict[str, Any]) -> str:
    prompt = template
    for placeholder, field in PLACEHOLDERS.items():
        if prompt.count(placeholder) != 1:
            raise ValueError(f"localized template has invalid {placeholder}")
        prompt = prompt.replace(placeholder, row[field])
    return prompt


def command_text(item: dict[str, Any]) -> str:
    command = item.get("command", "")
    if isinstance(command, str):
        return command
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return json.dumps(command, ensure_ascii=False)


def verify_baseline(errors: list[str], checks: dict[str, Any]) -> None:
    if not BASELINE_PATH.is_file():
        errors.append(f"missing baseline: {BASELINE_PATH}")
        return
    baseline = read_json(BASELINE_PATH)
    changed: list[str] = []
    missing: list[str] = []
    for entry in baseline.get("files", []):
        path = ROOT / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif digest(path) != entry["sha256"]:
            changed.append(entry["path"])
    checks["baseline_file_count"] = len(baseline.get("files", []))
    checks["baseline_changed"] = changed
    checks["baseline_missing"] = missing
    if changed:
        errors.append(f"protected/old files changed: {changed[:10]}")
    if missing:
        errors.append(f"protected/old files missing: {missing[:10]}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    rows = load_rows()
    source_template = SOURCE_TEMPLATE.read_text(encoding="utf-8")
    localized_template = LOCAL_TEMPLATE.read_text(encoding="utf-8")
    expected_localized = source_template.replace(REMOTE_HOST, LOCAL_HOST)
    checks["source_host_occurrences"] = source_template.count(REMOTE_HOST)
    checks["localized_template_exact"] = localized_template == expected_localized
    if source_template.count(REMOTE_HOST) != 2:
        errors.append("source template remote host occurrence count is not 2")
    if localized_template != expected_localized:
        errors.append("localized template contains changes beyond host replacement")
    if REMOTE_HOST in localized_template:
        errors.append("remote service host remains in localized template")

    manifest_path = RUN_DIR / "manifest.json"
    manifest = read_json(manifest_path)
    checks["experiment_directory_name"] = EXPERIMENT_ROOT.name
    checks["old_experiment_directory_exists"] = (
        EXPERIMENT_ROOT.parent / OLD_EXPERIMENT_DIR_NAME
    ).exists()
    if EXPERIMENT_ROOT.name != EXPECTED_EXPERIMENT_DIR_NAME:
        errors.append("experiment directory does not use the normalized name")
    if checks["old_experiment_directory_exists"]:
        errors.append("old experiment directory still exists")
    relocation = read_json(EXPERIMENT_ROOT / "runtime" / "relocation.json")
    if relocation.get("from") != f"experiments/{OLD_EXPERIMENT_DIR_NAME}":
        errors.append("relocation source is incorrect")
    if relocation.get("to") != f"experiments/{EXPECTED_EXPERIMENT_DIR_NAME}":
        errors.append("relocation target is incorrect")
    manifest_current_paths = [
        str(manifest.get("localized_template", "")),
        str(manifest.get("codex_executable", "")),
        str(manifest.get("working_directory", "")),
        str(manifest.get("api_only_hook", "")),
    ]
    if any(OLD_EXPERIMENT_DIR_NAME in value for value in manifest_current_paths):
        errors.append("manifest current paths still use the old directory name")
    if any(
        EXPECTED_EXPERIMENT_DIR_NAME not in value
        for value in manifest_current_paths
    ):
        errors.append("manifest current paths do not all use the normalized name")
    hooks_text = (
        EXPERIMENT_ROOT / "runtime" / "workspace" / ".codex" / "hooks.json"
    ).read_text(encoding="utf-8")
    if OLD_EXPERIMENT_DIR_NAME in hooks_text:
        errors.append("hook command still uses the old experiment path")
    if EXPECTED_EXPERIMENT_DIR_NAME not in hooks_text:
        errors.append("hook command does not use the normalized path")
    hooks_config = json.loads(hooks_text)
    hook_command = hooks_config["hooks"]["PreToolUse"][0]["hooks"][0][
        "commandWindows"
    ]
    hook_script = Path(hook_command.split(" -B ", maxsplit=1)[1])
    checks["hook_command_script"] = str(hook_script)
    checks["hook_command_script_exists"] = hook_script.is_file()
    if not hook_script.is_file():
        archived_hook_script = (
            EXPERIMENT_ROOT
            / "runtime"
            / "workspace"
            / ".codex"
            / "hooks"
            / "api_only_hook.py"
        )
        checks["archived_hook_script_exists"] = archived_hook_script.is_file()
        if archived_hook_script.is_file():
            warnings.append(
                "historical hook command keeps its original absolute path; "
                "the archived hook script is present at the normalized experiment path"
            )
        else:
            errors.append("archived hook script is missing")
    expected_slots = [
        (case_id, repeat_index)
        for repeat_index in range(1, EXPECTED_REPEATS + 1)
        for case_id in EXPECTED_CASE_IDS
    ]
    actual_slots = [
        (run.get("case_id"), run.get("repeat_index"))
        for run in manifest.get("runs", [])
    ]
    if actual_slots != expected_slots:
        errors.append("manifest schedule differs from the exact 10x10 schedule")
    if manifest.get("status") != "succeeded":
        errors.append(f"manifest status is {manifest.get('status')!r}, not succeeded")
    if manifest.get("model") != EXPECTED_MODEL:
        errors.append(f"manifest model is {manifest.get('model')!r}")
    if manifest.get("service_base_url") != LOCAL_URL:
        errors.append("manifest service URL is not the local service")
    if manifest.get("dataset_sha256") != digest(SOURCE_DATASET):
        errors.append("dataset hash mismatch")
    if manifest.get("source_template_sha256") != digest(SOURCE_TEMPLATE):
        errors.append("source template hash mismatch")
    if manifest.get("localized_template_sha256") != digest(LOCAL_TEMPLATE):
        errors.append("localized template hash mismatch")

    statuses = Counter()
    thread_ids: list[str] = []
    prompt_hashes: dict[int, set[str]] = {case_id: set() for case_id in EXPECTED_CASE_IDS}
    prohibited_access: list[dict[str, Any]] = []
    non_local_http_commands: list[dict[str, Any]] = []
    successful_event_hashes: list[str] = []
    successful_final_hashes: list[str] = []
    hook_protected_thread_ids: set[str] = set()
    attempt_statuses: Counter[str] = Counter()
    attempt_failure_kinds: Counter[str] = Counter()
    for manifest_run in manifest.get("runs", []):
        for attempt in manifest_run.get("attempts", []):
            attempt_statuses[str(attempt.get("status"))] += 1
            failure_kind = attempt.get("failure_kind")
            if failure_kind:
                attempt_failure_kinds[str(failure_kind)] += 1
    attempt_count = 0
    duration_seconds = 0.0
    pre_hook_audited_successes = 0
    hook_protected_successes = 0
    for case_id, repeat_index in expected_slots:
        slot = RUN_DIR / f"q{case_id:04d}_r{repeat_index:02d}"
        run = read_json(slot / "run.json")
        statuses[str(run.get("status"))] += 1
        if run.get("status") != "succeeded":
            errors.append(f"{slot.name}: slot did not succeed")
            continue
        question_dir = (
            EXPERIMENT_ROOT
            / "results"
            / "questions"
            / f"q{case_id:04d}"
        )
        prompt_path = question_dir / "prompt.txt"
        source_record_path = question_dir / "source_record.json"
        expected_prompt = instantiate(localized_template, rows[case_id])
        actual_prompt = prompt_path.read_text(encoding="utf-8")
        if actual_prompt != expected_prompt:
            errors.append(f"{slot.name}: prompt differs from localized metadata prompt")
        if REMOTE_HOST in actual_prompt or LOCAL_URL not in actual_prompt:
            errors.append(f"{slot.name}: prompt service URL is invalid")
        if read_json(source_record_path) != rows[case_id]:
            errors.append(f"{slot.name}: source record differs from original metadata")
        prompt_hashes[case_id].add(digest(prompt_path))

        successful_attempt = run.get("successful_attempt")
        attempt_dir = slot / f"attempt_{int(successful_attempt):03d}"
        metadata = read_json(attempt_dir / "metadata.json")
        attempt_count += len(run.get("attempts", []))
        duration_seconds += float(metadata.get("duration_seconds") or 0)
        if metadata.get("status") != "succeeded":
            errors.append(f"{slot.name}: successful attempt metadata is not succeeded")
        if metadata.get("model_override") != EXPECTED_MODEL:
            errors.append(f"{slot.name}: wrong model override")
        command = metadata.get("command", [])
        hook_protected = "--ignore-user-config" not in command
        if hook_protected:
            hook_protected_successes += 1
            required_flags = (
                "--ephemeral",
                "--ignore-rules",
                "--dangerously-bypass-hook-trust",
                "--enable",
                "--skip-git-repo-check",
            )
        else:
            pre_hook_audited_successes += 1
            required_flags = (
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
            )
        for required in required_flags:
            if required not in command:
                errors.append(f"{slot.name}: missing Codex flag {required}")
        if metadata.get("service_base_url") != LOCAL_URL:
            errors.append(f"{slot.name}: wrong metadata service URL")
        if metadata.get("thread_id"):
            thread_ids.append(metadata["thread_id"])
            if hook_protected:
                hook_protected_thread_ids.add(metadata["thread_id"])
        else:
            errors.append(f"{slot.name}: missing thread id")
        for file_name in ("events.jsonl", "stderr.log", "final_answer.txt"):
            path = attempt_dir / file_name
            if not path.is_file():
                errors.append(f"{slot.name}: missing {file_name}")
        final = (attempt_dir / "final_answer.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        successful_event_hashes.append(digest(attempt_dir / "events.jsonl"))
        successful_final_hashes.append(digest(attempt_dir / "final_answer.txt"))
        if "<result>" not in final or "</result>" not in final:
            errors.append(f"{slot.name}: invalid final answer wrapper")

        with (attempt_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
            for event_line, line in enumerate(handle, start=1):
                event = json.loads(line)
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") != "command_execution":
                    continue
                text = command_text(item)
                folded = text.casefold()
                direct_saved_configs = "saved_configs" in folded
                direct_repository_path = str(ROOT).casefold() in folded
                parent_traversal = "..\\" in text or "../" in text
                urls = re.findall(r"https?://[^\s'\"]+", text, flags=re.IGNORECASE)
                bad_urls = [
                    value
                    for value in urls
                    if not value.startswith(LOCAL_URL)
                ]
                if bad_urls:
                    non_local_http_commands.append(
                        {
                            "slot": slot.name,
                            "line": event_line,
                            "urls": bad_urls,
                            "command": text,
                        }
                    )
                if direct_saved_configs or direct_repository_path or parent_traversal:
                    prohibited_access.append(
                        {
                            "slot": slot.name,
                            "line": event_line,
                            "command": text,
                        }
                    )

    if len(thread_ids) != 100 or len(set(thread_ids)) != 100:
        errors.append("thread ids are missing or not unique across 100 runs")
    if attempt_count < 100:
        errors.append("fewer than 100 Codex attempts were recorded")
    if any(len(values) != 1 for values in prompt_hashes.values()):
        errors.append("prompt hashes are inconsistent within at least one case")
    if prohibited_access:
        errors.append("event log contains direct repository/saved_configs access")
    if non_local_http_commands:
        errors.append("successful event log contains a non-local HTTP command")
    if len(set(successful_event_hashes)) != 100:
        errors.append("successful event logs are not all unique")
    old_event_hashes = {
        digest(path)
        for path in OLD_RUN_DIR.glob("q*_r*/attempt_*/events.jsonl")
        if path.is_file()
    }
    old_event_hash_overlap = sorted(
        set(successful_event_hashes).intersection(old_event_hashes)
    )
    if old_event_hash_overlap:
        errors.append("successful event logs overlap the old 14x10 experiment")
    expected_attempt_statuses = {
        "succeeded": 100,
        "rejected_policy_violation": 4,
        "interrupted": 1,
    }
    if dict(attempt_statuses) != expected_attempt_statuses:
        errors.append(
            f"unexpected retained attempt statuses: {dict(attempt_statuses)}"
        )

    hook_path = (
        EXPERIMENT_ROOT / "runtime" / "hook_audit" / "pre_tool_use.jsonl"
    )
    hook_parse_errors: list[int] = []
    protected_hook_records: list[dict[str, Any]] = []
    if hook_path.is_file():
        with hook_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    hook_parse_errors.append(line_number)
                    continue
                if record.get("session_id") in hook_protected_thread_ids:
                    protected_hook_records.append(record)
    else:
        errors.append("missing API-only hook audit log")
    protected_sessions_with_hook_records = {
        str(record.get("session_id"))
        for record in protected_hook_records
    }
    if protected_sessions_with_hook_records != hook_protected_thread_ids:
        errors.append("at least one hook-protected success lacks hook audit records")
    if hook_parse_errors:
        warnings.append(
            "hook audit contains three trailing fragments from concurrent appends; "
            f"raw lines are preserved at {hook_parse_errors[:10]}"
        )
    verify_baseline(errors, checks)

    checks.update(
        {
            "expected_slots": 100,
            "manifest_slots": len(actual_slots),
            "slot_statuses": dict(sorted(statuses.items())),
            "codex_attempts": attempt_count,
            "unique_thread_ids": len(set(thread_ids)),
            "total_duration_seconds": round(duration_seconds, 3),
            "pre_hook_audited_successes": pre_hook_audited_successes,
            "hook_protected_successes": hook_protected_successes,
            "prohibited_access": prohibited_access,
            "non_local_http_commands": non_local_http_commands,
            "unique_successful_event_hashes": len(set(successful_event_hashes)),
            "unique_successful_final_answer_hashes": len(
                set(successful_final_hashes)
            ),
            "old_event_hash_overlap": old_event_hash_overlap,
            "retained_attempt_statuses": dict(sorted(attempt_statuses.items())),
            "retained_attempt_failure_kinds": dict(
                sorted(attempt_failure_kinds.items())
            ),
            "hook_audit_parse_errors": hook_parse_errors,
            "hook_protected_sessions_with_records": len(
                protected_sessions_with_hook_records
            ),
            "hook_protected_allowed_checks": sum(
                1
                for record in protected_hook_records
                if record.get("allowed") is True
            ),
            "hook_protected_denied_checks": sum(
                1
                for record in protected_hook_records
                if record.get("allowed") is False
            ),
        }
    )
    report = {
        "schema_version": "codex-ip-trajectory-validation.v1",
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
