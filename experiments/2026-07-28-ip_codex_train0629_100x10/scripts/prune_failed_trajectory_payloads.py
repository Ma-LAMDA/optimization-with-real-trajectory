#!/usr/bin/env python3
"""Prune bulky payloads from non-accepted attempts while retaining audit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT = Path(__file__).resolve().parents[1]
REPORT = EXPERIMENT / "results" / "report"
RUNS = EXPERIMENT / "results" / "runs"
MANIFEST = REPORT / "failed_trajectory_pruning.json"
PENDING_MANIFEST = REPORT / ".failed_trajectory_pruning.pending.json"
SCHEMA_VERSION = "ip-distill-failed-trajectory-pruning.v1"

PRUNED_FILES = (
    "events.jsonl",
    "hook_audit.jsonl",
    "stderr.log",
    "judge.stdout.log",
    "judge.stderr.log",
)
PRUNED_DIRECTORIES = ("hook_audit_parts",)
FORBIDDEN_HOOK_FRAGMENTS = (
    "train_0629.jsonl",
    "data/simulation",
    "saved_configs/",
    "saved_configs\\",
    "10.139.194.154:3080",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_experiment(path: Path) -> str:
    return path.relative_to(EXPERIMENT).as_posix()


def safe_target(attempt_dir: Path, relative_path: str) -> Path:
    attempt_root = attempt_dir.resolve()
    runs_root = RUNS.resolve()
    if runs_root not in attempt_root.parents:
        raise ValueError(f"attempt is outside the runs directory: {attempt_dir}")
    target = (attempt_dir / relative_path).resolve()
    if target != attempt_root and attempt_root not in target.parents:
        raise ValueError(f"artifact escapes its attempt directory: {target}")
    return target


def file_record(path: Path, attempt_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(attempt_dir).as_posix(),
        "kind": "file",
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def directory_record(path: Path, attempt_dir: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        child_size = child.stat().st_size
        child_sha256 = sha256(child)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(child_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(child_sha256.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        size_bytes += child_size
    return {
        "path": path.relative_to(attempt_dir).as_posix(),
        "kind": "directory",
        "file_count": file_count,
        "size_bytes": size_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def event_stats(path: Path) -> dict[str, Any]:
    line_count = 0
    invalid_lines: list[int] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            line_count += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
    return {"line_count": line_count, "invalid_line_numbers": invalid_lines}


def hook_stats(path: Path) -> dict[str, Any]:
    allowed_count = 0
    denied_count = 0
    invalid_lines: list[int] = []
    allowed_violation_lines: list[int] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            if item.get("allowed"):
                allowed_count += 1
                command = str(item.get("command") or "").lower()
                if "127.0.0.1:3080" not in command or any(
                    fragment.lower() in command
                    for fragment in FORBIDDEN_HOOK_FRAGMENTS
                ):
                    allowed_violation_lines.append(line_number)
            else:
                denied_count += 1
    return {
        "allowed_count": allowed_count,
        "denied_count": denied_count,
        "invalid_line_numbers": invalid_lines,
        "allowed_violation_line_numbers": allowed_violation_lines,
    }


def attempt_directories() -> list[Path]:
    return sorted(path for path in RUNS.glob("q*_r*/attempt_*") if path.is_dir())


def build_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    all_status_counts: Counter[str] = Counter()
    failed_status_counts: Counter[str] = Counter()
    pruned_bytes = 0
    retained_bytes = 0
    pruned_file_count = 0
    original_raw_event_lines = 0
    original_hook_allowed = 0
    original_hook_denied = 0

    attempts = attempt_directories()
    for attempt_dir in attempts:
        metadata_path = attempt_dir / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"missing metadata: {metadata_path}")
        metadata = load_json(metadata_path)
        status = str(metadata.get("status", "unknown"))
        all_status_counts[status] += 1
        if status == "accepted":
            continue

        failed_status_counts[status] += 1
        pruned_artifacts: list[dict[str, Any]] = []
        for name in PRUNED_FILES:
            path = attempt_dir / name
            if not path.exists():
                continue
            if not path.is_file():
                raise ValueError(f"expected a file: {path}")
            record = file_record(path, attempt_dir)
            if name == "events.jsonl":
                record["content_summary"] = event_stats(path)
                original_raw_event_lines += int(
                    record["content_summary"]["line_count"]
                )
            elif name == "hook_audit.jsonl":
                record["content_summary"] = hook_stats(path)
                original_hook_allowed += int(
                    record["content_summary"]["allowed_count"]
                )
                original_hook_denied += int(
                    record["content_summary"]["denied_count"]
                )
            pruned_artifacts.append(record)

        for name in PRUNED_DIRECTORIES:
            path = attempt_dir / name
            if not path.exists():
                continue
            if not path.is_dir():
                raise ValueError(f"expected a directory: {path}")
            pruned_artifacts.append(directory_record(path, attempt_dir))

        if not any(item["path"] == "events.jsonl" for item in pruned_artifacts):
            raise ValueError(f"failed attempt has no events.jsonl to archive: {attempt_dir}")

        pruned_roots = set(PRUNED_FILES) | set(PRUNED_DIRECTORIES)
        retained_files = [
            file_record(path, attempt_dir)
            for path in sorted(item for item in attempt_dir.rglob("*") if item.is_file())
            if path.relative_to(attempt_dir).parts[0] not in pruned_roots
        ]
        entry_pruned_bytes = sum(
            int(item["size_bytes"]) for item in pruned_artifacts
        )
        entry_retained_bytes = sum(
            int(item["size_bytes"]) for item in retained_files
        )
        entry_pruned_file_count = sum(
            int(item.get("file_count", 1)) for item in pruned_artifacts
        )
        pruned_bytes += entry_pruned_bytes
        retained_bytes += entry_retained_bytes
        pruned_file_count += entry_pruned_file_count
        entries.append(
            {
                "attempt_path": relative_to_experiment(attempt_dir),
                "row_index": metadata.get("row_index"),
                "original_id": metadata.get("original_id"),
                "sample_key": metadata.get("sample_key"),
                "target_success_slot": metadata.get("target_success_slot"),
                "attempt_index": metadata.get("attempt_index"),
                "status": status,
                "generation_status": metadata.get("generation_status"),
                "judge_status": metadata.get("judge_status"),
                "error_class": metadata.get("error_class"),
                "started_at": metadata.get("started_at"),
                "ended_at": metadata.get("ended_at"),
                "duration_seconds": metadata.get("duration_seconds"),
                "pruned_bytes": entry_pruned_bytes,
                "retained_bytes": entry_retained_bytes,
                "pruned_artifacts": pruned_artifacts,
                "retained_files": retained_files,
            }
        )

    final_audit_path = REPORT / "final_audit.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_root": relative_to_experiment(EXPERIMENT),
        "selection_rule": "metadata.status != 'accepted'",
        "retention_policy": {
            "pruned_files": list(PRUNED_FILES),
            "pruned_directories": list(PRUNED_DIRECTORIES),
            "retained_evidence": (
                "all other per-attempt files, including metadata.json, timing.json, "
                "judgment.json, final_answer.txt, exit_code.txt, and recovery.json when present"
            ),
        },
        "pre_pruning_final_audit": {
            "path": relative_to_experiment(final_audit_path),
            "sha256": sha256(final_audit_path),
        },
        "summary": {
            "attempt_count": len(attempts),
            "accepted_attempt_count": all_status_counts["accepted"],
            "pruned_failed_attempt_count": len(entries),
            "attempt_status_counts": dict(sorted(all_status_counts.items())),
            "pruned_status_counts": dict(sorted(failed_status_counts.items())),
            "pruned_file_count": pruned_file_count,
            "pruned_bytes": pruned_bytes,
            "retained_failed_evidence_bytes": retained_bytes,
            "original_pruned_raw_event_line_count": original_raw_event_lines,
            "original_pruned_hook_allowed_count": original_hook_allowed,
            "original_pruned_hook_denied_count": original_hook_denied,
        },
        "entries": entries,
    }


def validate_applied_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unexpected schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return {"passed": False, "errors": ["entries must be a list"], "entries": {}}

    entry_by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object entry")
            continue
        attempt_path = str(entry.get("attempt_path", ""))
        if not attempt_path or attempt_path in entry_by_path:
            errors.append(f"invalid or duplicate attempt_path: {attempt_path!r}")
            continue
        entry_by_path[attempt_path] = entry

    current_failed: set[str] = set()
    accepted_count = 0
    for attempt_dir in attempt_directories():
        attempt_path = relative_to_experiment(attempt_dir)
        metadata_path = attempt_dir / "metadata.json"
        if not metadata_path.is_file():
            errors.append(f"missing metadata: {attempt_path}")
            continue
        metadata = load_json(metadata_path)
        status = str(metadata.get("status", "unknown"))
        if status == "accepted":
            accepted_count += 1
            if attempt_path in entry_by_path:
                errors.append(f"accepted attempt appears in pruning manifest: {attempt_path}")
            continue
        current_failed.add(attempt_path)
        entry = entry_by_path.get(attempt_path)
        if entry is None:
            errors.append(f"failed attempt is absent from pruning manifest: {attempt_path}")
            continue
        if entry.get("status") != status:
            errors.append(f"status mismatch: {attempt_path}")
        if entry.get("attempt_index") != metadata.get("attempt_index"):
            errors.append(f"attempt_index mismatch: {attempt_path}")
        if entry.get("original_id") != metadata.get("original_id"):
            errors.append(f"original_id mismatch: {attempt_path}")

        for artifact in entry.get("pruned_artifacts", []):
            target = safe_target(attempt_dir, str(artifact.get("path", "")))
            if target.exists():
                errors.append(f"pruned artifact still exists: {relative_to_experiment(target)}")
        for record in entry.get("retained_files", []):
            target = safe_target(attempt_dir, str(record.get("path", "")))
            if not target.is_file():
                errors.append(f"retained file is missing: {relative_to_experiment(target)}")
                continue
            if target.stat().st_size != int(record.get("size_bytes", -1)):
                errors.append(f"retained size mismatch: {relative_to_experiment(target)}")
            elif sha256(target) != record.get("sha256"):
                errors.append(f"retained hash mismatch: {relative_to_experiment(target)}")

    if current_failed != set(entry_by_path):
        errors.append("manifest failed-attempt paths do not match the current archive")
    summary = manifest.get("summary", {})
    if int(summary.get("attempt_count", -1)) != len(attempt_directories()):
        errors.append("attempt_count does not match the current archive")
    if int(summary.get("accepted_attempt_count", -1)) != accepted_count:
        errors.append("accepted_attempt_count does not match the current archive")
    if int(summary.get("pruned_failed_attempt_count", -1)) != len(entry_by_path):
        errors.append("pruned_failed_attempt_count does not match manifest entries")

    return {"passed": not errors, "errors": errors, "entries": entry_by_path}


def apply_pruning() -> dict[str, Any]:
    if MANIFEST.exists() or PENDING_MANIFEST.exists():
        raise FileExistsError(
            "a pruning manifest already exists; use --check instead of pruning again"
        )
    manifest = build_manifest()
    atomic_json(PENDING_MANIFEST, manifest)
    for entry in manifest["entries"]:
        attempt_dir = EXPERIMENT / str(entry["attempt_path"])
        for artifact in entry["pruned_artifacts"]:
            target = safe_target(attempt_dir, str(artifact["path"]))
            if artifact["kind"] == "directory":
                shutil.rmtree(target)
            else:
                target.unlink()
    validation = validate_applied_manifest(manifest)
    if not validation["passed"]:
        raise RuntimeError("post-pruning validation failed: " + "; ".join(validation["errors"]))
    atomic_json(MANIFEST, manifest)
    PENDING_MANIFEST.unlink()
    return manifest


def summary_payload(manifest: dict[str, Any], *, mode: str) -> dict[str, Any]:
    summary = manifest["summary"]
    return {
        "mode": mode,
        "manifest": relative_to_experiment(MANIFEST),
        "attempt_count": summary["attempt_count"],
        "accepted_attempt_count": summary["accepted_attempt_count"],
        "pruned_failed_attempt_count": summary["pruned_failed_attempt_count"],
        "pruned_file_count": summary["pruned_file_count"],
        "pruned_bytes": summary["pruned_bytes"],
        "retained_failed_evidence_bytes": summary["retained_failed_evidence_bytes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write the manifest and delete payloads")
    mode.add_argument("--check", action="store_true", help="validate an applied pruning manifest")
    args = parser.parse_args()

    if args.check:
        manifest = load_json(MANIFEST)
        validation = validate_applied_manifest(manifest)
        print(
            json.dumps(
                {
                    **summary_payload(manifest, mode="check"),
                    "passed": validation["passed"],
                    "errors": validation["errors"],
                },
                ensure_ascii=False,
            )
        )
        return 0 if validation["passed"] else 1

    if args.apply:
        manifest = apply_pruning()
        print(json.dumps(summary_payload(manifest, mode="apply"), ensure_ascii=False))
        return 0

    manifest = build_manifest()
    print(json.dumps(summary_payload(manifest, mode="dry-run"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
