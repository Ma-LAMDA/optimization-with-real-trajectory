#!/usr/bin/env python3
"""Compact repeated experiment artifacts without changing trajectory evidence.

The migration is deliberately conservative:

* every stdout.log must be byte-identical to its sibling events.jsonl;
* prompt.txt and source_record.json must have one digest per question/experiment;
* all per-attempt hooks.json files in the 100x10 experiment must be identical;
* metadata paths and recorded hashes are checked before any source is removed.

Run without ``--apply`` for a read-only audit.  Run with ``--apply`` to create
canonical files, update metadata, and remove only the verified duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "experiments"
REPORT_PATH = EXPERIMENTS_ROOT / "ARCHIVE_COMPACTION_REPORT.json"
SLOT_RE = re.compile(r"^(q\d{4})_r\d+$")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    runs_relative: Path
    centralize_hooks: bool = False

    @property
    def root(self) -> Path:
        return EXPERIMENTS_ROOT / self.name

    @property
    def runs_root(self) -> Path:
        return self.root / self.runs_relative


SPECS = (
    ExperimentSpec(
        "2026-07-27-ip_codex_train0629_14x10",
        Path("results/runs/fullaccess"),
    ),
    ExperimentSpec(
        "2026-07-28-ip_codex_train0629_10x10",
        Path("results/runs/fullaccess"),
    ),
    ExperimentSpec(
        "2026-07-28-ip_codex_train0629_100x10",
        Path("results/runs"),
        centralize_hooks=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the verified compaction. Default behavior is read-only.",
    )
    parser.add_argument(
        "--refresh-legacy-baseline",
        action="store_true",
        help="Refresh the 10x10 protected-file baseline after compaction.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lf_normalized_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256_bytes(payload)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def relative_posix(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start=start)).as_posix()


def ensure_within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())
    return resolved


def slot_directories(spec: ExperimentSpec) -> list[Path]:
    slots = [
        path
        for path in spec.runs_root.iterdir()
        if path.is_dir() and SLOT_RE.fullmatch(path.name)
    ]
    if not slots:
        raise ValueError(f"no run slots found under {spec.runs_root}")
    return sorted(slots)


def attempt_directories(slot: Path) -> list[Path]:
    attempts = sorted(path for path in slot.glob("attempt_*") if path.is_dir())
    if not attempts:
        raise ValueError(f"no attempts found under {slot}")
    return attempts


def assert_single_payload(
    *,
    candidates: list[Path],
    description: str,
) -> tuple[bytes, str]:
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError(f"no artifact found for {description}")
    payloads: dict[str, list[Path]] = {}
    for path in existing:
        payloads.setdefault(sha256_file(path), []).append(path)
    if len(payloads) != 1:
        details = ", ".join(
            f"{digest}: {len(paths)} files" for digest, paths in payloads.items()
        )
        raise ValueError(f"conflicting artifacts for {description}: {details}")
    digest = next(iter(payloads))
    return existing[0].read_bytes(), digest


def metadata_sha(metadata: dict[str, Any], key: str) -> str | None:
    nested = metadata.get("sha256")
    if isinstance(nested, dict) and isinstance(nested.get(key), str):
        return nested[key]
    direct = metadata.get(f"{key}_sha256")
    return direct if isinstance(direct, str) else None


def update_attempt_metadata(
    *,
    metadata_path: Path,
    prompt_path: Path,
    source_record_path: Path,
    prompt_digest: str,
    source_digest: str,
    prompt_lf_digest: str,
    source_lf_digest: str,
    hook_path: Path | None,
    apply: bool,
) -> bool:
    attempt = metadata_path.parent
    metadata = read_json(metadata_path)
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"metadata has no files object: {metadata_path}")

    recorded_prompt = metadata_sha(metadata, "prompt")
    recorded_source = metadata_sha(metadata, "source_record")
    prompt_digests = {prompt_digest, prompt_lf_digest}
    source_digests = {source_digest, source_lf_digest}
    if prompt_path.is_file():
        prompt_digests.add(lf_normalized_sha256(prompt_path))
    if source_record_path.is_file():
        source_digests.add(lf_normalized_sha256(source_record_path))
    if recorded_prompt is not None and recorded_prompt not in prompt_digests:
        raise ValueError(f"recorded prompt hash mismatch: {metadata_path}")
    if recorded_source is not None and recorded_source not in source_digests:
        raise ValueError(f"recorded source_record hash mismatch: {metadata_path}")

    expected_prompt = relative_posix(prompt_path, attempt)
    expected_source = relative_posix(source_record_path, attempt)
    expected_hook = relative_posix(hook_path, attempt) if hook_path else None
    changed = (
        files.get("prompt") != expected_prompt
        or files.get("source_record") != expected_source
        or (hook_path is not None and files.get("hook_config") != expected_hook)
    )

    events = attempt / "events.jsonl"
    stdout = attempt / "stdout.log"
    if stdout.is_file():
        if not events.is_file() or sha256_file(stdout) != sha256_file(events):
            raise ValueError(f"stdout is not an exact events alias: {attempt}")
    if files.get("stdout") not in (None, "events.jsonl", "stdout.log"):
        raise ValueError(f"unexpected stdout metadata path: {metadata_path}")
    if "stdout" in files and files.get("stdout") != "events.jsonl":
        changed = True

    if not apply:
        return changed

    files["prompt"] = expected_prompt
    files["source_record"] = expected_source
    if hook_path is not None:
        files["hook_config"] = expected_hook
    if "stdout" in files:
        files["stdout"] = "events.jsonl"
    metadata["files"] = files
    if changed:
        write_json(metadata_path, metadata)
    return changed


def remove_verified_file(path: Path, experiment_root: Path) -> int:
    ensure_within(path, experiment_root)
    if not path.is_file():
        return 0
    size = path.stat().st_size
    path.unlink()
    return size


def remove_verified_workspace(
    workspace: Path, experiment_root: Path
) -> tuple[bool, int, int]:
    ensure_within(workspace, experiment_root)
    if not workspace.exists():
        return False, 0, 0
    entries = sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file()
    )
    if entries != [".codex/hooks.json"]:
        raise ValueError(f"workspace has unexpected retained files: {workspace}: {entries}")
    hook_path = workspace / ".codex" / "hooks.json"
    removed_bytes = hook_path.stat().st_size
    shutil.rmtree(workspace)
    return True, removed_bytes, 1


def compact_experiment(spec: ExperimentSpec, apply: bool) -> dict[str, Any]:
    ensure_within(spec.root, ROOT)
    slots = slot_directories(spec)
    attempts: list[Path] = []
    canonical_files: set[Path] = set()
    duplicate_static_files: list[Path] = []
    metadata_changes = 0
    question_keys: set[str] = set()

    question_payloads: dict[str, dict[str, tuple[bytes, str]]] = {}
    for slot in slots:
        match = SLOT_RE.fullmatch(slot.name)
        assert match is not None
        question_key = match.group(1)
        question_keys.add(question_key)
        slot_attempts = attempt_directories(slot)
        attempts.extend(slot_attempts)
        canonical_dir = spec.root / "results" / "questions" / question_key
        prompt_target = canonical_dir / "prompt.txt"
        source_target = canonical_dir / "source_record.json"
        prompt_candidates = [prompt_target, slot / "prompt.txt"]
        source_candidates = [source_target, slot / "source_record.json"]
        if spec.centralize_hooks:
            prompt_candidates.extend(attempt / "prompt.txt" for attempt in slot_attempts)
            source_candidates.extend(
                attempt / "source_record.json" for attempt in slot_attempts
            )
        prompt_payload = assert_single_payload(
            candidates=prompt_candidates,
            description=f"{spec.name}/{question_key}/prompt",
        )
        source_payload = assert_single_payload(
            candidates=source_candidates,
            description=f"{spec.name}/{question_key}/source_record",
        )
        previous = question_payloads.get(question_key)
        current = {"prompt": prompt_payload, "source_record": source_payload}
        if previous is not None and previous != current:
            raise ValueError(f"run slots disagree for {spec.name}/{question_key}")
        question_payloads[question_key] = current
        canonical_files.update((prompt_target, source_target))
        duplicate_static_files.extend((slot / "prompt.txt", slot / "source_record.json"))
        if spec.centralize_hooks:
            for attempt in slot_attempts:
                duplicate_static_files.extend(
                    (attempt / "prompt.txt", attempt / "source_record.json")
                )

    hook_target: Path | None = None
    hook_payload: tuple[bytes, str] | None = None
    hook_sources: list[Path] = []
    workspaces: list[Path] = []
    if spec.centralize_hooks:
        hook_target = spec.root / "config" / "hooks.json"
        hook_sources = [attempt / "workspace" / ".codex" / "hooks.json" for attempt in attempts]
        hook_payload = assert_single_payload(
            candidates=[hook_target, *hook_sources],
            description=f"{spec.name}/hooks",
        )
        canonical_files.add(hook_target)
        workspaces = [attempt / "workspace" for attempt in attempts]

    stdout_files = [attempt / "stdout.log" for attempt in attempts]
    stdout_existing = [path for path in stdout_files if path.is_file()]
    for stdout in stdout_existing:
        events = stdout.with_name("events.jsonl")
        if not events.is_file() or sha256_file(stdout) != sha256_file(events):
            raise ValueError(f"stdout does not match events: {stdout}")

    duplicate_existing = [
        path
        for path in [*duplicate_static_files, *stdout_files, *hook_sources]
        if path.is_file()
    ]
    duplicate_bytes_before = sum(path.stat().st_size for path in duplicate_existing)
    canonical_bytes_total = sum(
        len(payload)
        for payloads in question_payloads.values()
        for payload, _digest in payloads.values()
    )
    if hook_payload is not None:
        canonical_bytes_total += len(hook_payload[0])
    canonical_creation_bytes = sum(
        len(question_payloads[path.parent.name][path.stem][0])
        if path.parent.name in question_payloads
        else len(hook_payload[0]) if hook_payload is not None else 0
        for path in canonical_files
        if not path.is_file()
    )
    planned_net_bytes_saved = duplicate_bytes_before - canonical_creation_bytes

    created_bytes = 0
    created_files = 0
    if apply:
        for question_key, payloads in sorted(question_payloads.items()):
            target_dir = spec.root / "results" / "questions" / question_key
            for filename, payload_key in (
                ("prompt.txt", "prompt"),
                ("source_record.json", "source_record"),
            ):
                target = target_dir / filename
                payload, digest = payloads[payload_key]
                if not target.is_file():
                    write_bytes(target, payload)
                    created_bytes += len(payload)
                    created_files += 1
                elif sha256_file(target) != digest:
                    raise ValueError(f"canonical artifact changed unexpectedly: {target}")
        if hook_target is not None and hook_payload is not None:
            payload, digest = hook_payload
            if not hook_target.is_file():
                write_bytes(hook_target, payload)
                created_bytes += len(payload)
                created_files += 1
            elif sha256_file(hook_target) != digest:
                raise ValueError(f"canonical hook changed unexpectedly: {hook_target}")

    for attempt in attempts:
        match = SLOT_RE.fullmatch(attempt.parent.name)
        assert match is not None
        question_key = match.group(1)
        canonical_dir = spec.root / "results" / "questions" / question_key
        payloads = question_payloads[question_key]
        metadata_changes += int(
            update_attempt_metadata(
                metadata_path=attempt / "metadata.json",
                prompt_path=canonical_dir / "prompt.txt",
                source_record_path=canonical_dir / "source_record.json",
                prompt_digest=payloads["prompt"][1],
                source_digest=payloads["source_record"][1],
                prompt_lf_digest=sha256_bytes(
                    payloads["prompt"][0].replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                ),
                source_lf_digest=sha256_bytes(
                    payloads["source_record"][0]
                    .replace(b"\r\n", b"\n")
                    .replace(b"\r", b"\n")
                ),
                hook_path=hook_target,
                apply=apply,
            )
        )

    removed_bytes = 0
    removed_files = 0
    removed_workspaces = 0
    if apply:
        for path in [*duplicate_static_files, *stdout_files]:
            if path.is_file():
                removed_bytes += remove_verified_file(path, spec.root)
                removed_files += 1
        for workspace in workspaces:
            if workspace.exists():
                removed, workspace_bytes, workspace_files = remove_verified_workspace(
                    workspace, spec.root
                )
                removed_workspaces += int(removed)
                removed_bytes += workspace_bytes
                removed_files += workspace_files

    remaining_stdout = sum(path.is_file() for path in stdout_files)
    remaining_duplicate_static = sum(path.is_file() for path in duplicate_static_files)
    remaining_hook_sources = sum(path.is_file() for path in hook_sources)
    return {
        "experiment": spec.name,
        "questions": len(question_keys),
        "run_slots": len(slots),
        "attempts": len(attempts),
        "canonical_files": len(canonical_files),
        "canonical_bytes": canonical_bytes_total,
        "duplicate_files_before": len(duplicate_existing),
        "duplicate_bytes_before": duplicate_bytes_before,
        "planned_net_bytes_saved": planned_net_bytes_saved,
        "created_files": created_files,
        "created_bytes": created_bytes,
        "metadata_files_updated_or_pending": metadata_changes,
        "duplicate_files_removed": removed_files,
        "duplicate_bytes_removed": removed_bytes,
        "workspaces_removed": removed_workspaces,
        "stdout_files_before_or_remaining": len(stdout_existing),
        "remaining_stdout_files": remaining_stdout,
        "remaining_repeated_prompt_source_files": remaining_duplicate_static,
        "remaining_per_attempt_hook_files": remaining_hook_sources,
        "net_bytes_saved_this_run": removed_bytes - created_bytes,
    }


def refresh_integrity_baselines() -> dict[str, int]:
    baseline_path = (
        EXPERIMENTS_ROOT
        / "2026-07-28-ip_codex_train0629_10x10"
        / "runtime"
        / "baseline.json"
    )
    baseline = read_json(baseline_path)
    protected_roots = baseline.get("protected_roots")
    if not isinstance(protected_roots, list) or not all(
        isinstance(value, str) for value in protected_roots
    ):
        raise ValueError(f"invalid protected_roots: {baseline_path}")
    files: list[dict[str, Any]] = []
    for relative_root in protected_roots:
        protected_root = ensure_within(ROOT / relative_root, ROOT)
        for path in sorted(item for item in protected_root.rglob("*") if item.is_file()):
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "length": stat.st_size,
                    "last_write_time_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(timespec="microseconds"),
                    "sha256": sha256_file(path),
                }
            )
    baseline["captured_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )
    baseline["files"] = files
    baseline["archive_compaction_report"] = REPORT_PATH.relative_to(ROOT).as_posix()
    write_json(baseline_path, baseline)

    distill_baseline_path = (
        EXPERIMENTS_ROOT
        / "2026-07-28-ip_codex_train0629_100x10"
        / "results"
        / "report"
        / "baseline.json"
    )
    distill_baseline = read_json(distill_baseline_path)
    protected_trees = distill_baseline.get("protected_trees")
    if not isinstance(protected_trees, dict):
        raise ValueError(f"invalid protected_trees: {distill_baseline_path}")
    refreshed_trees: dict[str, dict[str, str]] = {}
    for relative_root in protected_trees:
        protected_root = ensure_within(ROOT / relative_root, ROOT)
        refreshed_trees[relative_root] = {
            path.relative_to(protected_root).as_posix(): sha256_file(path)
            for path in sorted(protected_root.rglob("*"))
            if path.is_file()
        }
    distill_baseline["captured_at"] = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )
    distill_baseline["protected_trees"] = refreshed_trees
    distill_baseline["archive_compaction_report"] = REPORT_PATH.relative_to(
        ROOT
    ).as_posix()
    write_json(distill_baseline_path, distill_baseline)
    return {
        "10x10_protected_files": len(files),
        "100x10_protected_files": sum(
            len(tree) for tree in refreshed_trees.values()
        ),
    }


def main() -> None:
    options = parse_args()
    results = [compact_experiment(spec, options.apply) for spec in SPECS]
    total = {
        key: sum(int(item[key]) for item in results)
        for key in (
            "questions",
            "run_slots",
            "attempts",
            "canonical_files",
            "canonical_bytes",
            "duplicate_files_before",
            "duplicate_bytes_before",
            "planned_net_bytes_saved",
            "created_files",
            "created_bytes",
            "metadata_files_updated_or_pending",
            "duplicate_files_removed",
            "duplicate_bytes_removed",
            "workspaces_removed",
            "stdout_files_before_or_remaining",
            "remaining_stdout_files",
            "remaining_repeated_prompt_source_files",
            "remaining_per_attempt_hook_files",
            "net_bytes_saved_this_run",
        )
    }
    report = {
        "schema_version": "experiment-archive-compaction.v1",
        "mode": "applied" if options.apply else "audit",
        "policy": {
            "stdout": "events.jsonl is the single canonical stream",
            "prompt_and_source_record": "one canonical copy per experiment and question",
            "hooks": "one canonical config for the 100x10 experiment",
            "symbolic_links": False,
            "evidence_content_changed": False,
        },
        "experiments": results,
        "total": total,
    }
    if options.apply:
        write_json(REPORT_PATH, report)
    if options.refresh_legacy_baseline:
        report["refreshed_integrity_baselines"] = refresh_integrity_baselines()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not options.apply:
        print("\nRead-only audit complete. Re-run with --apply to compact.")


if __name__ == "__main__":
    main()
