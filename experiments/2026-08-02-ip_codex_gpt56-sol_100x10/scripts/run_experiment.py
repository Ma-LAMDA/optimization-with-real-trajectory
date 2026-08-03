#!/usr/bin/env python3
"""Run the resumable 100-question Codex IP trajectory distillation experiment."""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
SCRIPTS_DIR = EXPERIMENT_ROOT / "scripts"
RUNTIME_DIR = EXPERIMENT_ROOT / "runtime"
REPORT_DIR = EXPERIMENT_ROOT / "results" / "report"
RUNS_DIR = EXPERIMENT_ROOT / "results" / "runs"
QUESTIONS_DIR = EXPERIMENT_ROOT / "results" / "questions"
DATASET = REPOSITORY_ROOT / "data" / "simulation" / "train_0629.jsonl"
SOURCE_TEMPLATE = EXPERIMENT_ROOT / "inputs" / "IP user prompt by text.txt"
SERVICE_SCRIPT = REPOSITORY_ROOT / "saved_configs_service" / "serve_saved_configs.py"
SAFE_INDEX = REPORT_DIR / "input_index.json"
LOCALIZED_TEMPLATE = REPORT_DIR / "localized_prompt_template.txt"
MANIFEST_PATH = REPORT_DIR / "manifest.json"
STATE_PATH = REPORT_DIR / "state.json"
ACCEPTED_PATH = REPORT_DIR / "accepted_index.json"
HEARTBEAT_PATH = REPORT_DIR / "heartbeat.json"
BASELINE_PATH = REPORT_DIR / "baseline.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
FINAL_REPORT_PATH = REPORT_DIR / "FINAL_REPORT.md"
STOPPED_PATH = REPORT_DIR / "STOPPED.json"
RUNNER_LOG = REPORT_DIR / "runner.log"
LOCK_PATH = REPORT_DIR / "runner.lock"
CLI_PATH = RUNTIME_DIR / "codex.exe"
MODEL_CACHE = Path.home() / ".codex" / "models_cache.json"
MODEL = "gpt-5.6-sol"
DISPLAY_MODEL = "GPT-5.6-Sol"
SOURCE_SHA256 = "79f961a2ce788fa2219e8ee5343b7fa87ca8d79ed3f3dec6049dca0ff7514ad9"
TARGET_CORRECT = 10
MAX_CONSECUTIVE_WRONG = 10
MAX_TOTAL_WRONG = 20
INITIAL_CONCURRENCY = 4
MAX_CONCURRENCY = 4
ATTEMPT_TIMEOUT_SECONDS = 45 * 60
RATE_BACKOFF_SECONDS = (60, 120, 300, 600, 1200, 1800)
QUOTA_MAX_WAIT_SECONDS = 48 * 60 * 60
OLD_EXPERIMENTS = (
    REPOSITORY_ROOT / "experiments" / "2026-07-27-ip_codex_train0629_14x10",
    REPOSITORY_ROOT / "experiments" / "2026-07-28-ip_codex_train0629_10x10",
    REPOSITORY_ROOT / "experiments" / "2026-07-28-ip_codex_train0629_100x10",
)

RATE_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
)
NETWORK_PATTERNS = (
    "os error 11001",
    "不知道这样的主机",
    "no such host",
    "name or service not known",
    "temporary failure in name resolution",
    "failed to lookup address information",
    "failed to connect to websocket",
    "stream disconnected before completion",
    "connection reset",
    "connection refused",
)
QUOTA_PATTERNS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient credits",
    "out of credits",
    "usage limit",
    "usage_limit",
    "hit your usage limit",
    "reached your usage limit",
    "quota exceeded",
    "额度不足",
    "用量限制",
    "达到使用上限",
    "余额不足",
    "配额不足",
)
AUTH_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "login required",
    "not logged in",
    "401",
)
MODEL_PATTERNS = (
    "model not found",
    "unknown model",
    "unsupported model",
    "does not exist",
    "not available for this account",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def log(message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with RUNNER_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def relative(path: Path) -> str:
    return path.resolve().relative_to(EXPERIMENT_ROOT.resolve()).as_posix()


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_runner_lock() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            existing = load_json(LOCK_PATH)
        except Exception:
            existing = {}
        old_pid = int(existing.get("pid") or 0)
        old_status = str(existing.get("status") or "")
        if old_status == "running" and process_alive(old_pid):
            raise RuntimeError(f"another experiment runner is active with PID {old_pid}")
        stale_name = REPORT_DIR / (
            "runner.lock.stale." + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
        )
        LOCK_PATH.replace(stale_name)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(LOCK_PATH, flags)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {"pid": os.getpid(), "status": "running", "started_at": utc_now()},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


def finish_runner_lock(status: str) -> None:
    atomic_json(
        LOCK_PATH,
        {"pid": os.getpid(), "status": status, "updated_at": utc_now()},
    )


def git_status_outside_target() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("git status failed during integrity capture")
    target = EXPERIMENT_ROOT.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    return sorted(
        line
        for line in completed.stdout.splitlines()
        if target not in line.replace("\\", "/")
    )


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def create_baseline() -> None:
    if BASELINE_PATH.exists():
        return
    baseline = {
        "schema_version": "ip-distill-integrity-baseline.v1",
        "captured_at": utc_now(),
        "source_path": str(DATASET.resolve()),
        "source_sha256": sha256(DATASET),
        "git_status_outside_target": git_status_outside_target(),
        "protected_trees": {
            "data/simulation": tree_hashes(REPOSITORY_ROOT / "data" / "simulation"),
            **{
                path.relative_to(REPOSITORY_ROOT).as_posix(): tree_hashes(path)
                for path in OLD_EXPERIMENTS
            },
        },
    }
    atomic_json(BASELINE_PATH, baseline)


def check_model_cache() -> dict[str, str]:
    cache = json.loads(MODEL_CACHE.read_text(encoding="utf-8"))
    models = cache.get("models", []) if isinstance(cache, dict) else []
    for item in models:
        if isinstance(item, dict) and item.get("slug") == MODEL:
            return {
                "slug": str(item.get("slug")),
                "display_name": str(item.get("display_name")),
                "cache_client_version": str(cache.get("client_version", "")),
                "cache_fetched_at": str(cache.get("fetched_at", "")),
            }
    raise RuntimeError(f"local Codex model cache does not contain exact slug {MODEL!r}")


def locate_and_copy_codex() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not CLI_PATH.exists():
        located = shutil.which("codex")
        if not located:
            raise RuntimeError("local Codex CLI is not on PATH")
        shutil.copy2(Path(located), CLI_PATH)
    version = subprocess.run(
        [str(CLI_PATH), "--version"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if version.returncode != 0:
        raise RuntimeError(f"copied Codex CLI cannot run: {version.stderr.strip()}")
    login = subprocess.run(
        [str(CLI_PATH), "login", "status"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if login.returncode != 0:
        raise RuntimeError("Codex CLI authentication status check failed")
    model_info = check_model_cache()
    if model_info["display_name"] != DISPLAY_MODEL:
        raise RuntimeError(
            f"model display name mismatch: {model_info['display_name']!r}"
        )
    atomic_json(
        REPORT_DIR / "cli_preflight.json",
        {
            "checked_at": utc_now(),
            "cli_version": version.stdout.strip() or version.stderr.strip(),
            "cli_sha256": sha256(CLI_PATH),
            "login_status": "authenticated",
            "credential_environment_presence": {
                key: key in os.environ
                for key in (
                    "OPENAI_API_KEY",
                    "CODEX_API_KEY",
                    "OPENAI_BASE_URL",
                    "AZURE_OPENAI_API_KEY",
                )
            },
            "model_confirmation": model_info,
        },
    )


def service_health(base_url: str) -> bool:
    request = urllib.request.Request(
        f"{base_url}/healthz",
        headers={"User-Agent": "ip-distill-runner/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            server = response.headers.get("Server", "")
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    return payload == {"status": "ok"} and "SavedConfigsService/" in server


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


class ServiceManager:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.managed = False
        self.base_url = ""
        self.port = 0

    def ensure(self, preferred_url: str | None = None) -> str:
        if preferred_url:
            self.base_url = preferred_url
            self.port = int(preferred_url.rsplit(":", 1)[1])
            if service_health(preferred_url):
                self._record("reused_existing_or_prior")
                return preferred_url
            if not port_is_free(self.port):
                raise RuntimeError(
                    f"required service port {self.port} is occupied by a non-matching service"
                )
            return self._start(self.port)
        default = "http://127.0.0.1:3080"
        if service_health(default):
            self.base_url = default
            self.port = 3080
            self._record("reused_existing")
            return default
        port = 3080 if port_is_free(3080) else free_port()
        return self._start(port)

    def _start(self, port: int) -> str:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        stdout_path = REPORT_DIR / "service.stdout.log"
        stderr_path = REPORT_DIR / "service.stderr.log"
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = str(RUNTIME_DIR / "pycache")
        command = [
            sys.executable,
            "-B",
            str(SERVICE_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with (
            stdout_path.open("a", encoding="utf-8", newline="\n") as stdout_handle,
            stderr_path.open("a", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            self.process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
            )
        self.managed = True
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"saved_configs_service exited with code {self.process.returncode}"
                )
            if service_health(self.base_url):
                self._record("started_by_experiment")
                return self.base_url
            time.sleep(0.5)
        raise RuntimeError("saved_configs_service did not become healthy within 30 seconds")

    def restart_if_owned(self) -> bool:
        if service_health(self.base_url):
            return True
        if not self.managed:
            return False
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=10)
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
        if not port_is_free(self.port):
            return False
        self._start(self.port)
        return True

    def _record(self, action: str) -> None:
        existing: dict[str, Any] = {}
        path = REPORT_DIR / "service.json"
        if path.exists():
            with contextlib.suppress(Exception):
                existing = load_json(path)
        history = list(existing.get("history", []))
        history.append(
            {
                "at": utc_now(),
                "action": action,
                "address": self.base_url,
                "pid": self.process.pid if self.process else existing.get("pid"),
            }
        )
        atomic_json(
            path,
            {
                "schema_version": "ip-distill-service.v1",
                "address": self.base_url,
                "pid": self.process.pid if self.process else existing.get("pid"),
                "managed_by_current_runner": self.managed,
                "healthy": service_health(self.base_url),
                "updated_at": utc_now(),
                "stdout": "service.stdout.log",
                "stderr": "service.stderr.log",
                "history": history,
            },
        )

    def stop(self) -> None:
        if not self.managed or self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self._record("stopped_by_experiment")


def run_input_broker(base_url: str) -> dict[str, Any]:
    if SAFE_INDEX.exists() and LOCALIZED_TEMPLATE.exists():
        return load_json(SAFE_INDEX)
    command = [
        sys.executable,
        "-B",
        str(SCRIPTS_DIR / "input_broker.py"),
        "--dataset",
        str(DATASET),
        "--template",
        str(SOURCE_TEMPLATE),
        "--base-url",
        base_url,
        "--output",
        str(SAFE_INDEX),
        "--localized-template",
        str(LOCALIZED_TEMPLATE),
    ]
    completed = subprocess.run(
        command,
        cwd=EXPERIMENT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    atomic_text(REPORT_DIR / "input_broker.stdout.log", completed.stdout)
    atomic_text(REPORT_DIR / "input_broker.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError("safe input broker failed; inspect its stderr log")
    return load_json(SAFE_INDEX)


def initial_sample(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": record["row_index"],
        "original_id": record["original_id"],
        "sample_id": record["sample_id"],
        "sample_key": record["sample_key"],
        "accepted_count": 0,
        "consecutive_wrong": 0,
        "total_wrong": 0,
        "total_attempts": 0,
        "total_model_attempts": 0,
        "infrastructure_failures": 0,
        "next_attempt_number": 1,
        "status": "pending",
        "accepted_attempts": [],
        "current_attempt": None,
        "last_error_class": None,
        "updated_at": utc_now(),
    }


def wrong_threshold_status(consecutive_wrong: int, total_wrong: int) -> str:
    if consecutive_wrong >= MAX_CONSECUTIVE_WRONG:
        return "abandoned_after_10_consecutive_wrong"
    if total_wrong >= MAX_TOTAL_WRONG:
        return "abandoned_after_20_total_wrong"
    return "pending"


def initialize_or_validate(base_url: str, safe_index: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_hash = sha256(DATASET)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(f"source hash changed before generation: {source_hash}")
    cli = load_json(REPORT_DIR / "cli_preflight.json")
    if MANIFEST_PATH.exists():
        manifest = load_json(MANIFEST_PATH)
        expected = {
            "source_sha256": SOURCE_SHA256,
            "model": MODEL,
            "target_correct_per_sample": TARGET_CORRECT,
            "record_count": 100,
            "service_base_url": base_url,
            "max_consecutive_wrong": MAX_CONSECUTIVE_WRONG,
            "max_total_wrong": MAX_TOTAL_WRONG,
        }
        mismatches = {
            key: {"existing": manifest.get(key), "required": value}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            conflict = REPORT_DIR / (
                "STOPPED_CONFLICT_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
            )
            atomic_json(
                conflict,
                {
                    "stopped_at": utc_now(),
                    "reason": "existing manifest is incompatible",
                    "mismatches": mismatches,
                    "recovery": "use the manifest-matching environment or a different experiment directory; do not overwrite this directory",
                },
            )
            raise RuntimeError("existing experiment manifest conflicts with requested task")
        state = load_json(STATE_PATH)
        return manifest, state

    manifest = {
        "schema_version": "ip-codex-distillation-manifest.v2",
        "status": "running",
        "started_at": utc_now(),
        "experiment_root": str(EXPERIMENT_ROOT.resolve()),
        "source_path": str(DATASET.resolve()),
        "source_sha256": source_hash,
        "record_count": 100,
        "row_range": [1, 100],
        "model": MODEL,
        "model_display_name": DISPLAY_MODEL,
        "model_confirmation": cli["model_confirmation"],
        "codex_cli_version": cli["cli_version"],
        "codex_cli_sha256": cli["cli_sha256"],
        "service_base_url": base_url,
        "initial_concurrency": INITIAL_CONCURRENCY,
        "maximum_concurrency": MAX_CONCURRENCY,
        "target_correct_per_sample": TARGET_CORRECT,
        "max_consecutive_wrong": MAX_CONSECUTIVE_WRONG,
        "max_total_wrong": MAX_TOTAL_WRONG,
        "attempt_timeout_seconds": ATTEMPT_TIMEOUT_SECONDS,
        "ephemeral_sessions": True,
        "fresh_process_per_attempt": True,
        "generator_input_fields": ["question", "output_format", "optimized copied user prompt", "local API URL"],
        "prompt_source": safe_index["prompt_source"],
        "prompt_localization": safe_index["prompt_localization"],
        "comparator": "established_fault_set_exact_with_alternatives_v2",
        "comparison_rule": "prediction must be a JSON string array exactly equal to the flat reference or to any explicitly listed nested reference alternative; order ignored; missing, extra, or different items fail",
        "generator_judge_separation": "separate judge_attempt.py process starts only after Codex exits",
        "runs_layout": "results/runs/qXXXX_rYY/attempt_ZZZ",
    }
    state = {
        "schema_version": "ip-codex-distillation-state.v2",
        "status": "running",
        "started_at": manifest["started_at"],
        "updated_at": utc_now(),
        "current_concurrency": INITIAL_CONCURRENCY,
        "concurrency_history": [
            {"at": utc_now(), "from": None, "to": INITIAL_CONCURRENCY, "reason": "initial"}
        ],
        "rate_backoff_index": 0,
        "global_pause_until": None,
        "quota_first_seen_at": None,
        "quota_wait_seconds": 0,
        "rate_limit_count": 0,
        "timeout_count": 0,
        "blocker_probe_counts": {},
        "samples": [initial_sample(record) for record in safe_index["records"]],
    }
    atomic_json(MANIFEST_PATH, manifest)
    atomic_json(STATE_PATH, state)
    write_accepted_index(state)
    return manifest, state


def write_accepted_index(state: dict[str, Any]) -> None:
    samples = []
    for sample in state["samples"]:
        mapping = {
            f"success_{index:02d}": item
            for index, item in enumerate(sample["accepted_attempts"], start=1)
        }
        samples.append(
            {
                "row_index": sample["row_index"],
                "original_id": sample["original_id"],
                "sample_key": sample["sample_key"],
                "accepted_count": sample["accepted_count"],
                "mapping": mapping,
            }
        )
    atomic_json(
        ACCEPTED_PATH,
        {
            "schema_version": "ip-distill-accepted-index.v1",
            "updated_at": utc_now(),
            "samples": samples,
        },
    )


def recover_interrupted(state: dict[str, Any]) -> None:
    changed = False
    for sample in state["samples"]:
        current = sample.get("current_attempt")
        if not isinstance(current, dict):
            continue
        attempt_dir = EXPERIMENT_ROOT / current["attempt_path"]
        metadata_path = attempt_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with contextlib.suppress(Exception):
                metadata = load_json(metadata_path)
        metadata.update(
            {
                "status": "interrupted",
                "generation_status": "interrupted",
                "judge_status": "not_run_interrupted",
                "error_class": "interrupted",
                "ended_at": utc_now(),
            }
        )
        for name in ("events.jsonl", "stderr.log", "final_answer.txt"):
            path = attempt_dir / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        atomic_json(metadata_path, metadata)
        atomic_json(
            attempt_dir / "recovery.json",
            {
                "recorded_at": utc_now(),
                "status": "interrupted",
                "reason": "prior runner ended before atomic attempt completion",
                "resume": "a new monotonically larger attempt number will be used",
            },
        )
        workspace = attempt_dir / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        sample["current_attempt"] = None
        sample["infrastructure_failures"] += 1
        sample["status"] = "pending"
        sample["last_error_class"] = "interrupted"
        sample["updated_at"] = utc_now()
        changed = True
    if changed:
        state["updated_at"] = utc_now()
        atomic_json(STATE_PATH, state)


def resume_infrastructure_checkpoint(manifest: dict[str, Any], state: dict[str, Any]) -> None:
    """Reopen non-terminal samples after the caller reruns a stopped checkpoint."""
    prior_blocker = str(state.get("blocker_reason") or "unknown")
    resumed_at = utc_now()
    for sample in state["samples"]:
        if sample["status"] != "stopped_by_infrastructure_blocker":
            continue
        if int(sample["accepted_count"]) >= TARGET_CORRECT:
            sample["status"] = "completed_with_10_correct"
        else:
            sample["status"] = wrong_threshold_status(
                int(sample["consecutive_wrong"]), int(sample["total_wrong"])
            )
        sample["updated_at"] = resumed_at

    state["status"] = "running"
    state["updated_at"] = resumed_at
    state["global_pause_until"] = None
    state["pause_reason"] = None
    state["blocker_probe_counts"] = {}
    state.pop("blocker_reason", None)
    state.setdefault("resume_history", []).append(
        {"resumed_at": resumed_at, "prior_blocker": prior_blocker}
    )
    manifest["status"] = "running"
    manifest["updated_at"] = resumed_at
    manifest.setdefault("resume_history", []).append(
        {"resumed_at": resumed_at, "prior_blocker": prior_blocker}
    )
    if STOPPED_PATH.exists():
        resolved_path = REPORT_DIR / (
            "STOPPED.resolved." + datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"
        )
        STOPPED_PATH.replace(resolved_path)
    atomic_json(STATE_PATH, state)
    atomic_json(MANIFEST_PATH, manifest)
    write_accepted_index(state)
    log(f"resumed infrastructure checkpoint after: {prior_blocker}")


def instantiate_prompt(template: str, record: dict[str, Any]) -> str:
    prompt = template
    replacements = {
        "{original_query}": record["question"],
        "{output_format}": record["output_format"],
    }
    for placeholder, value in replacements.items():
        if prompt.count(placeholder) != 1:
            raise RuntimeError(f"localized template has invalid {placeholder}")
        prompt = prompt.replace(placeholder, value)
    return prompt


def codex_command(workspace: Path, final_answer: Path) -> list[str]:
    trust = f"projects.'{workspace.resolve()}'.trust_level=\"trusted\""
    return [
        str(CLI_PATH),
        "exec",
        "--ephemeral",
        "--ignore-rules",
        "--dangerously-bypass-hook-trust",
        "--enable",
        "hooks",
        "--disable",
        "fast_mode",
        "--skip-git-repo-check",
        "--cd",
        str(workspace.resolve()),
        "-c",
        trust,
        "-c",
        'web_search="disabled"',
        "-c",
        "mcp_servers.openaiDeveloperDocs.enabled=false",
        "--json",
        "--sandbox",
        "danger-full-access",
        "--output-last-message",
        str(final_answer.resolve()),
        "--model",
        MODEL,
        "-",
    ]


def create_attempt(
    state: dict[str, Any],
    sample: dict[str, Any],
    record: dict[str, Any],
    template: str,
    base_url: str,
) -> dict[str, Any]:
    success_slot = int(sample["accepted_count"]) + 1
    attempt_number = int(sample["next_attempt_number"])
    slot_dir = RUNS_DIR / f"{sample['sample_key']}_r{success_slot:02d}"
    attempt_dir = slot_dir / f"attempt_{attempt_number:03d}"
    if attempt_dir.exists():
        raise RuntimeError(f"attempt directory already exists: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    workspace = attempt_dir / "workspace"
    # Codex 0.146 discovers project-local hooks only from an active project
    # config layer.  An empty Git marker is sufficient for this isolated,
    # answer-free workspace and is deleted with the workspace after the run.
    (workspace / ".git").mkdir(parents=True)
    hooks_dir = workspace / ".codex"
    hooks_dir.mkdir(parents=True)
    prompt = instantiate_prompt(template, record)
    source_record = {
        "row_index": sample["row_index"],
        "original_id": sample["original_id"],
        "sample_id": sample["sample_id"],
        "sample_key": sample["sample_key"],
        "question": record["question"],
        "output_format": record["output_format"],
        "contains_ground_answer": False,
    }
    question_dir = QUESTIONS_DIR / sample["sample_key"]
    question_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = question_dir / "prompt.txt"
    source_record_path = question_dir / "source_record.json"
    if not prompt_path.exists():
        atomic_text(prompt_path, prompt)
    elif prompt_path.read_text(encoding="utf-8") != prompt:
        raise RuntimeError(f"resume prompt mismatch in {question_dir}")
    if not source_record_path.exists():
        atomic_json(source_record_path, source_record)
    elif load_json(source_record_path) != source_record:
        raise RuntimeError(f"resume source record mismatch in {question_dir}")
    for name in (
        "events.jsonl",
        "stderr.log",
        "final_answer.txt",
        "hook_audit.jsonl",
    ):
        (attempt_dir / name).touch()
    hook_command = subprocess.list2cmdline(
        [sys.executable, "-B", str(SCRIPTS_DIR / "api_only_hook.py")]
    )
    atomic_json(
        hooks_dir / "hooks.json",
        {
            "description": "Allow only read-only saved_configs_service API queries.",
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "*",
                        "hooks": [
                            {
                                "type": "command",
                                "command": hook_command,
                                "commandWindows": hook_command,
                                "statusMessage": "Enforcing local API-only evidence access",
                                "timeout": 10,
                            }
                        ],
                    }
                ]
            },
        },
    )
    runtime_hook_config_sha256 = sha256(hooks_dir / "hooks.json")
    command = codex_command(workspace, attempt_dir / "final_answer.txt")
    metadata = {
        "schema_version": "ip-distill-attempt.v1",
        "row_index": sample["row_index"],
        "original_id": sample["original_id"],
        "sample_id": sample["sample_id"],
        "sample_key": sample["sample_key"],
        "target_success_slot": success_slot,
        "attempt_index": attempt_number,
        "status": "reserved",
        "generation_status": "not_started",
        "judge_status": "not_started",
        "error_class": None,
        "started_at": utc_now(),
        "model": MODEL,
        "cli_version": load_json(REPORT_DIR / "cli_preflight.json")["cli_version"],
        "service_base_url": base_url,
        "working_directory": str(workspace.resolve()),
        "ephemeral_session": True,
        "command": command,
        "prompt_sha256": sha256(prompt_path),
        "source_record_sha256": sha256(source_record_path),
        "contains_ground_answer": False,
        "api_only_hook_sha256": sha256(SCRIPTS_DIR / "api_only_hook.py"),
        "runtime_hook_config_sha256": runtime_hook_config_sha256,
        "files": {
            "prompt": Path(os.path.relpath(prompt_path, attempt_dir)).as_posix(),
            "source_record": Path(
                os.path.relpath(source_record_path, attempt_dir)
            ).as_posix(),
            "events": "events.jsonl",
            "stdout": "events.jsonl",
            "stderr": "stderr.log",
            "final_answer": "final_answer.txt",
            "hook_audit": "hook_audit.jsonl",
            "hook_audit_parts": "hook_audit_parts/",
            "judgment": "judgment.json",
        },
    }
    atomic_json(attempt_dir / "metadata.json", metadata)
    sample["total_attempts"] += 1
    sample["next_attempt_number"] += 1
    sample["status"] = "running"
    sample["current_attempt"] = {
        "attempt_index": attempt_number,
        "target_success_slot": success_slot,
        "attempt_path": relative(attempt_dir),
        "reserved_at": utc_now(),
    }
    sample["updated_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json(STATE_PATH, state)
    return {
        "sample_key": sample["sample_key"],
        "row_index": sample["row_index"],
        "original_id": sample["original_id"],
        "target_success_slot": success_slot,
        "attempt_index": attempt_number,
        "attempt_dir": attempt_dir,
        "workspace": workspace,
        "prompt": prompt,
        "command": command,
        "base_url": base_url,
    }



def parse_events(path: Path) -> dict[str, Any]:
    thread_id: str | None = None
    usage: dict[str, Any] | None = None
    invalid_lines: list[int] = []
    error_events = 0
    error_event_texts: list[str] = []
    event_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            if not isinstance(event, dict):
                invalid_lines.append(line_number)
                continue
            event_type = str(event.get("type", "unknown"))
            event_counts[event_type] += 1
            if event_type == "thread.started" and isinstance(event.get("thread_id"), str):
                thread_id = event["thread_id"]
            if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if event_type in {"error", "turn.failed"}:
                error_events += 1
                error_event_texts.append(
                    json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                )
            item = event.get("item")
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                item_counts[item["type"]] += 1
    return {
        "thread_id": thread_id,
        "usage": usage,
        "invalid_jsonl_lines": invalid_lines,
        "error_event_count": error_events,
        "error_event_texts": error_event_texts,
        "event_type_counts": dict(sorted(event_counts.items())),
        "item_type_counts": dict(sorted(item_counts.items())),
    }


def consolidate_hook_audit(parts_dir: Path, output_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    invalid_fragments = 0
    if parts_dir.is_dir():
        for fragment in sorted(parts_dir.glob("*.json")):
            try:
                value = json.loads(fragment.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid_fragments += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                invalid_fragments += 1
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = hook_audit_status(output_path)
    result["invalid"] += invalid_fragments
    result["fragment_count"] = len(records) + invalid_fragments
    return result

def hook_audit_status(path: Path) -> dict[str, Any]:
    allowed = 0
    denied = 0
    invalid = 0
    sessions: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if record.get("allowed") is True:
                allowed += 1
            else:
                denied += 1
            if record.get("session_id"):
                sessions.add(str(record["session_id"]))
    return {
        "allowed": allowed,
        "denied": denied,
        "invalid": invalid,
        "sessions": sorted(sessions),
    }


def classify_infrastructure(text: str) -> str:
    folded = text.casefold()
    if any(value in folded for value in NETWORK_PATTERNS):
        return "network_failure"
    if any(value in folded for value in QUOTA_PATTERNS):
        return "quota_exhausted"
    if any(value in folded for value in RATE_PATTERNS):
        return "rate_limited_429"
    if any(value in folded for value in AUTH_PATTERNS):
        return "authentication_failure"
    if any(value in folded for value in MODEL_PATTERNS):
        return "model_unavailable"
    return "cli_failure"


def execute_attempt(task: dict[str, Any]) -> dict[str, Any]:
    attempt_dir: Path = task["attempt_dir"]
    metadata_path = attempt_dir / "metadata.json"
    metadata = load_json(metadata_path)
    started_monotonic = time.monotonic()
    model_launched = False
    timed_out = False
    exit_code: int | None = None
    launch_error: str | None = None
    if not service_health(task["base_url"]):
        metadata.update(
            {
                "status": "infrastructure_failure",
                "generation_status": "not_started_service_unavailable",
                "judge_status": "not_run",
                "error_class": "service_unavailable",
                "ended_at": utc_now(),
                "duration_seconds": 0.0,
                "exit_code": None,
            }
        )
        atomic_json(metadata_path, metadata)
        atomic_text(attempt_dir / "exit_code.txt", "not_started\n")
        atomic_json(attempt_dir / "timing.json", {"started_at": metadata["started_at"], "ended_at": metadata["ended_at"], "duration_seconds": 0.0})
        return {**task, "outcome": "infrastructure_failure", "error_class": "service_unavailable", "model_launched": False}

    env = os.environ.copy()
    env["IP_DISTILL_ALLOWED_BASE"] = task["base_url"]
    env["IP_DISTILL_HOOK_AUDIT"] = str((attempt_dir / "hook_audit_parts").resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str((attempt_dir / "workspace" / ".pycache").resolve())
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process: subprocess.Popen[str] | None = None
    metadata["status"] = "running"
    metadata["generation_status"] = "running"
    atomic_json(metadata_path, metadata)
    try:
        with (
            (attempt_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as events_handle,
            (attempt_dir / "stderr.log").open("w", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            process = subprocess.Popen(
                task["command"],
                cwd=task["workspace"],
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            model_launched = True
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Codex process pipes were not created")
            process.stdin.write(task["prompt"])
            process.stdin.close()

            timeout_signal = threading.Event()

            def terminate_after_timeout() -> None:
                if process is None or process.poll() is not None:
                    return
                timeout_signal.set()
                process.terminate()
                time.sleep(15)
                if process.poll() is None:
                    process.kill()

            timeout_timer = threading.Timer(ATTEMPT_TIMEOUT_SECONDS, terminate_after_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()
            try:
                # Match the two old runners exactly: every raw Codex JSONL stdout
                # line is written once, in arrival order, and immediately flushed.
                for line in process.stdout:
                    events_handle.write(line)
                    events_handle.flush()
                exit_code = process.wait()
            finally:
                timeout_timer.cancel()
            timed_out = timeout_signal.is_set()
    except Exception as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
        exit_code = process.returncode if process is not None else None

    ended_at = utc_now()
    duration = round(time.monotonic() - started_monotonic, 3)
    event_info = parse_events(attempt_dir / "events.jsonl")
    audit = consolidate_hook_audit(attempt_dir / "hook_audit_parts", attempt_dir / "hook_audit.jsonl")
    stderr_text = (attempt_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
    failure_text = "\n".join(
        [stderr_text, *event_info["error_event_texts"], launch_error or ""]
    )

    error_class: str | None = None
    outcome = "judge_pending"
    if timed_out:
        error_class = "timeout"
        outcome = "infrastructure_failure"
    elif launch_error is not None:
        error_class = "cli_launch_failure"
        outcome = "infrastructure_failure"
    elif exit_code != 0:
        error_class = classify_infrastructure(failure_text)
        outcome = "infrastructure_failure"
    elif event_info["invalid_jsonl_lines"] or event_info["error_event_count"]:
        error_class = classify_infrastructure(failure_text)
        outcome = "infrastructure_failure"
    elif not event_info["thread_id"]:
        error_class = "missing_thread_id"
        outcome = "infrastructure_failure"
    elif audit["invalid"]:
        error_class = "hook_audit_invalid"
        outcome = "infrastructure_failure"
    elif audit["allowed"] == 0:
        error_class = "no_service_api_query"
        outcome = "infrastructure_failure"

    judgment: dict[str, Any] | None = None
    if outcome == "judge_pending":
        judge_command = [
            sys.executable,
            "-B",
            str(SCRIPTS_DIR / "judge_attempt.py"),
            "--dataset",
            str(DATASET),
            "--row-index",
            str(task["row_index"]),
            "--original-id",
            str(task["original_id"]),
            "--final-answer",
            str(attempt_dir / "final_answer.txt"),
            "--output",
            str(attempt_dir / "judgment.json"),
        ]
        judge = subprocess.run(
            judge_command,
            cwd=attempt_dir,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        atomic_text(attempt_dir / "judge.stdout.log", judge.stdout)
        atomic_text(attempt_dir / "judge.stderr.log", judge.stderr)
        if judge.returncode != 0 or not (attempt_dir / "judgment.json").exists():
            error_class = "judge_failure"
            outcome = "infrastructure_failure"
        else:
            judgment = load_json(attempt_dir / "judgment.json")
            if judgment.get("correct") is True:
                outcome = "correct"
            elif judgment.get("parsed") is True:
                outcome = "incorrect"
                error_class = "incorrect_result"
            else:
                outcome = "format_error"
                error_class = "format_error"

    metadata.update(
        {
            "status": (
                "accepted" if outcome == "correct" else "rejected" if outcome in {"incorrect", "format_error"} else "infrastructure_failure"
            ),
            "generation_status": "completed" if exit_code == 0 and model_launched and not timed_out else "failed",
            "judge_status": "completed" if judgment is not None else "not_run_or_failed",
            "error_class": error_class,
            "ended_at": ended_at,
            "duration_seconds": duration,
            "exit_code": exit_code,
            "model_process_started": model_launched,
            "timed_out": timed_out,
            "launch_error": launch_error,
            "thread_id": event_info["thread_id"],
            "usage": event_info["usage"],
            "event_type_counts": event_info["event_type_counts"],
            "item_type_counts": event_info["item_type_counts"],
            "invalid_jsonl_lines": event_info["invalid_jsonl_lines"],
            "error_event_count": event_info["error_event_count"],
            "hook_audit": audit,
            "sha256": {
                **{
                    "prompt": sha256(
                        QUESTIONS_DIR / task["sample_key"] / "prompt.txt"
                    ),
                    "source_record": sha256(
                        QUESTIONS_DIR
                        / task["sample_key"]
                        / "source_record.json"
                    ),
                },
                **{
                    name: sha256(attempt_dir / filename)
                    for name, filename in {
                    "events": "events.jsonl",
                    "stderr": "stderr.log",
                    "final_answer": "final_answer.txt",
                    "hook_audit": "hook_audit.jsonl",
                    }.items()
                },
            },
        }
    )
    atomic_json(metadata_path, metadata)
    atomic_text(attempt_dir / "exit_code.txt", (str(exit_code) if exit_code is not None else "not_started") + "\n")
    atomic_json(
        attempt_dir / "timing.json",
        {
            "started_at": metadata["started_at"],
            "ended_at": ended_at,
            "duration_seconds": duration,
        },
    )
    return {
        **task,
        "outcome": outcome,
        "error_class": error_class,
        "model_launched": model_launched,
        "thread_id": event_info["thread_id"],
        "duration_seconds": duration,
    }


def update_slot_run(task: dict[str, Any], sample: dict[str, Any], accepted: bool) -> None:
    slot_dir = task["attempt_dir"].parent
    existing: dict[str, Any] = {}
    run_path = slot_dir / "run.json"
    if run_path.exists():
        existing = load_json(run_path)
    attempts = list(existing.get("attempts", []))
    metadata = load_json(task["attempt_dir"] / "metadata.json")
    attempts.append(
        {
            "attempt_index": task["attempt_index"],
            "directory": task["attempt_dir"].name,
            "status": metadata["status"],
            "error_class": metadata.get("error_class"),
            "thread_id": metadata.get("thread_id"),
            "exit_code": metadata.get("exit_code"),
            "duration_seconds": metadata.get("duration_seconds"),
        }
    )
    atomic_json(
        run_path,
        {
            "schema_version": "ip-distill-success-slot.v1",
            "row_index": sample["row_index"],
            "original_id": sample["original_id"],
            "sample_key": sample["sample_key"],
            "success_slot": task["target_success_slot"],
            "status": "succeeded" if accepted else "pending",
            "successful_attempt": task["attempt_index"] if accepted else None,
            "attempts": attempts,
            "updated_at": utc_now(),
        },
    )


def record_concurrency(state: dict[str, Any], new_value: int, reason: str) -> None:
    old_value = int(state["current_concurrency"])
    if new_value == old_value:
        return
    state["current_concurrency"] = new_value
    state["concurrency_history"].append(
        {"at": utc_now(), "from": old_value, "to": new_value, "reason": reason}
    )


def schedule_pause(state: dict[str, Any], seconds: int, reason: str) -> None:
    seconds = max(1, int(seconds))
    target = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    existing = state.get("global_pause_until")
    if existing and parse_time(existing) > target:
        target = parse_time(existing)
    state["global_pause_until"] = target.isoformat(timespec="seconds")
    state["pause_reason"] = reason
    state["updated_at"] = utc_now()


def apply_outcome(state: dict[str, Any], task: dict[str, Any]) -> str | None:
    sample = next(item for item in state["samples"] if item["sample_key"] == task["sample_key"])
    sample["current_attempt"] = None
    if task["model_launched"]:
        sample["total_model_attempts"] += 1
    outcome = task["outcome"]
    blocker: str | None = None
    if outcome == "correct":
        sample["accepted_count"] += 1
        sample["consecutive_wrong"] = 0
        mapping = {
            "success_slot": task["target_success_slot"],
            "attempt_index": task["attempt_index"],
            "attempt_path": relative(task["attempt_dir"]),
            "events_path": relative(task["attempt_dir"] / "events.jsonl"),
            "thread_id": task.get("thread_id"),
        }
        sample["accepted_attempts"].append(mapping)
        sample["status"] = (
            "completed_with_10_correct"
            if sample["accepted_count"] == TARGET_CORRECT
            else "pending"
        )
        sample["last_error_class"] = None
        state["rate_backoff_index"] = 0
        update_slot_run(task, sample, accepted=True)
    elif outcome in {"incorrect", "format_error"}:
        sample["consecutive_wrong"] += 1
        sample["total_wrong"] += 1
        sample["last_error_class"] = task["error_class"]
        sample["status"] = wrong_threshold_status(
            sample["consecutive_wrong"], sample["total_wrong"]
        )
        update_slot_run(task, sample, accepted=False)
    else:
        sample["infrastructure_failures"] += 1
        sample["last_error_class"] = task["error_class"]
        sample["status"] = "pending"
        update_slot_run(task, sample, accepted=False)
        error_class = str(task["error_class"])
        if error_class == "rate_limited_429":
            state["rate_limit_count"] += 1
            index = min(int(state["rate_backoff_index"]), len(RATE_BACKOFF_SECONDS) - 1)
            delay = RATE_BACKOFF_SECONDS[index] + random.randint(0, 15)
            state["rate_backoff_index"] = min(index + 1, len(RATE_BACKOFF_SECONDS) - 1)
            record_concurrency(state, 2 if state["current_concurrency"] > 2 else 1, "rate_limit")
            schedule_pause(state, delay, "rate_limited_429")
        elif error_class == "quota_exhausted":
            if not state.get("quota_first_seen_at"):
                state["quota_first_seen_at"] = utc_now()
            elapsed = (datetime.now(timezone.utc) - parse_time(state["quota_first_seen_at"])).total_seconds()
            if elapsed >= QUOTA_MAX_WAIT_SECONDS:
                blocker = "quota_unavailable_after_48_hours"
            else:
                record_concurrency(state, 1, "quota_exhausted")
                schedule_pause(state, 3600, "quota_exhausted_hourly_probe")
                state["quota_wait_seconds"] += 3600
        elif error_class in {"authentication_failure", "model_unavailable"}:
            probes = state.setdefault("blocker_probe_counts", {})
            probes[error_class] = int(probes.get(error_class, 0)) + 1
            if probes[error_class] >= 3:
                blocker = f"{error_class}_after_3_checks"
            else:
                schedule_pause(state, (60, 300)[probes[error_class] - 1], error_class)
        elif error_class == "network_failure":
            schedule_pause(state, 60, "network_failure")
        elif error_class == "timeout":
            state["timeout_count"] += 1
            schedule_pause(state, 60, "attempt_timeout")
        elif error_class == "service_unavailable":
            schedule_pause(state, 60, "service_unavailable")
        elif error_class in {"cli_launch_failure", "missing_thread_id", "hook_audit_invalid"}:
            probes = state.setdefault("blocker_probe_counts", {})
            probes[error_class] = int(probes.get(error_class, 0)) + 1
            if probes[error_class] >= 3:
                blocker = f"{error_class}_after_3_checks"
            else:
                schedule_pause(state, 60 * probes[error_class], error_class)
        elif error_class in {"rejected_policy_violation", "no_service_api_query"}:
            schedule_pause(state, 5, error_class)
        else:
            schedule_pause(state, 60, error_class)
    sample["updated_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json(STATE_PATH, state)
    write_accepted_index(state)
    return blocker


def heartbeat(state: dict[str, Any], active: dict[Any, Any]) -> None:
    statuses = Counter(sample["status"] for sample in state["samples"])
    atomic_json(
        HEARTBEAT_PATH,
        {
            "at": utc_now(),
            "runner_pid": os.getpid(),
            "state_status": state["status"],
            "current_concurrency": state["current_concurrency"],
            "active_attempts": [
                {
                    "sample_key": task["sample_key"],
                    "attempt_index": task["attempt_index"],
                    "attempt_path": relative(task["attempt_dir"]),
                }
                for task in active.values()
            ],
            "status_counts": dict(sorted(statuses.items())),
            "accepted_total": sum(sample["accepted_count"] for sample in state["samples"]),
            "global_pause_until": state.get("global_pause_until"),
            "pause_reason": state.get("pause_reason"),
        },
    )


def mark_infrastructure_blocker(state: dict[str, Any], reason: str) -> None:
    for sample in state["samples"]:
        if sample["status"] not in {
            "completed_with_10_correct",
            "abandoned_after_10_consecutive_wrong",
            "abandoned_after_20_total_wrong",
        }:
            sample["status"] = "stopped_by_infrastructure_blocker"
            sample["updated_at"] = utc_now()
    state["status"] = "stopped_by_infrastructure_blocker"
    state["blocker_reason"] = reason
    state["updated_at"] = utc_now()
    atomic_json(STATE_PATH, state)
    atomic_json(
        STOPPED_PATH,
        {
            "stopped_at": utc_now(),
            "reason": reason,
            "checkpoint": relative(STATE_PATH),
            "resume_command": f"{sys.executable} -B {SCRIPTS_DIR / 'run_experiment.py'}",
            "recovery": "resolve the recorded infrastructure condition, then rerun the same command; no accepted attempt will be repeated",
        },
    )


def validate_integrity(state: dict[str, Any]) -> dict[str, Any]:
    baseline = load_json(BASELINE_PATH)
    protected: dict[str, Any] = {}
    for label, original in baseline["protected_trees"].items():
        current = tree_hashes(REPOSITORY_ROOT / label)
        protected[label] = {
            "unchanged": current == original,
            "initial_file_count": len(original),
            "current_file_count": len(current),
            "changed": sorted(
                key
                for key in set(original).union(current)
                if original.get(key) != current.get(key)
            ),
        }
    source_current = sha256(DATASET)
    outside_status = git_status_outside_target()
    accepted_paths: list[str] = []
    accepted_thread_ids: list[str] = []
    accepted_errors: list[str] = []
    all_thread_ids: list[str] = []
    wrong_model_attempts: list[str] = []
    missing_events: list[str] = []
    attempt_dirs = sorted(RUNS_DIR.glob("q*_r*/attempt_*")) + sorted(RUNS_DIR.glob("row*_r*/attempt_*"))
    for attempt_dir in attempt_dirs:
        if not attempt_dir.is_dir():
            continue
        if not (attempt_dir / "events.jsonl").is_file():
            missing_events.append(relative(attempt_dir))
        metadata_path = attempt_dir / "metadata.json"
        if metadata_path.exists():
            metadata = load_json(metadata_path)
            if metadata.get("model") != MODEL:
                wrong_model_attempts.append(relative(attempt_dir))
            if metadata.get("thread_id"):
                all_thread_ids.append(str(metadata["thread_id"]))
    for sample in state["samples"]:
        if len(sample["accepted_attempts"]) != sample["accepted_count"]:
            accepted_errors.append(f"{sample['sample_key']}: accepted count/list mismatch")
        if len(sample["accepted_attempts"]) != len({item["attempt_path"] for item in sample["accepted_attempts"]}):
            accepted_errors.append(f"{sample['sample_key']}: duplicate accepted attempt")
        for item in sample["accepted_attempts"]:
            attempt_path = str(item["attempt_path"])
            accepted_paths.append(attempt_path)
            attempt_dir = EXPERIMENT_ROOT / attempt_path
            judgment_path = attempt_dir / "judgment.json"
            metadata_path = attempt_dir / "metadata.json"
            if not judgment_path.exists() or load_json(judgment_path).get("correct") is not True:
                accepted_errors.append(f"{attempt_path}: missing positive judgment")
            if not metadata_path.exists() or load_json(metadata_path).get("status") != "accepted":
                accepted_errors.append(f"{attempt_path}: metadata is not accepted")
            if item.get("thread_id"):
                accepted_thread_ids.append(str(item["thread_id"]))
    terminal_allowed = {
        "completed_with_10_correct",
        "abandoned_after_10_consecutive_wrong",
        "abandoned_after_20_total_wrong",
        "stopped_by_infrastructure_blocker",
    }
    invalid_final_states = [
        sample["sample_key"] for sample in state["samples"] if sample["status"] not in terminal_allowed
    ]
    return {
        "validated_at": utc_now(),
        "source_record_count": load_json(SAFE_INDEX)["record_count"],
        "source_sha256_initial": baseline["source_sha256"],
        "source_sha256_current": source_current,
        "source_unchanged": source_current == baseline["source_sha256"],
        "protected_trees": protected,
        "git_status_outside_target_initial": baseline["git_status_outside_target"],
        "git_status_outside_target_current": outside_status,
        "outside_target_git_status_unchanged": outside_status == baseline["git_status_outside_target"],
        "attempt_directory_count": len([path for path in attempt_dirs if path.is_dir()]),
        "missing_events": missing_events,
        "wrong_model_attempts": wrong_model_attempts,
        "accepted_attempt_count": len(accepted_paths),
        "accepted_paths_unique": len(accepted_paths) == len(set(accepted_paths)),
        "accepted_thread_ids_unique": len(accepted_thread_ids) == len(set(accepted_thread_ids)),
        "all_thread_ids_unique": len(all_thread_ids) == len(set(all_thread_ids)),
        "accepted_errors": accepted_errors,
        "invalid_final_states": invalid_final_states,
        "passed": (
            source_current == baseline["source_sha256"]
            and all(item["unchanged"] for item in protected.values())
            and outside_status == baseline["git_status_outside_target"]
            and not missing_events
            and not wrong_model_attempts
            and len(accepted_paths) == len(set(accepted_paths))
            and len(accepted_thread_ids) == len(set(accepted_thread_ids))
            and not accepted_errors
            and not invalid_final_states
        ),
    }


def collect_attempt_statistics() -> dict[str, int]:
    counts: Counter[str] = Counter()
    for metadata_path in RUNS_DIR.glob("*_r*/attempt_*/metadata.json"):
        metadata = load_json(metadata_path)
        counts["total_attempts"] += 1
        if metadata.get("model_process_started"):
            counts["total_model_calls"] += 1
        status = metadata.get("status")
        error = metadata.get("error_class")
        if status == "accepted":
            counts["accepted"] += 1
        elif error == "incorrect_result":
            counts["incorrect"] += 1
        elif error == "format_error":
            counts["format_error"] += 1
        elif status in {"infrastructure_failure", "interrupted"}:
            counts["infrastructure_failure"] += 1
        if error == "timeout":
            counts["timeout"] += 1
        if error == "rate_limited_429":
            counts["rate_limited_429"] += 1
        if error == "quota_exhausted":
            counts["quota_exhausted"] += 1
    return dict(counts)


def write_final_report(manifest: dict[str, Any], state: dict[str, Any]) -> None:
    integrity = validate_integrity(state)
    statistics = collect_attempt_statistics()
    status_counts = Counter(sample["status"] for sample in state["samples"])
    summary = {
        "schema_version": "ip-distill-final-summary.v2",
        "generated_at": utc_now(),
        "experiment_status": state["status"],
        "experiment_root": str(EXPERIMENT_ROOT.resolve()),
        "input_path": str(DATASET.resolve()),
        "input_sha256": sha256(DATASET),
        "input_record_count": 100,
        "codex_cli_version": manifest["codex_cli_version"],
        "model": MODEL,
        "service_base_url": manifest["service_base_url"],
        "initial_concurrency": INITIAL_CONCURRENCY,
        "concurrency_history": state["concurrency_history"],
        "planned_samples": 100,
        "scheduled_samples": len(state["samples"]),
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_total": sum(sample["accepted_count"] for sample in state["samples"]),
        "quota_wait_seconds": state.get("quota_wait_seconds", 0),
        "attempt_statistics": statistics,
        "comparator": manifest["comparator"],
        "comparison_rule": manifest["comparison_rule"],
        "resume_state": relative(STATE_PATH),
        "integrity": integrity,
        "samples": state["samples"],
    }
    atomic_json(SUMMARY_PATH, summary)
    lines = [
        "# FINAL REPORT",
        "",
        f"- 实验目录：`{EXPERIMENT_ROOT.resolve()}`",
        f"- 实验状态：`{state['status']}`",
        f"- 输入文件：`{DATASET.resolve()}`",
        f"- 输入 SHA-256：`{summary['input_sha256']}`",
        "- 输入非空记录数：100",
        f"- Codex CLI：`{manifest['codex_cli_version']}`",
        f"- 模型准确标识：`{MODEL}`（{DISPLAY_MODEL}）",
        f"- saved_configs_service：`{manifest['service_base_url']}`",
        f"- 初始并发度：{INITIAL_CONCURRENCY}",
        f"- 计划/进入调度：100 / {len(state['samples'])}",
        f"- 完成 10 条正确轨迹：{status_counts.get('completed_with_10_correct', 0)} 题",
        f"- 连续 10 次错误后放弃：{status_counts.get('abandoned_after_10_consecutive_wrong', 0)} 题",
        f"- 累计 20 次错误后放弃：{status_counts.get('abandoned_after_20_total_wrong', 0)} 题",
        f"- 基础设施阻塞停止：{status_counts.get('stopped_by_infrastructure_blocker', 0)} 题",
        f"- 正确且 accepted：{summary['accepted_total']} 条",
        f"- 错误答案：{statistics.get('incorrect', 0)} 条",
        f"- 格式错误：{statistics.get('format_error', 0)} 条",
        f"- CLI/基础设施失败：{statistics.get('infrastructure_failure', 0)} 条",
        f"- 总 attempt / Codex CLI 调用：{statistics.get('total_attempts', 0)} / {statistics.get('total_model_calls', 0)}",
        f"- 超时：{statistics.get('timeout', 0)}；429：{statistics.get('rate_limited_429', 0)}；额度：{statistics.get('quota_exhausted', 0)}",
        f"- 额度等待累计秒数：{state.get('quota_wait_seconds', 0)}",
        f"- 判题器：`{manifest['comparator']}`；{manifest['comparison_rule']}",
        f"- 断点状态：`{relative(STATE_PATH)}`",
        "",
        "## 并发度调整",
        "",
    ]
    for item in state["concurrency_history"]:
        lines.append(f"- {item['at']}: {item['from']} -> {item['to']}（{item['reason']}）")
    lines.extend(["", "## 逐题状态", "", "| row | 原始 ID | accepted | 连续错误 | 累计错误 | 总 attempt | 模型调用 | 基础设施失败 | 最终状态 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for sample in state["samples"]:
        lines.append(
            f"| {sample['row_index']} | {sample['original_id']} | {sample['accepted_count']} | {sample['consecutive_wrong']} | {sample['total_wrong']} | {sample['total_attempts']} | {sample['total_model_attempts']} | {sample['infrastructure_failures']} | {sample['status']} |"
        )
    lines.extend(["", "## Accepted 唯一映射", ""])
    for sample in state["samples"]:
        mappings = ", ".join(
            f"success_{index:02d} -> {item['events_path']}"
            for index, item in enumerate(sample["accepted_attempts"], start=1)
        ) or "（无）"
        lines.append(f"- row {sample['row_index']} / ID {sample['original_id']}: {mappings}")
    accepted_lines = ['# ACCEPTED TRAJECTORIES', '', 'Only strict-correct accepted trajectories are listed. Rejected, format-error, interrupted, and infrastructure-failure attempts are excluded.', '']
    for sample in state["samples"]:
        for index, item in enumerate(sample["accepted_attempts"], start=1):
            accepted_lines.append(
                f"- row {sample['row_index']} / ID {sample['original_id']} / success_{index:02d} -> `{item['events_path']}`"
            )
    atomic_text(REPORT_DIR / 'ACCEPTED_TRAJECTORIES.md', '\n'.join(accepted_lines) + '\n')
    lines.extend(
        [
            "",
            "## 完整性验证",
            "",
            f"- train_0629.jsonl 未修改：{integrity['source_unchanged']}",
            f"- data/simulation 未修改：{integrity['protected_trees']['data/simulation']['unchanged']}",
            f"- 旧 14×10 实验未修改：{integrity['protected_trees']['experiments/2026-07-27-ip_codex_train0629_14x10']['unchanged']}",
            f"- 旧 10×10 实验未修改：{integrity['protected_trees']['experiments/2026-07-28-ip_codex_train0629_10x10']['unchanged']}",
            f"- 旧 100×10 实验未修改：{integrity['protected_trees']['experiments/2026-07-28-ip_codex_train0629_100x10']['unchanged']}",
            f"- 目标目录外 Git 状态与基线一致：{integrity['outside_target_git_status_unchanged']}",
            f"- accepted attempt 唯一且判题为真：{not integrity['accepted_errors'] and integrity['accepted_paths_unique']}",
            f"- 所有 attempt 模型均为 {MODEL}：{not integrity['wrong_model_attempts']}",
            f"- 所有 attempt 均有 events.jsonl：{not integrity['missing_events']}",
            f"- 综合完整性检查通过：{integrity['passed']}",
            "",
            "## 主要文件",
            "",
            f"- manifest：`{relative(MANIFEST_PATH)}`",
            f"- 全局 state：`{relative(STATE_PATH)}`",
            f"- accepted 索引：`{relative(ACCEPTED_PATH)}`",
            "- 正确轨迹 Markdown 清单：`results/report/ACCEPTED_TRAJECTORIES.md`",
            f"- 机器可读汇总：`{relative(SUMMARY_PATH)}`",
            "- 独立最终审计：`results/report/final_audit.json`",
            f"- 运行数据：`{relative(RUNS_DIR)}/`",
            "",
            f"仍需用户处理的硬性阻塞：{state.get('blocker_reason', '无')}",
            "",
        ]
    )
    atomic_text(FINAL_REPORT_PATH, "\n".join(lines))


def run_scheduler(manifest: dict[str, Any], state: dict[str, Any], safe_index: dict[str, Any], service: ServiceManager) -> None:
    records = {record["sample_key"]: record for record in safe_index["records"]}
    template = LOCALIZED_TEMPLATE.read_text(encoding="utf-8")
    active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    last_heartbeat = 0.0
    last_rate_event = time.monotonic()
    blocker: str | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        while True:
            terminal = all(
                sample["status"]
                in {
                    "completed_with_10_correct",
                    "abandoned_after_10_consecutive_wrong",
                    "abandoned_after_20_total_wrong",
                    "stopped_by_infrastructure_blocker",
                }
                for sample in state["samples"]
            )
            if terminal and not active:
                break

            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= 60:
                heartbeat(state, active)
                last_heartbeat = now_mono

            pause_until = state.get("global_pause_until")
            paused = bool(pause_until and parse_time(pause_until) > datetime.now(timezone.utc))
            if pause_until and not paused:
                state["global_pause_until"] = None
                state["pause_reason"] = None
                state["updated_at"] = utc_now()
                atomic_json(STATE_PATH, state)

            if (
                state["current_concurrency"] < MAX_CONCURRENCY
                and now_mono - last_rate_event >= 3600
                and not paused
            ):
                record_concurrency(state, int(state["current_concurrency"]) + 1, "stable_for_one_hour")
                atomic_json(STATE_PATH, state)
                last_rate_event = now_mono

            if not paused and not blocker:
                running_keys = {task["sample_key"] for task in active.values()}
                candidates = [
                    sample
                    for sample in state["samples"]
                    if sample["status"] == "pending" and sample["sample_key"] not in running_keys
                ]
                capacity = int(state["current_concurrency"]) - len(active)
                for sample in candidates[: max(0, capacity)]:
                    task = create_attempt(
                        state,
                        sample,
                        records[sample["sample_key"]],
                        template,
                        service.base_url,
                    )
                    future = executor.submit(execute_attempt, task)
                    active[future] = task
                    log(
                        f"scheduled {task['sample_key']} success_slot={task['target_success_slot']:02d} attempt={task['attempt_index']:03d}"
                    )

            if not active:
                if blocker:
                    break
                time.sleep(10 if paused else 1)
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
                    metadata = load_json(reserved["attempt_dir"] / "metadata.json")
                    metadata.update(
                        {
                            "status": "infrastructure_failure",
                            "generation_status": "worker_exception",
                            "judge_status": "not_run",
                            "error_class": "runner_worker_exception",
                            "ended_at": utc_now(),
                            "runner_exception_type": type(exc).__name__,
                        }
                    )
                    atomic_json(reserved["attempt_dir"] / "metadata.json", metadata)
                workspace = reserved["workspace"]
                if workspace.exists():
                    shutil.rmtree(workspace)
                log(
                    f"finished {result['sample_key']} attempt={result['attempt_index']:03d} outcome={result['outcome']} error={result.get('error_class')}"
                )
                outcome_blocker = apply_outcome(state, result)
                if result.get("error_class") == "rate_limited_429":
                    last_rate_event = time.monotonic()
                if result.get("error_class") == "service_unavailable":
                    with contextlib.suppress(Exception):
                        service.restart_if_owned()
                if outcome_blocker:
                    blocker = outcome_blocker

            if blocker and not active:
                mark_infrastructure_blocker(state, blocker)
                break

    if state["status"] != "stopped_by_infrastructure_blocker":
        state["status"] = "completed"
        state["completed_at"] = utc_now()
        state["updated_at"] = utc_now()
        manifest["status"] = "completed"
        manifest["ended_at"] = state["completed_at"]
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
        write_accepted_index(state)
    heartbeat(state, {})


def main() -> int:
    for path in (SCRIPTS_DIR, RUNTIME_DIR, REPORT_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    acquire_runner_lock()
    service = ServiceManager()
    manifest: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    try:
        create_baseline()
        locate_and_copy_codex()
        preferred_url = None
        if MANIFEST_PATH.exists():
            preferred_url = str(load_json(MANIFEST_PATH).get("service_base_url") or "") or None
        base_url = service.ensure(preferred_url)
        safe_index = run_input_broker(base_url)
        manifest, state = initialize_or_validate(base_url, safe_index)
        recover_interrupted(state)
        if state.get("status") == "completed":
            log("experiment already completed; running final integrity verification")
        else:
            if state.get("status") == "stopped_by_infrastructure_blocker":
                resume_infrastructure_checkpoint(manifest, state)
            state["status"] = "running"
            manifest["status"] = "running"
            atomic_json(STATE_PATH, state)
            atomic_json(MANIFEST_PATH, manifest)
            run_scheduler(manifest, state, safe_index, service)
        write_final_report(manifest, state)
        final_status = "completed" if state.get("status") == "completed" else "stopped"
        finish_runner_lock(final_status)
        log(f"final report written: {FINAL_REPORT_PATH}")
        return 0 if state.get("status") == "completed" else 2
    except KeyboardInterrupt:
        if state is not None:
            state["status"] = "interrupted"
            state["updated_at"] = utc_now()
            atomic_json(STATE_PATH, state)
        finish_runner_lock("interrupted")
        log("runner interrupted; checkpoint preserved")
        return 130
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if state is not None:
            mark_infrastructure_blocker(state, reason)
            if manifest is not None:
                manifest["status"] = "stopped"
                manifest["ended_at"] = utc_now()
                atomic_json(MANIFEST_PATH, manifest)
                with contextlib.suppress(Exception):
                    write_final_report(manifest, state)
        else:
            atomic_json(
                STOPPED_PATH,
                {
                    "stopped_at": utc_now(),
                    "reason": reason,
                    "checkpoint": relative(STATE_PATH) if STATE_PATH.exists() else None,
                    "recovery": "inspect preflight logs, resolve the blocker, and rerun the same command",
                },
            )
        finish_runner_lock("stopped")
        log(f"hard blocker: {reason}")
        return 2
    finally:
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
