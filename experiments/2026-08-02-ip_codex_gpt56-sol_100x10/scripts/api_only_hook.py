#!/usr/bin/env python3
"""Deny every Codex tool call except read-only queries to the configured API."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


FORBIDDEN_MARKERS = (
    "saved_configs",
    "train_0629",
    "data/simulation",
    "data\\simulation",
    "experiments",
    "get-content",
    "get-childitem",
    "select-string",
    "set-content",
    "add-content",
    "out-file",
    "remove-item",
    "copy-item",
    "move-item",
    "test-path",
    "resolve-path",
    "pathlib",
    "os.listdir",
    "os.scandir",
    "os.walk",
    "open(",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "git ",
    "rg ",
    "findstr",
    "type ",
    " dir ",
    " ls ",
    " cat ",
    " env",
    "set ",
    "../",
    "..\\",
    "invoke-expression",
    "iex ",
    "start-process",
    "new-object",
    "get-item",
    "get-command",
    "get-location",
    "get-variable",
    "get-process",
    "get-ciminstance",
    "get-wmiobject",
    "set-location",
    "push-location",
    "pop-location",
    "whoami",
    "python ",
    "py ",
    "node ",
    "cmd.exe",
    "bash ",
    "wsl ",
    "curl ",
    "curl.exe",
    "wget ",
    "certutil",
    "[environment]",
    "[system.environment]",
    "& ",
    "http://localhost",
)


def response_deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            ensure_ascii=False,
        )
    )


def evaluate(payload: dict[str, object], allowed_base: str) -> tuple[bool, str, str]:
    tool_name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    if tool_name != "Bash":
        return False, f"tool {tool_name!r} is disabled; use the local HTTP API", ""
    if not isinstance(tool_input, dict):
        return False, "shell input is not an object", ""
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return False, "shell command is missing", ""
    folded = command.casefold()
    if "powershell.exe" not in folded or " -command " not in folded:
        return False, "only the fixed PowerShell HTTP client path is allowed", command
    if "invoke-restmethod" not in folded:
        return False, "Invoke-RestMethod is required for service queries", command

    dotnet_types = {value.casefold() for value in re.findall(r"\[([A-Za-z0-9_.]+)\]", command)}
    if dotnet_types.difference({"uri", "system.uri"}):
        return False, "only the URI encoder .NET type is allowed", command
    static_methods = {value.casefold() for value in re.findall(r"::([A-Za-z0-9_]+)", command)}
    if static_methods.difference({"escapedatastring"}):
        return False, "non-URI static method call is forbidden", command
    marker = next((value for value in FORBIDDEN_MARKERS if value in folded), None)
    if marker:
        return False, f"forbidden marker {marker!r}; only the local API is allowed", command
    if re.search(r"(?<![<])>(?![=>])|>>", command):
        return False, "shell redirection is forbidden", command
    urls = re.findall(r"https?://[^\s'\"]+", command, flags=re.IGNORECASE)
    if not urls:
        return False, "no HTTP URL was found", command
    allowed = urlsplit(allowed_base)
    for raw in urls:
        parsed = urlsplit(raw.rstrip(");,`"))
        if (
            parsed.scheme != "http"
            or parsed.hostname != allowed.hostname
            or parsed.port != allowed.port
        ):
            return False, f"non-service URL is forbidden: {raw}", command
        if parsed.path == "/" or not parsed.path.startswith("/v3/"):
            return False, f"only saved_configs_service /v3 read APIs are allowed: {parsed.path}", command
    destructive = re.search(r"\b(post|put|patch|delete)\b", folded)
    if destructive:
        return False, "non-GET HTTP method is forbidden", command
    return True, "allowed local read-only API query", command


def append_audit(record: dict[str, object]) -> None:
    raw = os.environ.get("IP_DISTILL_HOOK_AUDIT")
    if not raw:
        return
    directory = Path(raw)
    directory.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    stem = f"{time.time_ns():020d}_{os.getpid()}_{uuid.uuid4().hex}"
    temporary = directory / f".{stem}.tmp"
    final = directory / f"{stem}.json"
    temporary.write_bytes(payload)
    temporary.replace(final)
def main() -> int:
    allowed_base = os.environ.get("IP_DISTILL_ALLOWED_BASE", "")
    if not allowed_base:
        response_deny("runner did not configure an allowed API base")
        return 0
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as exc:
        response_deny(f"hook input parse failure: {type(exc).__name__}")
        return 0
    allowed, reason, command = evaluate(payload, allowed_base)
    append_audit(
        {
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": payload.get("session_id"),
            "tool_name": payload.get("tool_name"),
            "allowed": allowed,
            "reason": reason,
            "command": command,
        }
    )
    if not allowed:
        response_deny(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
