#!/usr/bin/env python3
"""Allow only direct, read-only file inspection under saved_configs/."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_COMMANDS = {
    "get-childitem": "literalpath",
    "get-content": "literalpath",
    "select-string": "path",
    "test-path": "literalpath",
}
FORBIDDEN_MARKERS = (
    "invoke-restmethod",
    "invoke-webrequest",
    "start-bitstransfer",
    "curl ",
    "curl.exe",
    "wget ",
    "http://",
    "https://",
    "set-content",
    "add-content",
    "out-file",
    "remove-item",
    "copy-item",
    "move-item",
    "rename-item",
    "new-item",
    "clear-content",
    "set-item",
    "invoke-expression",
    "start-process",
    "stop-process",
    "git ",
    "python ",
    "py ",
    "node ",
    "cmd.exe",
    "bash ",
    "wsl ",
    "certutil",
    "train_0629",
    "data/simulation",
    "data\\simulation",
)
SHELL_SEPARATORS = (";", "|", "&", ">", "<", "\r", "\n")
EXPRESSION_MARKERS = ("(", ")", "{", "}", "::")


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


def unwrap_powershell(command: str) -> str:
    stripped = command.strip()
    executable = (
        r"(?:powershell(?:\.exe)?|"
        r"[^\s'\"&;|<>]*[\\/]+powershell(?:\.exe)?|"
        r"\"[^\"\r\n]*[\\/]+powershell(?:\.exe)?\"|"
        r"'[^'\r\n]*[\\/]+powershell(?:\.exe)?')"
    )
    match = re.match(
        rf"(?is)^(?:&\s+)?{executable}\s+"
        r"(?:(?:-NoLogo|-NoProfile|-NonInteractive|-Sta|-Mta)\s+)*"
        r"-Command\s+(.+)$",
        stripped,
    )
    if match:
        stripped = match.group(1).strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        stripped = stripped[1:-1].strip()
    return stripped


def path_is_allowed(raw_path: str, allowed_root: Path, allow_wildcard: bool) -> bool:
    candidate_text = raw_path.strip()
    if not candidate_text or ".." in Path(candidate_text).parts:
        return False
    has_wildcard = any(marker in candidate_text for marker in ("*", "?", "["))
    if has_wildcard and not allow_wildcard:
        return False
    if has_wildcard:
        first = min(
            index
            for marker in ("*", "?", "[")
            if (index := candidate_text.find(marker)) >= 0
        )
        prefix = candidate_text[:first]
        candidate = Path(prefix).parent.resolve()
    else:
        candidate = Path(candidate_text).resolve()
    try:
        return os.path.commonpath((str(candidate), str(allowed_root))) == str(allowed_root)
    except ValueError:
        return False


def evaluate(payload: dict[str, object], allowed_root: Path) -> tuple[bool, str, str]:
    tool_name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    if tool_name != "Bash":
        return False, f"tool {tool_name!r} is disabled; read saved_configs files only", ""
    if not isinstance(tool_input, dict):
        return False, "shell input is not an object", ""
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return False, "shell command is missing", ""
    body = unwrap_powershell(command)
    folded = body.casefold()
    separator = next((value for value in SHELL_SEPARATORS if value in body), None)
    if separator:
        return False, "command chaining, pipelines, redirection, and multiline input are forbidden", command
    if "$" in body or "`" in body:
        return False, "PowerShell variables and substitutions are forbidden", command
    expression_marker = next(
        (value for value in EXPRESSION_MARKERS if value in body),
        None,
    )
    if expression_marker:
        return False, "PowerShell expressions and static method calls are forbidden", command
    marker = next((value for value in FORBIDDEN_MARKERS if value in folded), None)
    if marker:
        return False, f"forbidden marker {marker!r}; use direct read-only files", command
    command_match = re.match(r"^([A-Za-z-]+)\b", body)
    command_name = command_match.group(1).casefold() if command_match else ""
    required_path_option = ALLOWED_COMMANDS.get(command_name)
    if required_path_option is None:
        return False, "only Get-ChildItem, Get-Content, Select-String, and Test-Path are allowed", command
    path_matches = re.findall(
        rf"(?is)-{required_path_option}\s+(['\"])(.*?)\1",
        body,
    )
    if len(path_matches) != 1:
        return False, f"{command_name} requires exactly one quoted -{required_path_option} value", command
    raw_path = path_matches[0][1]
    if not path_is_allowed(
        raw_path,
        allowed_root,
        allow_wildcard=(command_name == "select-string"),
    ):
        return False, f"path is outside the allowed saved_configs root: {raw_path}", command
    return True, "allowed direct read-only saved_configs file access", command


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
    raw_root = os.environ.get("IP_DISTILL_ALLOWED_ROOT", "")
    if not raw_root:
        response_deny("runner did not configure an allowed saved_configs root")
        return 0
    allowed_root = Path(raw_root).resolve()
    if not allowed_root.is_dir():
        response_deny("configured saved_configs root does not exist")
        return 0
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as exc:
        response_deny(f"hook input parse failure: {type(exc).__name__}")
        return 0
    allowed, reason, command = evaluate(payload, allowed_root)
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
