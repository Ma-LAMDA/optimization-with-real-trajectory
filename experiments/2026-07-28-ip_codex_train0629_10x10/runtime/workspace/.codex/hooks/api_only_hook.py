#!/usr/bin/env python3
"""Block every Codex tool call except read-only HTTP queries to localhost:3080."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


ALLOWED_BASE = "http://127.0.0.1:3080"
EXPERIMENT_ROOT = Path(__file__).resolve().parents[4]
AUDIT_PATH = EXPERIMENT_ROOT / "runtime" / "hook_audit" / "pre_tool_use.jsonl"
FILESYSTEM_MARKERS = (
    "saved_configs",
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
    "../",
    "..\\",
)


def deny(reason: str) -> None:
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


def evaluate(payload: dict[str, object]) -> tuple[bool, str, str]:
    tool_name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    if tool_name != "Bash":
        return False, f"Tool {tool_name!r} is disabled; query the local API through the shell.", ""
    if not isinstance(tool_input, dict):
        return False, "Shell input is not a JSON object.", ""
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return False, "Shell command is missing.", ""

    folded = command.casefold()
    if ALLOWED_BASE not in command:
        return False, "Only read-only queries to http://127.0.0.1:3080 are allowed.", command
    marker = next((value for value in FILESYSTEM_MARKERS if value in folded), None)
    if marker:
        return False, f"Filesystem access marker {marker!r} is forbidden; use the local API.", command
    if re.search(r"(?<![<])>(?![=>])|>>", command):
        return False, "Shell redirection is forbidden; keep API results in stdout.", command

    urls = re.findall(r"https?://[^\s'\"]+", command, flags=re.IGNORECASE)
    if not urls:
        return False, "No HTTP URL was found in the command.", command
    for value in urls:
        parsed = urlsplit(value.rstrip(");,"))
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port != 3080:
            return False, f"Non-local URL is forbidden: {value}", command
    return True, "allowed localhost API query", command


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as exc:
        deny(f"Hook could not parse tool input: {exc}")
        return 0
    allowed, reason, command = evaluate(payload)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "checked_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "session_id": payload.get("session_id"),
                    "tool_name": payload.get("tool_name"),
                    "allowed": allowed,
                    "reason": reason,
                    "command": command,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
    if not allowed:
        deny(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
