#!/usr/bin/env python3
"""Validate generated Qwen3.6-27B trajectory SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_DIR = PROJECT_ROOT / "data" / "sft"

NATIVE_FILENAME = "qwen3_6_27b_native_tool_sft.jsonl"
MS_SWIFT_FILENAME = "qwen3_6_27b_ms_swift_agent_sft.jsonl"
FINAL_ANSWER_FILENAME = "qwen3_6_27b_final_answer_sft.jsonl"
MANIFEST_FILENAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=DEFAULT_SFT_DIR,
        help=f"Generated SFT directory (default: {DEFAULT_SFT_DIR})",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def require_unique_ids(rows: list[dict[str, Any]], label: str) -> set[str]:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{label}: every sample must have a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate sample ids")
    return set(ids)


def validate_native(rows: list[dict[str, Any]]) -> dict[str, int]:
    tool_calls = 0
    tool_results = 0
    for row in rows:
        messages = row.get("messages")
        tools = row.get("tools")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{row.get('id')}: messages must be a non-empty list")
        if not isinstance(tools, list):
            raise ValueError(f"{row.get('id')}: tools must be a list")
        if messages[0].get("role") != "system":
            raise ValueError(f"{row.get('id')}: first message must be system")
        if messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
            raise ValueError(f"{row.get('id')}: last message must be a final assistant answer")

        declared_tools: set[str] = set()
        for tool in tools:
            try:
                function = tool["function"]
                name = function["name"]
                parameters = function["parameters"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"{row.get('id')}: malformed tool schema") from exc
            if tool.get("type") != "function" or not isinstance(name, str):
                raise ValueError(f"{row.get('id')}: malformed function tool")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError(f"{row.get('id')}: tool parameters must be an object schema")
            declared_tools.add(name)

        pending: set[str] = set()
        seen_call_ids: set[str] = set()
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"{row.get('id')}[{index}]: invalid role {role!r}")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"{row.get('id')}[{index}]: content must be a string")
            if role == "system" and index != 0:
                raise ValueError(f"{row.get('id')}[{index}]: system must be first")

            calls = message.get("tool_calls", [])
            if calls:
                if role != "assistant" or pending:
                    raise ValueError(f"{row.get('id')}[{index}]: invalid tool call placement")
                for call in calls:
                    try:
                        call_id = call["id"]
                        name = call["function"]["name"]
                        arguments = call["function"]["arguments"]
                    except (KeyError, TypeError) as exc:
                        raise ValueError(
                            f"{row.get('id')}[{index}]: malformed tool call"
                        ) from exc
                    if call.get("type") != "function":
                        raise ValueError(f"{row.get('id')}[{index}]: tool type must be function")
                    if not isinstance(call_id, str) or call_id in seen_call_ids:
                        raise ValueError(f"{row.get('id')}[{index}]: invalid/duplicate call id")
                    if name not in declared_tools or not isinstance(arguments, dict):
                        raise ValueError(
                            f"{row.get('id')}[{index}]: undeclared tool or invalid arguments"
                        )
                    pending.add(call_id)
                    seen_call_ids.add(call_id)
                    tool_calls += 1
                continue

            if role == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    raise ValueError(
                        f"{row.get('id')}[{index}]: tool result does not match pending call"
                    )
                pending.remove(call_id)
                tool_results += 1
            elif pending:
                raise ValueError(
                    f"{row.get('id')}[{index}]: missing tool results before next message"
                )

        if pending:
            raise ValueError(f"{row.get('id')}: unresolved tool calls {sorted(pending)}")

    if tool_calls != tool_results:
        raise ValueError(
            f"native: tool call/result mismatch ({tool_calls} != {tool_results})"
        )
    return {"samples": len(rows), "tool_calls": tool_calls, "tool_results": tool_results}


def validate_ms_swift(rows: list[dict[str, Any]]) -> dict[str, int]:
    tool_calls = 0
    tool_results = 0
    allowed_roles = {"system", "user", "assistant", "tool_call", "tool_response"}
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{row.get('id')}: messages must be a non-empty list")
        try:
            tools = json.loads(row["tools"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{row.get('id')}: tools must be a JSON-list string") from exc
        if not isinstance(tools, list):
            raise ValueError(f"{row.get('id')}: decoded tools must be a list")
        declared_tools = {
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict)
        }

        pending = 0
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in allowed_roles or not isinstance(content, str):
                raise ValueError(f"{row.get('id')}[{index}]: invalid role/content")

            if role == "tool_call":
                if pending < 0:
                    raise AssertionError("pending cannot be negative")
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{row.get('id')}[{index}]: invalid tool_call JSON"
                    ) from exc
                if (
                    not isinstance(payload, dict)
                    or payload.get("name") not in declared_tools
                    or not isinstance(payload.get("arguments"), dict)
                ):
                    raise ValueError(f"{row.get('id')}[{index}]: invalid tool_call payload")
                pending += 1
                tool_calls += 1
            elif role == "tool_response":
                if pending == 0:
                    raise ValueError(f"{row.get('id')}[{index}]: unexpected tool_response")
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{row.get('id')}[{index}]: invalid tool_response JSON"
                    ) from exc
                if (
                    not isinstance(payload, dict)
                    or not isinstance(payload.get("output"), str)
                    or not isinstance(payload.get("is_error"), bool)
                ):
                    raise ValueError(
                        f"{row.get('id')}[{index}]: invalid tool_response payload"
                    )
                pending -= 1
                tool_results += 1
            elif pending:
                raise ValueError(
                    f"{row.get('id')}[{index}]: missing tool responses before next message"
                )

        if pending:
            raise ValueError(f"{row.get('id')}: {pending} unresolved tool calls")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{row.get('id')}: last message must be assistant")

    if tool_calls != tool_results:
        raise ValueError(
            f"ms-swift: tool call/result mismatch ({tool_calls} != {tool_results})"
        )
    return {"samples": len(rows), "tool_calls": tool_calls, "tool_results": tool_results}


def validate_final_answer(rows: list[dict[str, Any]]) -> dict[str, int]:
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{row.get('id')}: messages must be a list")
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                f"{row.get('id')}: final-answer roles must be system/user/assistant"
            )
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError(f"{row.get('id')}: all contents must be strings")
        if not messages[-1]["content"]:
            raise ValueError(f"{row.get('id')}: final answer must not be empty")
    return {"samples": len(rows), "tool_calls": 0, "tool_results": 0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(sft_dir: Path, expected_samples: int) -> None:
    path = sft_dir / MANIFEST_FILENAME
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("sample_count") != expected_samples:
        raise ValueError("manifest sample_count does not match JSONL files")
    for output in manifest.get("outputs", []):
        output_path = PROJECT_ROOT / output["path"]
        if not output_path.is_file():
            raise ValueError(f"manifest output does not exist: {output_path}")
        if output.get("sha256") != sha256_file(output_path):
            raise ValueError(f"manifest hash mismatch: {output_path}")
        if output.get("bytes") != output_path.stat().st_size:
            raise ValueError(f"manifest size mismatch: {output_path}")


def main() -> None:
    args = parse_args()
    sft_dir = args.sft_dir.resolve()

    native_rows = load_jsonl(sft_dir / NATIVE_FILENAME)
    ms_swift_rows = load_jsonl(sft_dir / MS_SWIFT_FILENAME)
    final_answer_rows = load_jsonl(sft_dir / FINAL_ANSWER_FILENAME)

    native_ids = require_unique_ids(native_rows, "native")
    ms_swift_ids = require_unique_ids(ms_swift_rows, "ms-swift")
    final_answer_ids = require_unique_ids(final_answer_rows, "final-answer")
    if not (native_ids == ms_swift_ids == final_answer_ids):
        raise ValueError("Sample id sets differ across generated datasets")

    summaries = {
        NATIVE_FILENAME: validate_native(native_rows),
        MS_SWIFT_FILENAME: validate_ms_swift(ms_swift_rows),
        FINAL_ANSWER_FILENAME: validate_final_answer(final_answer_rows),
    }
    validate_manifest(sft_dir, len(native_rows))

    print(f"Validation passed for {len(native_rows)} samples: {sorted(native_ids)}")
    for filename, summary in summaries.items():
        print(
            f"- {filename}: samples={summary['samples']}, "
            f"tool_calls={summary['tool_calls']}, "
            f"tool_results={summary['tool_results']}"
        )


if __name__ == "__main__":
    main()

