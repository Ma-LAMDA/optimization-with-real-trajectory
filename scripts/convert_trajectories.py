#!/usr/bin/env python3
"""Convert evalrouter trajectories into Qwen3.6-27B SFT JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft"

NATIVE_FILENAME = "qwen3_6_27b_native_tool_sft.jsonl"
MS_SWIFT_FILENAME = "qwen3_6_27b_ms_swift_agent_sft.jsonl"
FINAL_ANSWER_FILENAME = "qwen3_6_27b_final_answer_sft.jsonl"
MANIFEST_FILENAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing source trajectories (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_trajectories(input_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(input_dir.rglob("conversation_trajectory.json"))
    if not paths:
        raise FileNotFoundError(f"No conversation_trajectory.json found under {input_dir}")

    trajectories: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise ValueError(f"{path}: expected a JSON object with a messages array")
        trajectories.append((path, data))
    return trajectories


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def infer_value_schema(values: Iterable[Any]) -> dict[str, Any]:
    observed = list(values)
    types = sorted({json_type(value) for value in observed})
    schema: dict[str, Any] = {"type": types[0] if len(types) == 1 else types}

    object_values = [value for value in observed if isinstance(value, dict)]
    if object_values and types == ["object"]:
        keys = sorted({key for value in object_values for key in value})
        schema["properties"] = {
            key: infer_value_schema(value[key] for value in object_values if key in value)
            for key in keys
        }
        required = sorted(set.intersection(*(set(value) for value in object_values)))
        if required:
            schema["required"] = required
        schema["additionalProperties"] = True

    array_values = [item for value in observed if isinstance(value, list) for item in value]
    if types == ["array"]:
        schema["items"] = infer_value_schema(array_values) if array_values else {}

    return schema


def iter_tool_uses(trajectory: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for message in trajectory["messages"]:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                yield part


def infer_tool_schemas(
    trajectories: Iterable[tuple[Path, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    calls_by_name: dict[str, list[dict[str, Any]]] = {}
    for _, trajectory in trajectories:
        for call in iter_tool_uses(trajectory):
            name = call.get("name")
            arguments = call.get("input")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("Every tool_use must contain a string name and object input")
            calls_by_name.setdefault(name, []).append(arguments)

    schemas: dict[str, dict[str, Any]] = {}
    for name, arguments_list in sorted(calls_by_name.items()):
        parameter_schema = infer_value_schema(arguments_list)
        parameter_schema["description"] = (
            f"Arguments inferred from {len(arguments_list)} observed {name} call(s)."
        )
        schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool observed in the source trajectories: {name}.",
                "parameters": parameter_schema,
            },
        }
    return schemas


def text_from_parts(parts: Any, allowed_types: set[str]) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        raise ValueError(f"Expected content array or string, got {type(parts).__name__}")

    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Content array items must be objects")
        part_type = part.get("type")
        if part_type not in allowed_types:
            raise ValueError(f"Unexpected content type {part_type!r}")
        if part_type == "text":
            value = part.get("text")
        elif part_type == "thinking":
            value = part.get("thinking")
        elif part_type == "tool_result":
            value = part.get("content")
        else:
            raise AssertionError(f"Unhandled content type {part_type!r}")
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        chunks.append(value)
    return "\n".join(chunks)


def sample_id(path: Path, trajectory: Mapping[str, Any]) -> str:
    question_no = trajectory.get("question_no")
    if isinstance(question_no, str) and question_no:
        return question_no
    if path.parent.name:
        return path.parent.name
    session_id = trajectory.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    raise ValueError(f"Cannot determine sample id for {path}")


def source_metadata(path: Path, trajectory: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "session_id",
        "question_no",
        "harness",
        "harness_session_id",
        "model",
        "num_turns",
        "stop_reason",
        "is_error",
        "tainted",
        "taint_reason",
    )
    metadata = {key: trajectory.get(key) for key in keys if key in trajectory}
    metadata["source_file"] = path.as_posix()
    return metadata


def convert_native_messages(
    trajectory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    converted: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    call_names: dict[str, str] = {}
    used_tools: set[str] = set()

    def ensure_pending() -> dict[str, Any]:
        nonlocal pending
        if pending is None:
            pending = {"content_parts": [], "reasoning_parts": [], "tool_calls": []}
        return pending

    def flush_pending() -> None:
        nonlocal pending
        if pending is None:
            return
        if not any(pending.values()):
            pending = None
            return
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(pending["content_parts"]),
        }
        if pending["reasoning_parts"]:
            message["reasoning_content"] = "\n".join(pending["reasoning_parts"])
        if pending["tool_calls"]:
            message["tool_calls"] = pending["tool_calls"]
        converted.append(message)
        pending = None

    for source_message in trajectory["messages"]:
        role = source_message.get("role")
        content = source_message.get("content", [])

        if role in {"system", "user"}:
            flush_pending()
            converted.append(
                {
                    "role": role,
                    "content": text_from_parts(content, {"text"}),
                }
            )
            continue

        if role == "assistant":
            current = ensure_pending()
            if not isinstance(content, list):
                raise ValueError("Assistant content must be an array in source trajectories")
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Assistant content items must be objects")
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError("Assistant text part must contain a string")
                    current["content_parts"].append(text)
                elif part_type == "thinking":
                    thinking = part.get("thinking")
                    if not isinstance(thinking, str):
                        raise ValueError("Assistant thinking part must contain a string")
                    current["reasoning_parts"].append(thinking)
                elif part_type == "tool_use":
                    call_id = part.get("id")
                    name = part.get("name")
                    arguments = part.get("input")
                    if not isinstance(call_id, str):
                        raise ValueError("tool_use.id must be a string")
                    if not isinstance(name, str):
                        raise ValueError("tool_use.name must be a string")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool_use.input must be an object")
                    current["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    )
                    call_names[call_id] = name
                    used_tools.add(name)
                else:
                    raise ValueError(f"Unexpected assistant content type {part_type!r}")
            continue

        if role == "tool":
            flush_pending()
            if not isinstance(content, list) or len(content) != 1:
                raise ValueError("Each source tool message must contain one tool_result")
            part = content[0]
            if not isinstance(part, dict) or part.get("type") != "tool_result":
                raise ValueError("Tool content must be a tool_result object")
            call_id = part.get("tool_use_id") or source_message.get("tool_call_id")
            if not isinstance(call_id, str):
                raise ValueError("tool_result.tool_use_id must be a string")
            if call_id not in call_names:
                raise ValueError(f"Tool result references unknown call id {call_id!r}")
            result = part.get("content")
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            converted.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": call_id,
                    "name": call_names[call_id],
                    "is_error": bool(part.get("is_error", source_message.get("is_error", False))),
                }
            )
            continue

        raise ValueError(f"Unexpected source message role {role!r}")

    flush_pending()
    return converted, used_tools


def native_sample(
    path: Path,
    trajectory: Mapping[str, Any],
    tool_schemas: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    messages, used_tools = convert_native_messages(trajectory)
    return {
        "id": sample_id(path, trajectory),
        "messages": messages,
        "tools": [tool_schemas[name] for name in sorted(used_tools)],
        "metadata": source_metadata(path, trajectory),
    }


def assistant_text_with_thinking(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise ValueError("Assistant content and reasoning_content must be strings")
    if reasoning:
        thinking = f"<think>\n{reasoning}\n</think>"
        return f"{thinking}\n\n{content}" if content else thinking
    return content


def ms_swift_sample(native: Mapping[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    for message in native["messages"]:
        role = message["role"]
        if role in {"system", "user"}:
            messages.append({"role": role, "content": message["content"]})
        elif role == "assistant":
            assistant_content = assistant_text_with_thinking(message)
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            for call in message.get("tool_calls", []):
                payload = {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                messages.append(
                    {
                        "role": "tool_call",
                        "content": json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
        elif role == "tool":
            payload = {
                "output": message["content"],
                "is_error": bool(message.get("is_error", False)),
            }
            messages.append(
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        else:
            raise ValueError(f"Unexpected native message role {role!r}")

    return {
        "id": native["id"],
        "messages": messages,
        "tools": json.dumps(native["tools"], ensure_ascii=False, separators=(",", ":")),
        "metadata": native["metadata"],
    }


def final_answer_sample(native: Mapping[str, Any]) -> dict[str, Any]:
    system = next(
        (message for message in native["messages"] if message["role"] == "system"),
        None,
    )
    user = next(
        (message for message in native["messages"] if message["role"] == "user"),
        None,
    )
    final_assistant = next(
        (
            message
            for message in reversed(native["messages"])
            if message["role"] == "assistant" and not message.get("tool_calls")
        ),
        None,
    )
    if system is None or user is None or final_assistant is None:
        raise ValueError(f"{native['id']}: cannot construct final-answer sample")

    return {
        "id": native["id"],
        "messages": [
            {"role": "system", "content": system["content"]},
            {"role": "user", "content": user["content"]},
            {"role": "assistant", "content": assistant_text_with_thinking(final_assistant)},
        ],
        "metadata": native["metadata"],
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def message_stats(messages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    stats = {
        "messages": 0,
        "assistant_messages": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "content_characters": 0,
        "reasoning_characters": 0,
    }
    for message in messages:
        stats["messages"] += 1
        if message.get("role") == "assistant":
            stats["assistant_messages"] += 1
        stats["tool_calls"] += len(message.get("tool_calls", []))
        if message.get("role") in {"tool", "tool_response"}:
            stats["tool_results"] += 1
        content = message.get("content", "")
        if isinstance(content, str):
            stats["content_characters"] += len(content)
        reasoning = message.get("reasoning_content", "")
        if isinstance(reasoning, str):
            stats["reasoning_characters"] += len(reasoning)
    return stats


def build_manifest(
    trajectories: list[tuple[Path, dict[str, Any]]],
    native_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    sources = []
    native_by_id = {row["id"]: row for row in native_rows}
    for path, trajectory in trajectories:
        row = native_by_id[sample_id(path, trajectory)]
        sources.append(
            {
                "id": row["id"],
                "path": path.relative_to(PROJECT_ROOT).as_posix()
                if path.is_relative_to(PROJECT_ROOT)
                else path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "source_messages": len(trajectory["messages"]),
                **message_stats(row["messages"]),
            }
        )

    outputs = []
    for filename in (NATIVE_FILENAME, MS_SWIFT_FILENAME, FINAL_ANSWER_FILENAME):
        path = output_dir / filename
        outputs.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix()
                if path.is_relative_to(PROJECT_ROOT)
                else path.as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "samples": len(native_rows),
            }
        )

    return {
        "schema_version": "qwen36-trajectory-sft.v1",
        "model_target": "Qwen/Qwen3.6-27B",
        "sample_count": len(native_rows),
        "source_count": len(trajectories),
        "sources": sources,
        "outputs": outputs,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(input_dir)
    tool_schemas = infer_tool_schemas(trajectories)
    native_rows = [
        native_sample(path, trajectory, tool_schemas)
        for path, trajectory in trajectories
    ]
    ms_swift_rows = [ms_swift_sample(row) for row in native_rows]
    final_answer_rows = [final_answer_sample(row) for row in native_rows]

    write_jsonl(output_dir / NATIVE_FILENAME, native_rows)
    write_jsonl(output_dir / MS_SWIFT_FILENAME, ms_swift_rows)
    write_jsonl(output_dir / FINAL_ANSWER_FILENAME, final_answer_rows)

    manifest = build_manifest(trajectories, native_rows, output_dir)
    with (output_dir / MANIFEST_FILENAME).open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Converted {len(native_rows)} trajectories into {output_dir}")
    for output in manifest["outputs"]:
        print(
            f"- {output['path']}: {output['samples']} samples, "
            f"{output['bytes']} bytes"
        )


if __name__ == "__main__":
    main()

