#!/usr/bin/env python3
"""Validate the multi-stage reasoning, planning, and decision SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data" / "2026-07-27" / "sft"
OUTPUT = "qwen3_6_27b_reasoning_decision_sft.jsonl"
MANIFEST = "manifest.json"
TARGET_TYPES = {"planning", "reasoning", "decision"}
OBSOLETE = (
    "qwen3_6_27b_native_tool_sft.jsonl",
    "qwen3_6_27b_ms_swift_agent_sft.jsonl",
    "qwen3_6_27b_final_answer_sft.jsonl",
)
FORBIDDEN_PROTOCOL = (
    '"tools"',
    '"tool_calls"',
    '"tool_call_id"',
    '"role":"tool"',
    '"role":"tool_call"',
    '"role":"tool_response"',
    "<!-- GNS3_API_DOC_START -->",
    "## GNS3 环境接口",
    "TodoWrite",
    "WebFetch",
    "restore_tool_result",
)
FORBIDDEN_OUTPUT_OPERATIONS = (
    "grep",
    "bash",
    "skill",
    "curl",
    "urllib",
    "http://",
    "https://",
    "调用工具",
    "调用接口",
    "执行命令",
    "读取文件",
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def check_result(answer: str, identifier: str) -> int:
    lines = answer.splitlines()
    if (
        len(lines) < 5
        or lines[:2] != ["<result>", "["]
        or lines[-2:] != ["]", "</result>"]
    ):
        raise ValueError(f"{identifier}: malformed result wrapper")
    items = lines[2:-2]
    for index, line in enumerate(items):
        suffix = '",' if index < len(items) - 1 else '"'
        if (
            not line.startswith('"')
            or not line.endswith(suffix)
            or line.count(";") != 1
            or line.strip() != line
        ):
            raise ValueError(f"{identifier}: malformed result item {line!r}")
    return len(items)


def check_row(row: dict[str, Any]) -> tuple[str, str, str]:
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Every row needs a non-empty id")
    if set(row) != {"id", "messages", "metadata"}:
        raise ValueError(f"{identifier}: unexpected top-level fields")
    messages = row["messages"]
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError(f"{identifier}: roles must be system/user/assistant")
    if any(
        not isinstance(item, dict)
        or set(item) != {"role", "content"}
        or not isinstance(item["content"], str)
        or not item["content"]
        for item in messages
    ):
        raise ValueError(f"{identifier}: malformed message")

    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    for marker in FORBIDDEN_PROTOCOL:
        if marker in serialized:
            raise ValueError(f"{identifier}: forbidden protocol marker {marker!r}")
    user = messages[1]["content"]
    if "## 当前任务阶段" not in user or "## 当前已知证据" not in user:
        raise ValueError(f"{identifier}: missing stage or evidence context")

    assistant = messages[2]["content"]
    separator = "\n</think>\n\n"
    if not assistant.startswith("<think>\n") or separator not in assistant:
        raise ValueError(f"{identifier}: malformed reasoning block")
    reasoning, response = assistant[len("<think>\n") :].split(separator, maxsplit=1)
    if not reasoning.strip() or not response.strip():
        raise ValueError(f"{identifier}: empty reasoning or response")
    output = f"{reasoning}\n{response}".lower()
    for marker in FORBIDDEN_OUTPUT_OPERATIONS:
        if marker.lower() in output:
            raise ValueError(f"{identifier}: operation marker remains: {marker!r}")

    metadata = row["metadata"]
    target_type = metadata.get("target_type") if isinstance(metadata, dict) else None
    source = metadata.get("source_id") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset_type") != "reasoning_decision"
        or target_type not in TARGET_TYPES
        or not isinstance(source, str)
        or metadata.get("review_status") not in {"draft", "reviewed"}
        or not isinstance(metadata.get("source_file"), str)
        or not isinstance(metadata.get("source_sha256"), str)
        or len(metadata["source_sha256"]) != 64
        or not isinstance(metadata.get("annotation_file"), str)
        or not isinstance(metadata.get("source_message_index"), int)
        or not isinstance(metadata.get("evidence_message_indices"), list)
        or not isinstance(metadata.get("evidence_count"), int)
        or metadata["evidence_count"] < 0
    ):
        raise ValueError(f"{identifier}: malformed metadata")

    if target_type == "planning":
        if not response.startswith("下一步：") or "<result>" in response:
            raise ValueError(f"{identifier}: planning response is malformed")
    elif target_type == "reasoning":
        if not response.startswith("阶段判断：") or "<result>" in response:
            raise ValueError(f"{identifier}: reasoning response is malformed")
    else:
        check_result(response, identifier)
    return identifier, target_type, source


def check_manifest(
    folder: Path,
    rows: list[dict[str, Any]],
    output: Path,
    type_counts: Counter[str],
) -> None:
    with (folder / MANIFEST).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "qwen36-reasoning-decision-sft.v2":
        raise ValueError("Unexpected manifest schema")
    if manifest.get("sample_count") != len(rows):
        raise ValueError("Manifest sample count does not match the dataset")
    split = manifest.get("split")
    if (
        not isinstance(split, dict)
        or split.get("train") != len(rows)
        or split.get("validation") != 0
    ):
        raise ValueError("Manifest must declare train-only data")
    expected_counts = {
        target_type: type_counts.get(target_type, 0)
        for target_type in sorted(TARGET_TYPES)
    }
    if manifest.get("target_type_counts") != expected_counts:
        raise ValueError("Manifest target type counts do not match")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Manifest must declare one output")
    declared = outputs[0]
    if (
        declared.get("samples") != len(rows)
        or declared.get("bytes") != output.stat().st_size
        or declared.get("sha256") != digest(output)
    ):
        raise ValueError("Manifest output metadata mismatch")
    annotation_path = ROOT / manifest.get("annotation_file", "")
    if not annotation_path.is_file():
        raise ValueError(f"Manifest annotation file does not exist: {annotation_path}")
    if manifest.get("annotation_sha256") != digest(annotation_path):
        raise ValueError("Manifest annotation hash mismatch")


def main() -> None:
    folder = args().sft_dir.resolve()
    for filename in OBSOLETE:
        if (folder / filename).exists():
            raise ValueError(f"Obsolete tool-training output remains: {filename}")
    output = folder / OUTPUT
    rows = load(output)
    identifiers: set[str] = set()
    type_counts: Counter[str] = Counter()
    sources: set[str] = set()
    for row in rows:
        identifier, target_type, source = check_row(row)
        if identifier in identifiers:
            raise ValueError(f"Duplicate id {identifier!r}")
        identifiers.add(identifier)
        type_counts[target_type] += 1
        sources.add(source)
    check_manifest(folder, rows, output, type_counts)
    print(f"Validation passed for {len(rows)} train-only samples")
    print(f"- source trajectories: {len(sources)}")
    for target_type in sorted(TARGET_TYPES):
        print(f"- {target_type}: {type_counts.get(target_type, 0)}")
    print("- validation: 0")
    print("- tool protocol and concrete operation targets: 0")


if __name__ == "__main__":
    main()
