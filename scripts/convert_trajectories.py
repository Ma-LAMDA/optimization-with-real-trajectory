#!/usr/bin/env python3
"""Build multi-stage reasoning, planning, and decision SFT samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ANNOTATIONS = ROOT / "data" / "curation" / "reasoning_decision_annotations.json"
SFT_DIR = ROOT / "data" / "sft"
OUTPUT = "qwen3_6_27b_reasoning_decision_sft.jsonl"
MANIFEST = "manifest.json"
OBSOLETE = (
    "qwen3_6_27b_native_tool_sft.jsonl",
    "qwen3_6_27b_ms_swift_agent_sft.jsonl",
    "qwen3_6_27b_final_answer_sft.jsonl",
)
TARGET_TYPES = {"planning", "reasoning", "decision"}
SYSTEM = (
    "你是一名网络故障分析专家。请根据题目和当前已知证据逐步分析。"
    "信息不足时，说明下一步需要核验的事实以及核验目的；证据充分时，比较候选根因并作出决策。"
    "不得补充题目未提供的事实。先在 <think>...</think> 中给出简洁、可复核的思考，"
    "再输出当前计划、阶段判断或最终结论。"
)
API_DOC = re.compile(r"\n*<!--\s*GNS3_API_DOC_START\s*-->.*\Z", re.DOTALL)
RESULT_ITEMS = re.compile(r'"([^"\r\n]+;[^"\r\n]+)"')
FORBIDDEN_OPERATION_MARKERS = (
    "tool_call",
    "tool_response",
    "todowrite",
    "webfetch",
    "restore_tool_result",
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
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--annotation-file", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=SFT_DIR)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def text_parts(message: Mapping[str, Any], allowed: set[str]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Message content must be a string or list")
    values = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in allowed:
            continue
        value = part.get("text") if part.get("type") == "text" else part.get("thinking")
        if isinstance(value, str):
            values.append(value)
    return "\n".join(values)


def source_id(path: Path, trajectory: Mapping[str, Any]) -> str:
    value = trajectory.get("question_no")
    return value if isinstance(value, str) and value else path.parent.name


def load_sources(folder: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(folder.rglob("conversation_trajectory.json")):
        with path.open("r", encoding="utf-8") as handle:
            trajectory = json.load(handle)
        identifier = source_id(path, trajectory)
        if identifier in sources or not isinstance(trajectory.get("messages"), list):
            raise ValueError(f"Invalid or duplicate trajectory {identifier!r}")
        sources[identifier] = (path, trajectory)
    if not sources:
        raise ValueError(f"No trajectories found under {folder}")
    return sources


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    samples = document.get("samples") if isinstance(document, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("Annotation file must contain a non-empty samples list")
    identifiers: set[str] = set()
    for sample in samples:
        identifier = sample.get("id") if isinstance(sample, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"Invalid or duplicate annotation id {identifier!r}")
        identifiers.add(identifier)
    return samples


def require_strings(value: Any, name: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a string list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return [item.strip() for item in value]


def validate_annotation(
    annotation: Mapping[str, Any],
    sources: Mapping[str, tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any], list[str], str, str, list[int], int, str]:
    identifier = annotation["id"]
    source = annotation.get("source_id")
    if source not in sources:
        raise ValueError(f"{identifier}: unknown source_id {source!r}")
    path, trajectory = sources[source]
    messages = trajectory["messages"]
    target_type = annotation.get("target_type")
    if target_type not in TARGET_TYPES:
        raise ValueError(f"{identifier}: invalid target_type {target_type!r}")

    source_index = annotation.get("source_message_index")
    if (
        not isinstance(source_index, int)
        or isinstance(source_index, bool)
        or not 0 <= source_index < len(messages)
        or messages[source_index].get("role") != "assistant"
        or not text_parts(messages[source_index], {"text", "thinking"}).strip()
    ):
        raise ValueError(f"{identifier}: invalid source_message_index")

    evidence = require_strings(
        annotation.get("evidence"), f"{identifier}.evidence", allow_empty=True
    )
    indices = annotation.get("evidence_message_indices")
    if not isinstance(indices, list) or any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < source_index
        or messages[index].get("role") != "tool"
        for index in indices
    ):
        raise ValueError(f"{identifier}: invalid evidence provenance")

    reasoning = annotation.get("reasoning")
    response = annotation.get("response")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError(f"{identifier}: reasoning must be non-empty")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{identifier}: response must be non-empty")
    reasoning = reasoning.strip()
    response = response.strip()
    curated_text = "\n".join([*evidence, reasoning, response]).lower()
    for marker in FORBIDDEN_OPERATION_MARKERS:
        if marker.lower() in curated_text:
            raise ValueError(f"{identifier}: contains operation marker {marker!r}")

    if target_type == "decision":
        source_items = RESULT_ITEMS.findall(
            text_parts(messages[source_index], {"text"})
        )
        if not source_items or RESULT_ITEMS.findall(response) != source_items:
            raise ValueError(f"{identifier}: decision differs from source answer")

    review_status = annotation.get("review_status")
    if review_status not in {"draft", "reviewed"}:
        raise ValueError(f"{identifier}: review_status must be draft or reviewed")
    return (
        path,
        trajectory,
        evidence,
        reasoning,
        response,
        indices,
        source_index,
        review_status,
    )


def user_prompt(trajectory: Mapping[str, Any]) -> str:
    for message in trajectory["messages"]:
        if message.get("role") == "user":
            value = API_DOC.sub("", text_parts(message, {"text"})).strip()
            if value:
                return value
    raise ValueError("Trajectory has no user prompt")


def build_sample(
    annotation: Mapping[str, Any],
    sources: Mapping[str, tuple[Path, dict[str, Any]]],
    annotation_path: Path,
) -> dict[str, Any]:
    (
        path,
        trajectory,
        evidence,
        reasoning,
        response,
        evidence_indices,
        source_index,
        review_status,
    ) = validate_annotation(annotation, sources)
    evidence_text = (
        "\n".join(
            f"{index}. {item}" for index, item in enumerate(evidence, start=1)
        )
        if evidence
        else "除题目中给出的信息外，当前尚无新增证据。"
    )
    stage_instruction = {
        "planning": "当前证据不足以形成最终结论。请说明下一步需要核验的事实及其目的，不要提前输出最终根因。",
        "reasoning": "请根据当前证据给出阶段判断，并说明尚需核验的关键事实。",
        "decision": "当前证据已完成收敛。请给出最小根因集合，并严格遵守题目要求的答案格式。",
    }[annotation["target_type"]]
    user = (
        f"{user_prompt(trajectory)}\n\n"
        f"## 当前任务阶段\n\n{stage_instruction}\n\n"
        f"## 当前已知证据\n\n{evidence_text}"
    )
    return {
        "id": annotation["id"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": f"<think>\n{reasoning}\n</think>\n\n{response}",
            },
        ],
        "metadata": {
            "dataset_type": "reasoning_decision",
            "target_type": annotation["target_type"],
            "review_status": review_status,
            "source_id": annotation["source_id"],
            "source_file": label(path),
            "source_sha256": digest(path),
            "annotation_file": label(annotation_path),
            "source_message_index": source_index,
            "evidence_message_indices": evidence_indices,
            "evidence_count": len(evidence),
        },
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    options = args()
    input_dir = options.input_dir.resolve()
    annotation_path = options.annotation_file.resolve()
    output_dir = options.output_dir.resolve()
    sources = load_sources(input_dir)
    annotations = load_annotations(annotation_path)
    annotated_sources = {annotation.get("source_id") for annotation in annotations}
    if annotated_sources != set(sources):
        raise ValueError(
            "Every trajectory needs at least one annotation; "
            f"missing={sorted(set(sources) - annotated_sources)}, "
            f"extra={sorted(annotated_sources - set(sources))}"
        )

    rows = [
        build_sample(annotation, sources, annotation_path)
        for annotation in annotations
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT
    write_jsonl(output_path, rows)
    for filename in OBSOLETE:
        obsolete = output_dir / filename
        if obsolete.exists():
            obsolete.unlink()

    type_counts = Counter(row["metadata"]["target_type"] for row in rows)
    manifest = {
        "schema_version": "qwen36-reasoning-decision-sft.v2",
        "source_count": len(sources),
        "sample_count": len(rows),
        "annotation_file": label(annotation_path),
        "annotation_sha256": digest(annotation_path),
        "target_type_counts": {
            target_type: type_counts.get(target_type, 0)
            for target_type in sorted(TARGET_TYPES)
        },
        "split": {"train": len(rows), "validation": 0, "strategy": "train_only"},
        "outputs": [
            {
                "path": label(output_path),
                "samples": len(rows),
                "bytes": output_path.stat().st_size,
                "sha256": digest(output_path),
            }
        ],
    }
    with (output_dir / MANIFEST).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Generated {len(rows)} train-only samples from {len(sources)} trajectories")
    for target_type in sorted(TARGET_TYPES):
        print(f"- {target_type}: {type_counts.get(target_type, 0)}")
    print("- validation: 0")


if __name__ == "__main__":
    main()
