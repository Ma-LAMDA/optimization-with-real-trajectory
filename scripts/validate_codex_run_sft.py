#!/usr/bin/env python3
"""Validate the date-scoped Codex SFT train/validation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-07-28"
CURATION_FILE_NAME = "trajectory_selection.json"
TRAIN_FILE = "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_FILE = "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_FILE = "manifest.json"
FORBIDDEN_PROTOCOL = (
    '"tools"',
    '"tool_calls"',
    '"tool_call_id"',
    '"role":"tool"',
    '"role":"tool_call"',
    '"role":"tool_response"',
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
    "saved_configs",
    ".txt",
    "powershell",
    "调用工具",
    "调用接口",
    "执行命令",
    "读取文件",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Date-scoped Codex dataset directory. The historical "
            "data/2026-07-28 directory is used when it exists."
        ),
    )
    options = parser.parse_args()
    if options.data_root is None:
        if DEFAULT_DATA_ROOT.is_dir():
            options.data_root = DEFAULT_DATA_ROOT
        else:
            parser.error(
                "the historical default data/2026-07-28 is not present; "
                "recreate it with convert_codex_run_trajectories.py or pass "
                "--data-root PATH"
            )
    return options


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def check_result(value: str, identifier: str) -> list[str]:
    lines = value.splitlines()
    if (
        len(lines) < 5
        or lines[:2] != ["<result>", "["]
        or lines[-2:] != ["]", "</result>"]
    ):
        raise ValueError(f"{identifier}: malformed result wrapper")
    items: list[str] = []
    result_lines = lines[2:-2]
    for index, line in enumerate(result_lines):
        suffix = '",' if index < len(result_lines) - 1 else '"'
        if (
            not line.startswith('"')
            or not line.endswith(suffix)
            or line.count(";") != 1
            or line.strip() != line
        ):
            raise ValueError(f"{identifier}: malformed result item {line!r}")
        items.append(line[1 : -2 if suffix == '",' else -1])
    if not items:
        raise ValueError(f"{identifier}: empty result")
    return items


def canonical_result(items: list[str]) -> str:
    lines = [
        f'"{item}",' if index < len(items) - 1 else f'"{item}"'
        for index, item in enumerate(items)
    ]
    return "\n".join(["<result>", "[", *lines, "]", "</result>"])


def check_row(
    row: dict[str, Any],
    expected_split: str,
    curation_path: Path,
    curation_sha256: str,
) -> tuple[str, int, str]:
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Every sample needs a non-empty id")
    if set(row) != {"id", "messages", "metadata"}:
        raise ValueError(f"{identifier}: unexpected top-level fields")
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [message.get("role") for message in messages]
        != ["system", "user", "assistant"]
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["content"], str)
            or not message["content"]
            for message in messages
        )
    ):
        raise ValueError(f"{identifier}: malformed messages")

    serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    for marker in FORBIDDEN_PROTOCOL:
        if marker in serialized:
            raise ValueError(f"{identifier}: forbidden protocol marker {marker!r}")
    user = messages[1]["content"]
    if (
        "## 当前任务阶段" not in user
        or "## 当前已知证据" not in user
        or "## 离线组网配置查询 Skills" in user
    ):
        raise ValueError(f"{identifier}: malformed user context")

    assistant = messages[2]["content"]
    separator = "\n</think>\n\n"
    if not assistant.startswith("<think>\n") or separator not in assistant:
        raise ValueError(f"{identifier}: malformed reasoning block")
    reasoning, response = assistant[len("<think>\n") :].split(
        separator, maxsplit=1
    )
    if not reasoning.strip() or not response.strip():
        raise ValueError(f"{identifier}: empty reasoning or response")
    output_text = reasoning + "\n" + response
    for marker in FORBIDDEN_OUTPUT_OPERATIONS:
        if marker.lower() in output_text.lower():
            raise ValueError(
                f"{identifier}: operation marker remains: {marker!r}"
            )
    result = check_result(response, identifier)

    metadata = row.get("metadata")
    required_metadata = {
        "dataset_type",
        "target_type",
        "review_status",
        "split",
        "source_id",
        "case_id",
        "repeat_index",
        "source_file",
        "source_sha256",
        "source_event_file",
        "source_event_sha256",
        "annotation_file",
        "annotation_sha256",
        "source_message_index",
        "evidence_message_indices",
        "evidence_count",
        "expected_result_items",
        "reference_answer_match",
        "source_answer_format_normalized",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required_metadata
        or metadata.get("dataset_type") != "reasoning_decision"
        or metadata.get("target_type") != "decision"
        or metadata.get("review_status") != "draft"
        or metadata.get("split") != expected_split
        or not isinstance(metadata.get("source_id"), str)
        or not isinstance(metadata.get("case_id"), int)
        or not isinstance(metadata.get("repeat_index"), int)
        or metadata.get("reference_answer_match") is not True
        or not isinstance(metadata.get("source_answer_format_normalized"), bool)
        or metadata.get("expected_result_items") != result
        or metadata.get("evidence_count") != 1
        or not isinstance(metadata.get("source_message_index"), int)
        or not isinstance(metadata.get("evidence_message_indices"), list)
        or len(metadata["evidence_message_indices"]) != 1
        or not isinstance(metadata["evidence_message_indices"][0], int)
        or metadata.get("annotation_file") != curation_path.relative_to(
            ROOT
        ).as_posix()
        or metadata.get("annotation_sha256") != curation_sha256
    ):
        raise ValueError(f"{identifier}: malformed metadata")

    source_file = ROOT / metadata["source_file"]
    source_event_file = ROOT / metadata["source_event_file"]
    if (
        not source_file.is_file()
        or digest(source_file) != metadata["source_sha256"]
        or not source_event_file.is_file()
        or digest(source_event_file) != metadata["source_event_sha256"]
    ):
        raise ValueError(f"{identifier}: source provenance mismatch")
    raw = load_json(source_file)
    if (
        raw.get("id") != metadata["source_id"]
        or raw.get("case_id") != metadata["case_id"]
        or raw.get("repeat_index") != metadata["repeat_index"]
        or raw.get("answer_matches_reference") is not True
        or raw.get("actual_result_items") != result
        or response != canonical_result(result)
        or metadata.get("source_answer_format_normalized")
        is not (raw.get("final_answer") != response)
    ):
        raise ValueError(f"{identifier}: normalized raw trajectory mismatch")
    return identifier, metadata["case_id"], metadata["source_id"]


def main() -> None:
    options = parse_args()
    data_root = options.data_root.resolve()
    raw_dir = data_root / "raw"
    curation_path = data_root / "curation" / CURATION_FILE_NAME
    sft_dir = data_root / "sft"
    train_path = sft_dir / TRAIN_FILE
    validation_path = sft_dir / VALIDATION_FILE
    manifest_path = sft_dir / MANIFEST_FILE

    curation = load_json(curation_path)
    manifest = load_json(manifest_path)
    if curation.get("schema_version") != "codex-ip-trajectory-curation.v1":
        raise ValueError("Unexpected curation schema")
    if manifest.get("schema_version") != "qwen36-reasoning-decision-sft.v3":
        raise ValueError("Unexpected SFT manifest schema")
    curation_sha256 = digest(curation_path)
    if (
        manifest.get("curation_file")
        != curation_path.relative_to(ROOT).as_posix()
        or manifest.get("curation_sha256") != curation_sha256
    ):
        raise ValueError("Manifest curation provenance mismatch")

    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    identifiers: set[str] = set()
    sources: set[str] = set()
    train_cases: set[int] = set()
    validation_cases: set[int] = set()
    for split, rows, cases in (
        ("train", train_rows, train_cases),
        ("validation", validation_rows, validation_cases),
    ):
        for row in rows:
            identifier, case_id, source_id = check_row(
                row, split, curation_path, curation_sha256
            )
            if identifier in identifiers:
                raise ValueError(f"Duplicate sample id {identifier}")
            if source_id in sources:
                raise ValueError(f"Source trajectory appears twice: {source_id}")
            identifiers.add(identifier)
            sources.add(source_id)
            cases.add(case_id)
    if train_cases & validation_cases:
        raise ValueError("Train and validation case groups overlap")

    split = manifest.get("split")
    selection = manifest.get("selection")
    user_excluded_case_ids = (
        set(selection.get("excluded_case_ids", []))
        if isinstance(selection, dict)
        else set()
    )
    quality_excluded_case_ids = (
        set(selection.get("quality_excluded_case_ids", []))
        if isinstance(selection, dict)
        else set()
    )
    excluded_case_ids = user_excluded_case_ids | quality_excluded_case_ids
    if (
        not isinstance(split, dict)
        or not user_excluded_case_ids
        or selection.get("required_case_eligibility_rate") != 1.0
        or split.get("strategy") != "leave_one_case_out"
        or split.get("group_key") != "case_id"
        or split.get("train") != len(train_rows)
        or split.get("validation") != len(validation_rows)
        or split.get("train_case_ids") != sorted(train_cases)
        or split.get("validation_case_ids") != sorted(validation_cases)
        or split.get("case_groups_disjoint") is not True
        or len(validation_cases) != 1
        or bool((train_cases | validation_cases) & excluded_case_ids)
    ):
        raise ValueError("Manifest split metadata mismatch")

    curation_counts = curation.get("counts")
    trajectories = curation.get("trajectories")
    case_quality = curation.get("case_quality")
    if not isinstance(trajectories, list) or not isinstance(
        curation_counts, dict
    ) or not isinstance(case_quality, list):
        raise ValueError("Malformed curation inventory")
    selected = [
        item
        for item in trajectories
        if isinstance(item, dict) and item.get("selected") is True
    ]
    split_counter = Counter(
        item.get("split") for item in trajectories if isinstance(item, dict)
    )
    selected_ids = {item["id"] for item in selected}
    excluded_case_entries = [
        item
        for item in trajectories
        if isinstance(item, dict)
        and item.get("case_id") in user_excluded_case_ids
    ]
    if (
        curation_counts.get("raw") != len(trajectories)
        or curation_counts.get("selected") != len(selected)
        or curation_counts.get("train") != len(train_rows)
        or curation_counts.get("validation") != len(validation_rows)
        or curation_counts.get("excluded") != split_counter["excluded"]
        or selected_ids != sources
        or not excluded_case_entries
        or any(
            item.get("selected") is not False
            or "case_excluded_by_user"
            not in item.get("exclusion_reasons", [])
            for item in excluded_case_entries
        )
    ):
        raise ValueError("Curation counts or selected ids do not match")

    quality_by_case = {
        item.get("case_id"): item
        for item in case_quality
        if isinstance(item, dict) and isinstance(item.get("case_id"), int)
    }
    source_cases = {
        item.get("case_id")
        for item in trajectories
        if isinstance(item, dict)
    }
    if set(quality_by_case) != source_cases:
        raise ValueError("Curation case quality inventory is incomplete")
    for case_id in source_cases:
        case_rows = [
            item
            for item in trajectories
            if isinstance(item, dict) and item.get("case_id") == case_id
        ]
        exact_answers = sum(
            "final_answer_differs_from_reference"
            not in item.get("exclusion_reasons", [])
            for item in case_rows
        )
        selected_rows = [
            item for item in case_rows if item.get("selected") is True
        ]
        expected_split = (
            selected_rows[0].get("split") if selected_rows else "excluded"
        )
        quality = quality_by_case[case_id]
        if (
            quality.get("trajectories") != len(case_rows)
            or quality.get("exact_reference_answers") != exact_answers
            or quality.get("accuracy")
            != round(exact_answers / len(case_rows), 6)
            or quality.get("selected_trajectories") != len(selected_rows)
            or quality.get("split") != expected_split
            or quality.get("user_excluded")
            is not (case_id in user_excluded_case_ids)
            or quality.get("quality_excluded")
            is not (case_id in quality_excluded_case_ids)
        ):
            raise ValueError(f"Case quality mismatch for case {case_id}")
    expected_case_accuracy = {
        str(case_id): quality_by_case[case_id]["accuracy"]
        for case_id in sorted(source_cases)
    }
    if selection.get("case_accuracy") != expected_case_accuracy:
        raise ValueError("Manifest case accuracy summary mismatch")

    raw_files = sorted(raw_dir.rglob("conversation_trajectory.json"))
    if (
        manifest.get("raw_trajectory_count") != len(raw_files)
        or manifest.get("raw_trajectory_count") != len(trajectories)
        or manifest.get("selected_trajectory_count") != len(sources)
        or manifest.get("excluded_trajectory_count")
        != split_counter["excluded"]
    ):
        raise ValueError("Raw or selected trajectory count mismatch")

    expected_outputs = {
        train_path.relative_to(ROOT).as_posix(): train_rows,
        validation_path.relative_to(ROOT).as_posix(): validation_rows,
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError("Manifest must declare train and validation outputs")
    for output in outputs:
        if not isinstance(output, dict) or output.get("path") not in expected_outputs:
            raise ValueError("Unexpected output entry")
        path = ROOT / output["path"]
        rows = expected_outputs[output["path"]]
        if (
            output.get("samples") != len(rows)
            or output.get("bytes") != path.stat().st_size
            or output.get("sha256") != digest(path)
        ):
            raise ValueError(f"Output metadata mismatch: {path}")

    source_manifest = ROOT / manifest.get("source_manifest", "")
    if (
        not source_manifest.is_file()
        or manifest.get("source_manifest_sha256") != digest(source_manifest)
    ):
        raise ValueError("Experiment source manifest mismatch")

    print(
        f"Validation passed for {len(train_rows) + len(validation_rows)} "
        "selected trajectories"
    )
    print(f"- raw trajectories: {len(raw_files)}")
    print(f"- train: {len(train_rows)} across {len(train_cases)} cases")
    print(
        f"- validation: {len(validation_rows)} from case "
        f"{next(iter(validation_cases))}"
    )
    print(f"- excluded incorrect/unsafe trajectories: {split_counter['excluded']}")
    print(f"- user-excluded cases: {sorted(user_excluded_case_ids)}")
    print(f"- quality-excluded cases: {sorted(quality_excluded_case_ids)}")
    print("- train/validation case overlap: 0")
    print("- tool protocol and concrete operation targets: 0")


if __name__ == "__main__":
    main()
