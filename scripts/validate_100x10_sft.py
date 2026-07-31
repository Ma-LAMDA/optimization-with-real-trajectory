#!/usr/bin/env python3
"""Validate SFT data generated from independently correct 100x10 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-07-31"
CURATION_NAME = "accepted_trajectory_selection.json"
TRAIN_NAME = "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_NAME = "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_NAME = "manifest.json"
RESULT_RE = re.compile(r"<result>\s*([\s\S]*?)\s*</result>")
FORBIDDEN_ASSISTANT_MARKERS = (
    "tool_call",
    "tool_response",
    "webfetch",
    "restore_tool_result",
    "http://",
    "https://",
    "saved_configs",
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
        default=DEFAULT_DATA_ROOT,
        help="Date-scoped generated data directory.",
    )
    return parser.parse_args()


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def stable_digest(path: Path) -> str:
    return hashlib.sha256(normalized_bytes(path)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def parse_result(text: str) -> list[str] | None:
    matches = RESULT_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return sorted(set(value))


def check_row(
    row: dict[str, Any],
    *,
    expected_split: str,
    curation_path: Path,
    curation_digest: str,
) -> tuple[str, str, int]:
    identifier = row.get("id")
    messages = row.get("messages")
    metadata = row.get("metadata")
    if not isinstance(identifier, str) or not identifier.endswith("_decision"):
        raise ValueError("sample id is malformed")
    if (
        not isinstance(messages, list)
        or len(messages) != 3
        or [message.get("role") for message in messages] != [
            "system",
            "user",
            "assistant",
        ]
        or any(not isinstance(message.get("content"), str) for message in messages)
    ):
        raise ValueError(f"{identifier}: messages are malformed")
    assistant = messages[2]["content"]
    lowered = assistant.lower()
    if any(marker.lower() in lowered for marker in FORBIDDEN_ASSISTANT_MARKERS):
        raise ValueError(f"{identifier}: assistant contains an operation marker")
    if not assistant.startswith("<think>\n") or "\n</think>\n\n<result>" not in assistant:
        raise ValueError(f"{identifier}: assistant format is malformed")
    result = parse_result(assistant)
    if not result:
        raise ValueError(f"{identifier}: result is missing or invalid")
    if not isinstance(metadata, dict):
        raise ValueError(f"{identifier}: metadata is missing")
    if (
        metadata.get("dataset_type") != "reasoning_decision"
        or metadata.get("target_type") != "decision"
        or metadata.get("review_status") != "draft"
        or metadata.get("split") != expected_split
        or not isinstance(metadata.get("source_id"), str)
        or not isinstance(metadata.get("case_id"), int)
        or not isinstance(metadata.get("row_index"), int)
        or not isinstance(metadata.get("repeat_index"), int)
        or not isinstance(metadata.get("attempt_index"), int)
        or metadata.get("reference_answer_match") is not True
        or metadata.get("actual_result_items") != result
        or result not in metadata.get("reference_answer_options", [])
        or metadata.get("evidence_count") != 1
        or not isinstance(metadata.get("source_message_index"), int)
        or not isinstance(metadata.get("evidence_message_indices"), list)
        or len(metadata["evidence_message_indices"]) != 1
        or not isinstance(metadata["evidence_message_indices"][0], int)
        or metadata.get("annotation_file")
        != curation_path.relative_to(ROOT).as_posix()
        or metadata.get("annotation_sha256_lf_normalized") != curation_digest
    ):
        raise ValueError(f"{identifier}: metadata is malformed")

    source_path = ROOT / metadata["source_file"]
    events_path = ROOT / metadata["source_event_file"]
    judgment_path = ROOT / metadata["source_judgment_file"]
    if (
        not source_path.is_file()
        or stable_digest(source_path)
        != metadata["source_sha256_lf_normalized"]
        or not events_path.is_file()
        or stable_digest(events_path)
        != metadata["source_event_sha256_lf_normalized"]
        or not judgment_path.is_file()
        or stable_digest(judgment_path)
        != metadata["source_judgment_sha256_lf_normalized"]
    ):
        raise ValueError(f"{identifier}: source provenance mismatch")
    raw = load_json(source_path)
    if (
        raw.get("id") != metadata["source_id"]
        or raw.get("case_id") != metadata["case_id"]
        or raw.get("row_index") != metadata["row_index"]
        or raw.get("success_slot") != metadata["repeat_index"]
        or raw.get("attempt_index") != metadata["attempt_index"]
        or raw.get("answer_matches_reference") is not True
        or raw.get("actual_result_items") != result
    ):
        raise ValueError(f"{identifier}: normalized raw trajectory mismatch")
    judgment = load_json(judgment_path)
    if (
        judgment.get("judge_status") != "completed"
        or judgment.get("parsed") is not True
        or judgment.get("correct") is not True
    ):
        raise ValueError(f"{identifier}: judgment is not strictly correct")
    return identifier, metadata["source_id"], metadata["case_id"]


def main() -> None:
    options = parse_args()
    data_root = options.data_root.resolve()
    raw_dir = data_root / "raw"
    curation_path = data_root / "curation" / CURATION_NAME
    train_path = data_root / "sft" / TRAIN_NAME
    validation_path = data_root / "sft" / VALIDATION_NAME
    manifest_path = data_root / "sft" / MANIFEST_NAME

    curation = load_json(curation_path)
    manifest = load_json(manifest_path)
    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    curation_digest = stable_digest(curation_path)
    if curation.get("schema_version") != "codex-ip-accepted-trajectory-curation.v2":
        raise ValueError("unexpected curation schema")
    if manifest.get("schema_version") != "qwen36-reasoning-decision-sft.v4":
        raise ValueError("unexpected manifest schema")
    if (
        manifest.get("curation_file")
        != curation_path.relative_to(ROOT).as_posix()
        or manifest.get("curation_sha256_lf_normalized") != curation_digest
    ):
        raise ValueError("manifest curation provenance mismatch")

    source_audit = ROOT / manifest["source_final_audit"]
    accepted_index = ROOT / manifest["source_accepted_index"]
    source_dataset = ROOT / manifest["source_dataset"]
    if (
        not source_audit.is_file()
        or stable_digest(source_audit)
        != manifest["source_final_audit_sha256_lf_normalized"]
        or load_json(source_audit).get("passed") is not True
        or not accepted_index.is_file()
        or stable_digest(accepted_index)
        != manifest["source_accepted_index_sha256_lf_normalized"]
        or not source_dataset.is_file()
        or stable_digest(source_dataset)
        != manifest["source_dataset_sha256_lf_normalized"]
    ):
        raise ValueError("source audit or dataset provenance mismatch")

    identifiers: set[str] = set()
    source_ids: set[str] = set()
    train_cases: set[int] = set()
    validation_cases: set[int] = set()
    for split, rows, cases in (
        ("train", train_rows, train_cases),
        ("validation", validation_rows, validation_cases),
    ):
        for row in rows:
            identifier, source_id, case_id = check_row(
                row,
                expected_split=split,
                curation_path=curation_path,
                curation_digest=curation_digest,
            )
            if identifier in identifiers:
                raise ValueError(f"duplicate sample id {identifier}")
            if source_id in source_ids:
                raise ValueError(f"source trajectory appears twice: {source_id}")
            identifiers.add(identifier)
            source_ids.add(source_id)
            cases.add(case_id)
    if train_cases & validation_cases:
        raise ValueError("train and validation case groups overlap")

    split = manifest.get("split")
    if (
        not isinstance(split, dict)
        or split.get("strategy") != "leave_one_case_out"
        or split.get("group_key") != "case_id"
        or split.get("train") != len(train_rows)
        or split.get("validation") != len(validation_rows)
        or split.get("train_case_ids") != sorted(train_cases)
        or split.get("validation_case_ids") != sorted(validation_cases)
        or split.get("case_groups_disjoint") is not True
        or len(validation_cases) != 1
    ):
        raise ValueError("manifest split metadata mismatch")

    trajectories = curation.get("trajectories")
    counts = curation.get("counts")
    if not isinstance(trajectories, list) or not isinstance(counts, dict):
        raise ValueError("curation inventory is malformed")
    selected = [
        item
        for item in trajectories
        if isinstance(item, dict) and item.get("selected") is True
    ]
    selected_ids = {item["id"] for item in selected}
    exclusion_counts = Counter(
        reason
        for item in trajectories
        if isinstance(item, dict)
        for reason in item.get("exclusion_reasons", [])
    )
    if (
        counts.get("accepted_candidates") != len(trajectories)
        or counts.get("selected") != len(selected)
        or counts.get("train") != len(train_rows)
        or counts.get("validation") != len(validation_rows)
        or counts.get("excluded_candidates") != len(trajectories) - len(selected)
        or selected_ids != source_ids
        or curation.get("candidate_exclusion_reason_counts")
        != dict(sorted(exclusion_counts.items()))
    ):
        raise ValueError("curation counts do not match outputs")

    raw_files = sorted(raw_dir.rglob("conversation_trajectory.json"))
    if len(raw_files) != len(trajectories):
        raise ValueError("raw normalized trajectory count mismatch")
    if (
        manifest.get("accepted_candidate_count") != len(trajectories)
        or manifest.get("selected_trajectory_count") != len(source_ids)
        or manifest.get("excluded_candidate_count")
        != len(trajectories) - len(selected)
    ):
        raise ValueError("manifest selection counts mismatch")

    expected_outputs = {
        train_path.relative_to(ROOT).as_posix(): train_rows,
        validation_path.relative_to(ROOT).as_posix(): validation_rows,
    }
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError("manifest outputs are missing")
    for output in outputs:
        if not isinstance(output, dict) or output.get("path") not in expected_outputs:
            raise ValueError("unexpected output metadata")
        path = ROOT / output["path"]
        content = normalized_bytes(path)
        if (
            output.get("samples") != len(expected_outputs[output["path"]])
            or output.get("normalized_bytes") != len(content)
            or output.get("sha256_lf_normalized")
            != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"output metadata mismatch: {path}")

    print(f"Validation passed for {len(source_ids)} fully correct trajectories")
    print(f"- source attempts: {manifest['source_attempt_count']}")
    print(f"- accepted candidates: {manifest['accepted_candidate_count']}")
    print(f"- train: {len(train_rows)} across {len(train_cases)} cases")
    print(
        f"- validation: {len(validation_rows)} from case "
        f"{next(iter(validation_cases))}"
    )
    print(
        f"- filtered non-accepted attempts: "
        f"{manifest['filtered_nonaccepted_attempt_count']}"
    )
    print("- train/validation case overlap: 0")


if __name__ == "__main__":
    main()
