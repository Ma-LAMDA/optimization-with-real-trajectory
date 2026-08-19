#!/usr/bin/env python3
"""Validate the 0804 one-best-trajectory weighted multi-turn SFT dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-04"
SOURCE_CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
BEST_SELECTION = DATA_ROOT / "curation" / "best_trajectory_per_case.json"
MANIFEST = DATA_ROOT / "sft" / "reasoning_trajectory_best1_manifest.json"
TRAIN = DATA_ROOT / "sft" / "qwen3_6_27b_reasoning_trajectory_best1_train.jsonl"
VALIDATION = DATA_ROOT / "sft" / "qwen3_6_27b_reasoning_trajectory_best1_validation.jsonl"

THINKING_LOSS_SCALE = 0.4
TARGET_LOSS_SCALE = 1.0
HISTORY_LOSS_SCALE = 0.0


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected object rows")
    return rows


def digest_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def source_commands(path: Path) -> set[str]:
    commands: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        item = event.get("item") if event.get("type") == "item.completed" else None
        if isinstance(item, dict) and item.get("type") == "command_execution":
            commands.add(normalize_text(str(item.get("command", ""))))
    return commands


def validate_message_loss(row: dict[str, Any]) -> None:
    messages = row["messages"]
    if [messages[0]["role"], messages[1]["role"]] != ["system", "user"]:
        raise ValueError(f"{row['id']}: invalid initial roles")
    if any("loss" in message for message in messages):
        raise ValueError(f"{row['id']}: binary loss field is forbidden")
    if any(
        "loss_scale" in message
        for message in messages
        if message["role"] in {"system", "user", "tool_response"}
    ):
        raise ValueError(f"{row['id']}: context-only role has loss_scale")

    assistants = [message for message in messages if message["role"] == "assistant"]
    target_assistants = [
        message for message in assistants if message.get("loss_scale") in {THINKING_LOSS_SCALE, TARGET_LOSS_SCALE}
    ]
    if len(target_assistants) != 2:
        raise ValueError(f"{row['id']}: expected two target assistant segments")
    if target_assistants[0].get("loss_scale") != THINKING_LOSS_SCALE:
        raise ValueError(f"{row['id']}: target thinking weight mismatch")
    if target_assistants[1].get("loss_scale") != TARGET_LOSS_SCALE:
        raise ValueError(f"{row['id']}: target conclusion weight mismatch")
    if not target_assistants[0]["content"].startswith("<think>\n"):
        raise ValueError(f"{row['id']}: target thinking marker missing")
    if any(
        message.get("loss_scale") != HISTORY_LOSS_SCALE
        for message in assistants
        if message not in target_assistants
    ):
        raise ValueError(f"{row['id']}: historical assistant is not masked")

    target_calls = [
        message
        for message in messages
        if message["role"] == "tool_call" and message.get("loss_scale") == TARGET_LOSS_SCALE
    ]
    historical_calls = [
        message
        for message in messages
        if message["role"] == "tool_call" and message.get("loss_scale") == HISTORY_LOSS_SCALE
    ]
    if len(target_calls) != row["metadata"]["current_action_count"]:
        raise ValueError(f"{row['id']}: target tool-call count mismatch")
    if any(
        message.get("loss_scale") not in {HISTORY_LOSS_SCALE, TARGET_LOSS_SCALE}
        for message in messages
        if message["role"] == "tool_call"
    ):
        raise ValueError(f"{row['id']}: unsupported tool-call weight")
    if len(historical_calls) != sum(
        1
        for message in messages
        if message["role"] == "tool_response"
    ):
        raise ValueError(f"{row['id']}: historical calls/results are not paired")
    if messages[-1]["role"] == "tool_response":
        raise ValueError(f"{row['id']}: future tool result leaked into target")


def main() -> None:
    source = load_json(SOURCE_CURATION)
    selection = load_json(BEST_SELECTION)
    manifest = load_json(MANIFEST)
    train_rows = load_jsonl(TRAIN)
    validation_rows = load_jsonl(VALIDATION)

    if selection["source_curation_sha256_lf_normalized"] != digest_file(SOURCE_CURATION):
        raise ValueError("best-selection source hash mismatch")
    if manifest["source_curation_sha256_lf_normalized"] != digest_file(SOURCE_CURATION):
        raise ValueError("manifest source hash mismatch")
    if manifest["best_selection_sha256_lf_normalized"] != digest_file(BEST_SELECTION):
        raise ValueError("manifest best-selection hash mismatch")
    for split, path in (("train", TRAIN), ("validation", VALIDATION)):
        expected = manifest["outputs"][split]
        if expected["sha256_lf_normalized"] != digest_file(path):
            raise ValueError(f"{split}: output hash mismatch")

    source_annotations = {item["id"]: item for item in source["trajectories"]}
    selected_cases = selection["cases"]
    if len(selected_cases) != 84 or len({item["case_id"] for item in selected_cases}) != 84:
        raise ValueError("selection must contain exactly one trajectory for each of 84 cases")
    selected_ids = {item["selected_trajectory_id"] for item in selected_cases}
    if len(selected_ids) != 84:
        raise ValueError("selected trajectory IDs are not unique")

    selected_source: dict[str, dict[str, Any]] = {}
    commands_by_trajectory: dict[str, set[str]] = {}
    label_by_case: dict[int, str] = {}
    for trajectory_id in selected_ids:
        annotation = source_annotations[trajectory_id]
        raw_path = ROOT / annotation["raw_file"]
        events_path = ROOT / annotation["events_file"]
        raw = load_json(raw_path)
        if not annotation.get("selected") or not raw.get("answer_matches_reference"):
            raise ValueError(f"{trajectory_id}: selected source is not admitted and correct")
        if not raw.get("independent_judgment", {}).get("correct"):
            raise ValueError(f"{trajectory_id}: independent judgment is not correct")
        if digest_file(raw_path) != annotation["raw_sha256_lf_normalized"]:
            raise ValueError(f"{trajectory_id}: raw hash mismatch")
        if digest_file(events_path) != annotation["events_sha256_lf_normalized"]:
            raise ValueError(f"{trajectory_id}: event hash mismatch")
        selected_source[trajectory_id] = raw
        commands_by_trajectory[trajectory_id] = source_commands(events_path)
        labels = {
            str(item).split(";", 1)[1].strip()
            for item in raw["actual_result_items"]
        }
        if len(labels) != 1:
            raise ValueError(f"{trajectory_id}: expected one merged fault label")
        label_by_case[int(raw["case_id"])] = next(iter(labels))

    all_rows = train_rows + validation_rows
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate SFT row ID")
    if len(train_rows) != manifest["counts"]["train_sft_rows"]:
        raise ValueError("train row count mismatch")
    if len(validation_rows) != manifest["counts"]["validation_sft_rows"]:
        raise ValueError("validation row count mismatch")

    rows_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    trajectory_by_case: dict[int, set[str]] = defaultdict(set)
    split_by_case: dict[int, set[str]] = defaultdict(set)
    target_counts = Counter()
    char_lengths: list[int] = []
    for row in all_rows:
        metadata = row["metadata"]
        case_id = int(metadata["case_id"])
        trajectory_id = str(metadata["trajectory_id"])
        rows_by_case[case_id].append(row)
        trajectory_by_case[case_id].add(trajectory_id)
        split_by_case[case_id].add(metadata["split"])
        target_counts[metadata["target_type"]] += 1
        validate_message_loss(row)
        current_action_count = int(metadata["current_action_count"])
        if metadata["target_type"] == "decision_ready":
            if metadata["final_answer_visible"] or current_action_count != 0:
                raise ValueError(f"{row['id']}: invalid decision_ready checkpoint")
            if not metadata.get("evidence_converged_without_next_tool_call"):
                raise ValueError(f"{row['id']}: missing evidence-convergence marker")
        elif not metadata["final_answer_visible"] and current_action_count == 0:
            raise ValueError(f"{row['id']}: zero-action checkpoint is not decision_ready")
        elif metadata.get("evidence_converged_without_next_tool_call"):
            raise ValueError(f"{row['id']}: unexpected evidence-convergence marker")
        if trajectory_id not in selected_ids:
            raise ValueError(f"{row['id']}: row uses an unselected trajectory")
        if metadata["source_event_sha256_lf_normalized"] != source_annotations[trajectory_id]["events_sha256_lf_normalized"]:
            raise ValueError(f"{row['id']}: source event hash mismatch")
        for message in row["messages"]:
            if message["role"] != "tool_call":
                continue
            payload = json.loads(message["content"])
            if payload.get("name") != "powershell":
                raise ValueError(f"{row['id']}: unsupported tool")
            command = normalize_text(str(payload.get("arguments", {}).get("command", "")))
            if command not in commands_by_trajectory[trajectory_id]:
                raise ValueError(f"{row['id']}: tool call is not from source events")
        if metadata["final_answer_visible"]:
            if metadata["target_type"] != "decision":
                raise ValueError(f"{row['id']}: final answer on non-decision row")
            target = [m for m in row["messages"] if m["role"] == "assistant" and m.get("loss_scale") == 1.0][-1]
            if normalize_text(target["content"]) != normalize_text(selected_source[trajectory_id]["final_answer"]):
                raise ValueError(f"{row['id']}: final target mismatch")
        elif metadata["target_type"] == "decision":
            raise ValueError(f"{row['id']}: decision row hides final answer")
        char_lengths.append(sum(len(m["content"]) for m in row["messages"]) + len(row.get("tools", "")))

    if any(len(value) != 1 for value in trajectory_by_case.values()):
        raise ValueError("a case uses more than one selected trajectory")
    if any(len(value) != 1 for value in split_by_case.values()):
        raise ValueError("a case appears in multiple splits")
    for case_id, rows in rows_by_case.items():
        rows.sort(key=lambda row: row["metadata"]["step_index"])
        expected_count = rows[0]["metadata"]["step_count"]
        if len(rows) != expected_count:
            raise ValueError(f"q{case_id:04d}: incomplete step set")
        if [row["metadata"]["step_index"] for row in rows] != list(range(1, expected_count + 1)):
            raise ValueError(f"q{case_id:04d}: non-contiguous steps")
        if sum(row["metadata"]["target_type"] == "decision" for row in rows) != 1:
            raise ValueError(f"q{case_id:04d}: expected exactly one decision step")

    train_cases = {int(row["metadata"]["case_id"]) for row in train_rows}
    validation_cases = {int(row["metadata"]["case_id"]) for row in validation_rows}
    expected_train = set(source["split"]["train_case_ids"])
    expected_validation = set(source["split"]["validation_case_ids"])
    if train_cases != expected_train or validation_cases != expected_validation:
        raise ValueError("case split differs from frozen 0804 split")
    if train_cases & validation_cases:
        raise ValueError("train/validation case leakage")
    if len(train_cases) != 72 or len(validation_cases) != 12:
        raise ValueError("unexpected case counts")
    validation_label_counts = Counter(label_by_case[case_id] for case_id in validation_cases)
    if len(validation_label_counts) != 6 or set(validation_label_counts.values()) != {2}:
        raise ValueError(f"validation is not two complete cases per label: {validation_label_counts}")
    train_label_counts = Counter(label_by_case[case_id] for case_id in train_cases)

    print("0804 best1 reasoning SFT validation passed")
    print(f"- trajectories: train=72, validation=12")
    print(f"- rows: train={len(train_rows)}, validation={len(validation_rows)}")
    print(f"- target types: {dict(sorted(target_counts.items()))}")
    print(f"- train cases by label: {dict(sorted(train_label_counts.items()))}")
    print(f"- validation cases by label: {dict(sorted(validation_label_counts.items()))}")
    print(f"- character length: min={min(char_lengths)}, max={max(char_lengths)}")
    print("- source hashes, exact commands, loss weights, temporal ordering, and case isolation: OK")
    print("- tokenizer length preflight: REQUIRED on the training host")


if __name__ == "__main__":
    main()
