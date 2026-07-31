#!/usr/bin/env python3
"""Validate SFT data generated from independently correct 100x10 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-07-31"
CURATION_NAME = "accepted_trajectory_selection.json"
TRAIN_NAME = "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_NAME = "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_NAME = "manifest.json"
FILTER_REPORT_NAME = "FILTER_REPORT.md"
ATTEMPT_STATUSES = {
    "accepted",
    "rejected",
    "interrupted",
    "infrastructure_failure",
}
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


def format_duration(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def validate_filter_report(
    *,
    report_path: Path,
    experiment_root: Path,
    curation: dict[str, Any],
) -> None:
    runs_dir = experiment_root / "results" / "runs"
    state = load_json(experiment_root / "results" / "report" / "state.json")
    if state.get("status") != "completed":
        raise ValueError("filter report source state is not completed")

    state_by_case = {
        int(item["original_id"]): item
        for item in state.get("samples", [])
        if isinstance(item, dict)
    }
    status_by_case: defaultdict[int, Counter[str]] = defaultdict(Counter)
    durations_by_case: defaultdict[int, list[float]] = defaultdict(list)
    successful_durations_by_case: defaultdict[int, list[float]] = defaultdict(
        list
    )
    for metadata_path in runs_dir.glob("q*_r*/attempt_*/metadata.json"):
        metadata = load_json(metadata_path)
        run_key = metadata_path.relative_to(runs_dir).parts[0]
        run_match = re.fullmatch(r"q(\d+)_r\d+", run_key)
        if run_match is None:
            raise ValueError(f"cannot determine case id from {metadata_path}")
        case_id = int(run_match.group(1))
        if int(metadata.get("original_id", -1)) != case_id:
            raise ValueError(f"{metadata_path}: case identity mismatch")
        state_item = state_by_case.get(case_id)
        if (
            state_item is None
            or int(metadata.get("row_index", -1))
            != int(state_item.get("row_index", -2))
        ):
            raise ValueError(f"{metadata_path}: row identity mismatch")
        status = str(metadata.get("status", "unknown"))
        if status not in ATTEMPT_STATUSES:
            raise ValueError(f"{metadata_path}: unexpected status {status!r}")
        status_by_case[case_id][status] += 1
        duration = metadata.get("duration_seconds")
        if duration is None:
            continue
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
            or not math.isfinite(duration)
        ):
            raise ValueError(f"{metadata_path}: invalid duration_seconds")
        duration_value = float(duration)
        durations_by_case[case_id].append(duration_value)
        if status == "accepted":
            successful_durations_by_case[case_id].append(duration_value)

    trajectories = curation.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("curation trajectories are missing")
    selected_by_case: Counter[int] = Counter()
    splits_by_case: defaultdict[int, set[str]] = defaultdict(set)
    label_case_ids: defaultdict[str, set[int]] = defaultdict(set)
    label_trajectory_counts: Counter[str] = Counter()
    for item in trajectories:
        if not isinstance(item, dict) or item.get("selected") is not True:
            continue
        case_id = int(item["case_id"])
        selected_by_case[case_id] += 1
        splits_by_case[case_id].add(str(item["split"]))
        answer_labels = item.get("actual_result_items")
        if (
            not isinstance(answer_labels, list)
            or not answer_labels
            or any(not isinstance(label, str) for label in answer_labels)
            or len(set(answer_labels)) != len(answer_labels)
        ):
            raise ValueError("selected trajectory answer labels are malformed")
        for answer_label in answer_labels:
            label_case_ids[answer_label].add(case_id)
            label_trajectory_counts[answer_label] += 1

    report_lines = report_path.read_text(encoding="utf-8-sig").splitlines()
    case_rows: dict[int, list[str]] = {}
    total_cells: list[str] | None = None
    success_distribution_rows: dict[int, list[str]] = {}
    success_distribution_total_cells: list[str] | None = None
    label_rows: dict[str, list[str]] = {}
    label_total_cells: list[str] | None = None
    for line in report_lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0].isdigit() and len(cells) == 13:
            case_id = int(cells[0])
            if case_id in case_rows:
                raise ValueError(f"filter report repeats case {case_id}")
            case_rows[case_id] = cells
        elif cells and cells[0].isdigit() and len(cells) == 3:
            success_count = int(cells[0])
            if success_count in success_distribution_rows:
                raise ValueError(
                    f"filter report repeats success count {success_count}"
                )
            success_distribution_rows[success_count] = cells
        elif cells and cells[0] == "**总计**" and len(cells) == 13:
            total_cells = [cell.replace("**", "") for cell in cells]
        elif cells and cells[0] == "**总计**" and len(cells) == 3:
            success_distribution_total_cells = [
                cell.replace("**", "") for cell in cells
            ]
        elif (
            cells
            and len(cells) == 3
            and cells[0].startswith("`")
            and cells[0].endswith("`")
        ):
            answer_label = cells[0][1:-1]
            if answer_label in label_rows:
                raise ValueError(
                    f"filter report repeats answer label {answer_label!r}"
                )
            label_rows[answer_label] = cells
        elif cells and cells[0] == "**去重总计**" and len(cells) == 3:
            label_total_cells = [cell.replace("**", "") for cell in cells]

    expected_case_ids = set(state_by_case)
    if set(case_rows) != expected_case_ids:
        raise ValueError("filter report case ids do not match source state")
    all_durations: list[float] = []
    all_successful_durations: list[float] = []
    total_status_counts: Counter[str] = Counter()
    for case_id in sorted(expected_case_ids):
        status_counts = status_by_case[case_id]
        attempts = sum(status_counts.values())
        state_item = state_by_case[case_id]
        if attempts != int(state_item.get("total_attempts", -1)):
            raise ValueError(f"case {case_id}: attempt count mismatch")
        successes = status_counts["accepted"]
        if successes != int(state_item.get("accepted_count", -1)):
            raise ValueError(f"case {case_id}: accepted count mismatch")
        durations = durations_by_case[case_id]
        successful_durations = successful_durations_by_case[case_id]
        all_durations.extend(durations)
        all_successful_durations.extend(successful_durations)
        total_status_counts.update(status_counts)
        average_duration = sum(durations) / len(durations) if durations else None
        successful_average_duration = (
            sum(successful_durations) / len(successful_durations)
            if successful_durations
            else None
        )
        split_values = splits_by_case[case_id]
        if len(split_values) > 1:
            raise ValueError(f"case {case_id}: selected trajectories cross splits")
        split = next(iter(split_values), "excluded")
        expected_cells = [
            str(case_id),
            str(attempts),
            str(successes),
            f"{successes / attempts:.2%}",
            format_duration(average_duration),
            format_duration(successful_average_duration),
            f"{len(durations)}/{attempts}",
            str(status_counts["rejected"]),
            str(status_counts["interrupted"]),
            str(status_counts["infrastructure_failure"]),
            str(selected_by_case[case_id]),
            split,
            str(state_item.get("status")),
        ]
        if case_rows[case_id] != expected_cells:
            raise ValueError(f"filter report row mismatch for case {case_id}")

    total_attempts = sum(total_status_counts.values())
    average_duration = (
        sum(all_durations) / len(all_durations) if all_durations else None
    )
    successful_average_duration = (
        sum(all_successful_durations) / len(all_successful_durations)
        if all_successful_durations
        else None
    )
    expected_total_cells = [
        "总计",
        str(total_attempts),
        str(total_status_counts["accepted"]),
        f"{total_status_counts['accepted'] / total_attempts:.2%}",
        format_duration(average_duration),
        format_duration(successful_average_duration),
        f"{len(all_durations)}/{total_attempts}",
        str(total_status_counts["rejected"]),
        str(total_status_counts["interrupted"]),
        str(total_status_counts["infrastructure_failure"]),
        str(sum(selected_by_case.values())),
        "—",
        "—",
    ]
    if total_cells != expected_total_cells:
        raise ValueError("filter report total row mismatch")
    expected_success_distribution = Counter(
        status_by_case[case_id]["accepted"] for case_id in expected_case_ids
    )
    expected_success_distribution_rows = {
        successes: [
            str(successes),
            str(question_count),
            str(successes * question_count),
        ]
        for successes, question_count in expected_success_distribution.items()
    }
    if success_distribution_rows != expected_success_distribution_rows:
        raise ValueError("filter report success-count distribution mismatch")
    expected_success_distribution_total = [
        "总计",
        str(len(expected_case_ids)),
        str(total_status_counts["accepted"]),
    ]
    if (
        success_distribution_total_cells
        != expected_success_distribution_total
    ):
        raise ValueError("filter report success-count distribution total mismatch")
    expected_label_rows = {
        answer_label: [
            f"`{answer_label}`",
            str(len(label_case_ids[answer_label])),
            str(label_trajectory_counts[answer_label]),
        ]
        for answer_label in label_case_ids
    }
    if label_rows != expected_label_rows:
        raise ValueError("filter report answer-label distribution mismatch")
    expected_label_total = [
        "去重总计",
        str(len(selected_by_case)),
        str(sum(selected_by_case.values())),
    ]
    if label_total_cells != expected_label_total:
        raise ValueError("filter report answer-label total mismatch")


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
    filter_report_path = data_root / "curation" / FILTER_REPORT_NAME
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

    experiment_root = ROOT / manifest["source_experiment"]
    validate_filter_report(
        report_path=filter_report_path,
        experiment_root=experiment_root,
        curation=curation,
    )

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
    print("- per-case, success-count, and answer-label reports verified")


if __name__ == "__main__":
    main()
