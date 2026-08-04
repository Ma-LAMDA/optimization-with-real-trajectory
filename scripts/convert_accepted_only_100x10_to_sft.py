#!/usr/bin/env python3
"""Archive an accepted-only 100x10 run as grouped reasoning/decision SFT data."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from convert_100x10_accepted_to_sft import (
    ROOT,
    SYSTEM,
    answer_label_fault_type,
    build_user_prompt,
    candidate_reasons,
    canonical_result,
    decision_reasoning,
    label,
    load_json,
    load_jsonl,
    output_metadata,
    raw_path,
    reference_options,
    stable_digest,
    write_json,
    write_jsonl,
    write_text,
)


DEFAULT_EXPERIMENT_ROOT = (
    ROOT / "experiments" / "2026-08-02-ip_codex_gpt56-sol_100x10"
)
DEFAULT_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "2026-08-04"
CURATION_NAME = "accepted_trajectory_selection.json"
FILTER_REPORT_NAME = "FILTER_REPORT.md"
TRAIN_NAME = "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_NAME = "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_NAME = "manifest.json"
RELAXED_SUCCESS_RATE_FAULT_TYPES = ("全局STP未使能",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validation-cases-per-fault-type", type=int, default=2
    )
    parser.add_argument(
        "--required-validation-trajectories", type=int, default=10
    )
    return parser.parse_args()


def as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def source_preflight(
    *,
    experiment_root: Path,
    dataset_path: Path,
    source_manifest: dict[str, Any],
    state: dict[str, Any],
    accepted_index: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], Counter[str]]:
    if source_manifest.get("status") != "completed":
        raise ValueError("source experiment manifest is not completed")
    if state.get("status") != "completed":
        raise ValueError("source experiment state is not completed")
    if source_manifest.get("model") != "gpt-5.6-sol":
        raise ValueError("source experiment model is not gpt-5.6-sol")
    if source_manifest.get("attempt_retention") != "accepted_only":
        raise ValueError("source experiment is not accepted-only")
    if source_manifest.get("target_correct_per_sample") != 10:
        raise ValueError("source experiment target is not 10 accepted trajectories")
    if source_manifest.get("record_count") != len(source_rows):
        raise ValueError("source manifest record count does not match the dataset")
    if source_manifest.get("source_sha256") != hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest():
        raise ValueError("source dataset raw hash does not match the experiment manifest")

    source_by_index = {
        index: row for index, row in enumerate(source_rows, start=1)
    }
    state_samples = state.get("samples")
    index_samples = accepted_index.get("samples")
    if not isinstance(state_samples, list) or not isinstance(index_samples, list):
        raise ValueError("source state or accepted index has no sample inventory")
    if len(state_samples) != len(source_rows) or len(index_samples) != len(source_rows):
        raise ValueError("source state or accepted index does not cover every source row")

    state_by_row = {
        int(item["row_index"]): item
        for item in state_samples
        if isinstance(item, dict)
    }
    index_by_row = {
        int(item["row_index"]): item
        for item in index_samples
        if isinstance(item, dict)
    }
    expected_rows = set(source_by_index)
    if set(state_by_row) != expected_rows or set(index_by_row) != expected_rows:
        raise ValueError("source state or accepted index row coverage is malformed")

    accepted_total = 0
    for row_index, source_row in source_by_index.items():
        state_item = state_by_row[row_index]
        index_item = index_by_row[row_index]
        case_id = int(source_row["id"])
        if (
            int(state_item.get("original_id", -1)) != case_id
            or int(index_item.get("original_id", -1)) != case_id
        ):
            raise ValueError(f"row {row_index}: source identity mismatch")
        state_accepted = int(state_item.get("accepted_count", -1))
        index_accepted = int(index_item.get("accepted_count", -1))
        if state_accepted != index_accepted:
            raise ValueError(f"row {row_index}: accepted count mismatch")
        if state_accepted < 0 or state_accepted > 10:
            raise ValueError(f"row {row_index}: invalid accepted count")
        mapping = index_item.get("mapping")
        if not isinstance(mapping, dict) or len(mapping) != state_accepted:
            raise ValueError(f"row {row_index}: accepted mapping count mismatch")
        accepted_total += state_accepted

    all_outcome_counts = Counter(
        {
            str(key): int(value)
            for key, value in as_mapping(
                state.get("outcome_counts"), name="state outcome_counts"
            ).items()
        }
    )
    if all_outcome_counts["accepted"] != accepted_total:
        raise ValueError("global accepted count does not match accepted index")
    recorded_outcome_counts = Counter(
        {
            status: all_outcome_counts[status]
            for status in ("accepted", "incorrect", "format_error")
        }
    )
    return source_by_index, state_by_row, recorded_outcome_counts


def main() -> None:
    options = parse_args()
    if options.validation_cases_per_fault_type < 1:
        raise ValueError("validation cases per fault type must be positive")
    if options.required_validation_trajectories < 1:
        raise ValueError("required validation trajectories must be positive")

    experiment_root = options.experiment_root.resolve()
    dataset_path = options.dataset.resolve()
    output_root = options.output_root.resolve()
    report_dir = experiment_root / "results" / "report"
    runs_dir = experiment_root / "results" / "runs"
    accepted_index_path = report_dir / "accepted_index.json"
    state_path = report_dir / "state.json"
    source_manifest_path = report_dir / "manifest.json"

    raw_dir = output_root / "raw"
    curation_path = output_root / "curation" / CURATION_NAME
    filter_report_path = output_root / "curation" / FILTER_REPORT_NAME
    train_path = output_root / "sft" / TRAIN_NAME
    validation_path = output_root / "sft" / VALIDATION_NAME
    manifest_path = output_root / "sft" / MANIFEST_NAME

    accepted_index = load_json(accepted_index_path)
    state = load_json(state_path)
    source_manifest = load_json(source_manifest_path)
    source_rows = load_jsonl(dataset_path)
    source_by_index, state_by_row, outcome_counts = source_preflight(
        experiment_root=experiment_root,
        dataset_path=dataset_path,
        source_manifest=source_manifest,
        state=state,
        accepted_index=accepted_index,
        source_rows=source_rows,
    )

    processed: list[dict[str, Any]] = []
    raw_documents: dict[str, dict[str, Any]] = {}
    seen_attempt_paths: set[str] = set()
    seen_event_paths: set[str] = set()
    seen_thread_ids: set[str] = set()
    referenced_metadata_paths: set[str] = set()
    accepted_durations_by_case: defaultdict[int, list[float]] = defaultdict(list)

    for sample in accepted_index["samples"]:
        row_index = int(sample["row_index"])
        case_id = int(sample["original_id"])
        source_row = source_by_index[row_index]
        mapping = sample["mapping"]
        expected_keys = [
            f"success_{index:02d}"
            for index in range(1, int(sample["accepted_count"]) + 1)
        ]
        if list(mapping) != expected_keys:
            raise ValueError(f"row {row_index}: success slots are not sequential")

        for success_key, index_item in mapping.items():
            if not isinstance(index_item, dict):
                raise ValueError(f"row {row_index} {success_key}: malformed mapping")
            attempt_relative = str(index_item["attempt_path"])
            events_relative = str(index_item["events_path"])
            thread_id = str(index_item["thread_id"])
            if attempt_relative in seen_attempt_paths:
                raise ValueError(f"duplicate accepted attempt {attempt_relative}")
            if events_relative in seen_event_paths:
                raise ValueError(f"duplicate accepted events {events_relative}")
            if thread_id in seen_thread_ids:
                raise ValueError(f"duplicate accepted thread {thread_id}")
            seen_attempt_paths.add(attempt_relative)
            seen_event_paths.add(events_relative)
            seen_thread_ids.add(thread_id)
            referenced_metadata_paths.add(
                (Path(attempt_relative) / "metadata.json").as_posix()
            )

            attempt_dir = experiment_root / attempt_relative
            source_record_path = (
                experiment_root
                / "results"
                / "questions"
                / f"q{case_id:04d}"
                / "source_record.json"
            )
            reasons, details = candidate_reasons(
                attempt_dir=attempt_dir,
                source_record_path=source_record_path,
                index_item=index_item,
                sample=sample,
                source_row=source_row,
            )
            success_slot = int(index_item["success_slot"])
            source_id = f"q{case_id:04d}_success_{success_slot:02d}"
            raw_file = raw_path(raw_dir, case_id, success_slot)
            selected = not reasons
            duration = details.get("metadata", {}).get("duration_seconds")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                accepted_durations_by_case[case_id].append(float(duration))

            raw_document = {
                "schema_version": "codex-ip-normalized-accepted-trajectory.v1",
                "id": source_id,
                "case_id": case_id,
                "row_index": row_index,
                "success_slot": success_slot,
                "attempt_index": int(index_item["attempt_index"]),
                "source_record": details.get("source_record"),
                "agent_messages": details.get("agent_messages", []),
                "final_answer": details.get("final_text"),
                "actual_result_items": details.get("prediction"),
                "reference_answer_options": details.get("reference_options", []),
                "answer_matches_reference": (
                    details.get("prediction") in details.get("reference_options", [])
                    if details
                    else False
                ),
                "independent_judgment": {
                    "correct": details.get("judgment", {}).get("correct"),
                    "comparator": details.get("judgment", {}).get("comparator"),
                    "judgment_file": label(attempt_dir / "judgment.json"),
                    "judgment_sha256_lf_normalized": (
                        stable_digest(attempt_dir / "judgment.json")
                        if (attempt_dir / "judgment.json").is_file()
                        else None
                    ),
                },
                "source": {
                    "experiment": label(experiment_root),
                    "attempt_directory": label(attempt_dir),
                    "events_file": label(attempt_dir / "events.jsonl"),
                    "events_sha256_lf_normalized": (
                        stable_digest(attempt_dir / "events.jsonl")
                        if (attempt_dir / "events.jsonl").is_file()
                        else None
                    ),
                    "final_answer_file": label(attempt_dir / "final_answer.txt"),
                    "final_answer_sha256_lf_normalized": (
                        stable_digest(attempt_dir / "final_answer.txt")
                        if (attempt_dir / "final_answer.txt").is_file()
                        else None
                    ),
                    "source_record_file": label(source_record_path),
                    "source_record_sha256_lf_normalized": (
                        stable_digest(source_record_path)
                        if source_record_path.is_file()
                        else None
                    ),
                    "thread_id": thread_id,
                    "duration_seconds": duration,
                },
            }
            raw_documents[label(raw_file)] = raw_document
            processed.append(
                {
                    "id": source_id,
                    "case_id": case_id,
                    "row_index": row_index,
                    "success_slot": success_slot,
                    "attempt_index": int(index_item["attempt_index"]),
                    "selected": selected,
                    "split": "pending" if selected else "excluded",
                    "review_status": "draft",
                    "selection_reason": (
                        "accepted_independent_exact_match_and_clean_evidence"
                        if selected
                        else "excluded"
                    ),
                    "exclusion_reasons": reasons,
                    "evidence_marker_hits": details.get("evidence_hits", []),
                    "evidence": details.get("evidence") if selected else None,
                    "evidence_message_index": (
                        len(details.get("agent_messages", [])) - 2
                        if len(details.get("agent_messages", [])) >= 2
                        else None
                    ),
                    "final_message_index": (
                        len(details.get("agent_messages", [])) - 1
                        if details.get("agent_messages")
                        else None
                    ),
                    "actual_result_items": details.get("prediction"),
                    "reference_answer_options": details.get("reference_options", []),
                    "source_answer_format_normalized": (
                        selected
                        and details.get("final_text")
                        != canonical_result(details["prediction"])
                    ),
                    "raw_file": label(raw_file),
                    "events_file": label(attempt_dir / "events.jsonl"),
                    "events_sha256_lf_normalized": (
                        stable_digest(attempt_dir / "events.jsonl")
                        if (attempt_dir / "events.jsonl").is_file()
                        else None
                    ),
                    "judgment_file": label(attempt_dir / "judgment.json"),
                    "judgment_sha256_lf_normalized": (
                        stable_digest(attempt_dir / "judgment.json")
                        if (attempt_dir / "judgment.json").is_file()
                        else None
                    ),
                    "thread_id": thread_id,
                }
            )

    disk_metadata_paths = {
        path.relative_to(experiment_root).as_posix()
        for path in runs_dir.glob("q*_r*/attempt_*/metadata.json")
    }
    if disk_metadata_paths != referenced_metadata_paths:
        raise ValueError("retained attempt directories do not exactly match accepted index")

    eligible_case_ids = sorted(
        {item["case_id"] for item in processed if item["selected"]}
    )
    selected_counts_by_case: Counter[int] = Counter()
    fault_types_by_case: defaultdict[int, set[str]] = defaultdict(set)
    for item in processed:
        if not item["selected"]:
            continue
        case_id = int(item["case_id"])
        selected_counts_by_case[case_id] += 1
        fault_types_by_case[case_id].update(
            answer_label_fault_type(answer_label)
            for answer_label in item["actual_result_items"]
        )
    if any(len(fault_types_by_case[case_id]) != 1 for case_id in eligible_case_ids):
        raise ValueError("each eligible case must belong to exactly one fault type")

    case_fault_types = {
        case_id: next(iter(fault_types_by_case[case_id]))
        for case_id in eligible_case_ids
    }
    source_case_fault_types: dict[int, str] = {}
    source_cases_by_fault_type: defaultdict[str, list[int]] = defaultdict(list)
    for source_row in source_rows:
        case_id = int(source_row["id"])
        source_fault_types = {
            answer_label_fault_type(answer_label)
            for option in reference_options(str(source_row.get("answer", "")))
            for answer_label in option
        }
        if len(source_fault_types) != 1:
            raise ValueError(
                f"source case {case_id} must belong to exactly one fault type"
            )
        fault_type = next(iter(source_fault_types))
        source_case_fault_types[case_id] = fault_type
        source_cases_by_fault_type[fault_type].append(case_id)

    state_by_case = {
        int(item["original_id"]): item for item in state_by_row.values()
    }
    success_rate_by_case: dict[int, float] = {}
    for case_id, state_item in state_by_case.items():
        accepted = int(state_item.get("accepted_count", 0))
        wrong = int(state_item.get("total_wrong", 0))
        valid_attempts = accepted + wrong
        success_rate_by_case[case_id] = (
            accepted / valid_attempts if valid_attempts else 0.0
        )

    full_cases_by_fault_type: defaultdict[str, list[int]] = defaultdict(list)
    perfect_cases_by_fault_type: defaultdict[str, list[int]] = defaultdict(list)
    for case_id, fault_type in case_fault_types.items():
        if selected_counts_by_case[case_id] == options.required_validation_trajectories:
            full_cases_by_fault_type[fault_type].append(case_id)
            if success_rate_by_case[case_id] == 1.0:
                perfect_cases_by_fault_type[fault_type].append(case_id)
    for fault_type, case_ids in full_cases_by_fault_type.items():
        case_ids.sort(reverse=True)
    for fault_type, case_ids in perfect_cases_by_fault_type.items():
        case_ids.sort(reverse=True)
    all_fault_types = sorted(source_cases_by_fault_type)
    missing_strict = [
        fault_type
        for fault_type in all_fault_types
        if fault_type not in RELAXED_SUCCESS_RATE_FAULT_TYPES
        and len(perfect_cases_by_fault_type[fault_type])
        < options.validation_cases_per_fault_type
    ]
    if missing_strict:
        raise ValueError(
            "not enough 100%-success full validation cases for fault types: "
            f"{missing_strict}"
        )
    missing_fallback = [
        fault_type
        for fault_type in RELAXED_SUCCESS_RATE_FAULT_TYPES
        if len(full_cases_by_fault_type[fault_type])
        < options.validation_cases_per_fault_type
    ]
    if missing_fallback:
        raise ValueError(
            f"not enough full fallback validation cases: {missing_fallback}"
        )

    validation_cases_by_fault_type: dict[str, list[int]] = {}
    for fault_type in all_fault_types:
        if fault_type in RELAXED_SUCCESS_RATE_FAULT_TYPES:
            ranked = sorted(
                full_cases_by_fault_type[fault_type],
                key=lambda case_id: (success_rate_by_case[case_id], case_id),
                reverse=True,
            )
        else:
            ranked = perfect_cases_by_fault_type[fault_type]
        validation_cases_by_fault_type[fault_type] = ranked[
            : options.validation_cases_per_fault_type
        ]
    validation_case_ids = sorted(
        case_id
        for case_ids in validation_cases_by_fault_type.values()
        for case_id in case_ids
    )
    validation_case_id_set = set(validation_case_ids)
    for item in processed:
        if item["selected"]:
            item["split"] = (
                "validation"
                if item["case_id"] in validation_case_id_set
                else "train"
            )
    train_case_ids = sorted(
        {item["case_id"] for item in processed if item["split"] == "train"}
    )
    if set(train_case_ids) & validation_case_id_set:
        raise ValueError("train and validation case groups overlap")

    for item in processed:
        raw_file = Path(item["raw_file"])
        write_json(raw_file, raw_documents[item["raw_file"]])
        item["raw_sha256_lf_normalized"] = stable_digest(raw_file)

    split_counts = Counter(item["split"] for item in processed)
    exclusion_counts = Counter(
        reason for item in processed for reason in item["exclusion_reasons"]
    )
    source_attempt_count = sum(outcome_counts.values())
    filtered_nonaccepted = source_attempt_count - outcome_counts["accepted"]
    case_quality = []
    for row_index, source_row in source_by_index.items():
        case_id = int(source_row["id"])
        state_item = state_by_row[row_index]
        case_items = [item for item in processed if item["row_index"] == row_index]
        selected_items = [item for item in case_items if item["selected"]]
        case_quality.append(
            {
                "case_id": case_id,
                "row_index": row_index,
                "valid_model_attempts": int(state_item.get("accepted_count", 0))
                + int(state_item.get("total_wrong", 0)),
                "model_valid_success_rate": success_rate_by_case[case_id],
                "success_rate_is_100_percent": (
                    success_rate_by_case[case_id] == 1.0
                ),
                "accepted_candidates": len(case_items),
                "selected_trajectories": len(selected_items),
                "terminal_status": state_item.get("status"),
                "split": selected_items[0]["split"] if selected_items else "excluded",
            }
        )

    split_metadata = {
        "strategy": "leave_n_full_cases_out_per_fault_type_by_success_rate",
        "group_key": "case_id",
        "stratification_key": "fault_type",
        "validation_case_count_per_fault_type": options.validation_cases_per_fault_type,
        "required_selected_trajectories_per_validation_case": (
            options.required_validation_trajectories
        ),
        "success_rate_formula": "accepted/(accepted+incorrect+format_error)",
        "required_success_rate": 1.0,
        "selection_rule": "success_rate_equals_1_then_max_case_id",
        "relaxed_fault_types": list(RELAXED_SUCCESS_RATE_FAULT_TYPES),
        "fallback_selection_rule": "max_success_rate_then_max_case_id",
        "source_query_counts_by_fault_type": {
            fault_type: len(source_cases_by_fault_type[fault_type])
            for fault_type in all_fault_types
        },
        "full_query_counts_by_fault_type": {
            fault_type: len(full_cases_by_fault_type[fault_type])
            for fault_type in all_fault_types
        },
        "perfect_success_query_counts_by_fault_type": {
            fault_type: len(perfect_cases_by_fault_type[fault_type])
            for fault_type in all_fault_types
        },
        "perfect_success_case_ids_by_fault_type": {
            fault_type: sorted(perfect_cases_by_fault_type[fault_type])
            for fault_type in all_fault_types
        },
        "validation_cases_by_fault_type": validation_cases_by_fault_type,
        "validation_case_ids": validation_case_ids,
        "train_case_ids": train_case_ids,
        "case_groups_disjoint": True,
    }
    curation_document = {
        "schema_version": "codex-ip-accepted-trajectory-curation.v4",
        "source_experiment": label(experiment_root),
        "source_dataset": label(dataset_path),
        "source_dataset_sha256_lf_normalized": stable_digest(dataset_path),
        "source_accepted_index": label(accepted_index_path),
        "source_accepted_index_sha256_lf_normalized": stable_digest(
            accepted_index_path
        ),
        "source_state": label(state_path),
        "source_state_sha256_lf_normalized": stable_digest(state_path),
        "source_manifest": label(source_manifest_path),
        "source_manifest_sha256_lf_normalized": stable_digest(
            source_manifest_path
        ),
        "selection": {
            "required_source_state": "completed",
            "required_source_retention": "accepted_only",
            "required_metadata_status": "accepted",
            "required_judgment_correct": True,
            "answer_filter": "independent_exact_fault_set_match_with_alternatives",
            "require_final_event_match": True,
            "require_clean_pre_final_evidence": True,
            "review_status": "draft",
        },
        "split": split_metadata,
        "counts": {
            "source_attempt_count_scope": "model_valid_outcomes_only",
            "source_records": len(source_rows),
            "source_attempts": source_attempt_count,
            "accepted_candidates": len(processed),
            "selected": split_counts["train"] + split_counts["validation"],
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "excluded_candidates": split_counts["excluded"],
            "filtered_nonaccepted_attempts": filtered_nonaccepted,
        },
        "source_attempt_status_counts": dict(sorted(outcome_counts.items())),
        "candidate_exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "eligible_case_ids": eligible_case_ids,
        "case_quality": case_quality,
        "trajectories": processed,
    }
    write_json(curation_path, curation_document)
    curation_digest = stable_digest(curation_path)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for item in processed:
        if not item["selected"]:
            continue
        raw_document = raw_documents[item["raw_file"]]
        actual_items = item["actual_result_items"]
        row = {
            "id": f"{item['id']}_decision",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        raw_document["source_record"], str(item["evidence"])
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"<think>\n{decision_reasoning(actual_items)}\n</think>\n\n"
                        f"{canonical_result(actual_items)}"
                    ),
                },
            ],
            "metadata": {
                "dataset_type": "reasoning_decision",
                "target_type": "decision",
                "review_status": "draft",
                "split": item["split"],
                "source_id": item["id"],
                "case_id": item["case_id"],
                "row_index": item["row_index"],
                "repeat_index": item["success_slot"],
                "attempt_index": item["attempt_index"],
                "source_file": item["raw_file"],
                "source_sha256_lf_normalized": item[
                    "raw_sha256_lf_normalized"
                ],
                "source_event_file": item["events_file"],
                "source_event_sha256_lf_normalized": item[
                    "events_sha256_lf_normalized"
                ],
                "source_judgment_file": item["judgment_file"],
                "source_judgment_sha256_lf_normalized": item[
                    "judgment_sha256_lf_normalized"
                ],
                "annotation_file": label(curation_path),
                "annotation_sha256_lf_normalized": curation_digest,
                "source_message_index": item["final_message_index"],
                "evidence_message_indices": [item["evidence_message_index"]],
                "evidence_count": 1,
                "actual_result_items": actual_items,
                "reference_answer_options": item["reference_answer_options"],
                "reference_answer_match": True,
                "source_answer_format_normalized": item[
                    "source_answer_format_normalized"
                ],
                "source_thread_id": item["thread_id"],
            },
        }
        if item["split"] == "validation":
            validation_rows.append(row)
        else:
            train_rows.append(row)

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    output_manifest = {
        "schema_version": "qwen36-reasoning-decision-sft.v6",
        "source_experiment": label(experiment_root),
        "source_dataset": label(dataset_path),
        "source_dataset_sha256_lf_normalized": stable_digest(dataset_path),
        "source_accepted_index": label(accepted_index_path),
        "source_accepted_index_sha256_lf_normalized": stable_digest(
            accepted_index_path
        ),
        "source_state": label(state_path),
        "source_state_sha256_lf_normalized": stable_digest(state_path),
        "source_manifest": label(source_manifest_path),
        "source_manifest_sha256_lf_normalized": stable_digest(
            source_manifest_path
        ),
        "source_attempt_count": source_attempt_count,
        "source_attempt_count_scope": "model_valid_outcomes_only",
        "accepted_candidate_count": len(processed),
        "selected_trajectory_count": len(train_rows) + len(validation_rows),
        "excluded_candidate_count": split_counts["excluded"],
        "filtered_nonaccepted_attempt_count": filtered_nonaccepted,
        "curation_file": label(curation_path),
        "curation_sha256_lf_normalized": curation_digest,
        "target_type_counts": {
            "train": {"decision": len(train_rows)},
            "validation": {"decision": len(validation_rows)},
        },
        "split": {
            **split_metadata,
            "train": len(train_rows),
            "validation": len(validation_rows),
        },
        "selection": {
            "included": len(train_rows) + len(validation_rows),
            "excluded_candidates": split_counts["excluded"],
            "filtered_nonaccepted_attempts": filtered_nonaccepted,
            "candidate_exclusion_reason_counts": dict(
                sorted(exclusion_counts.items())
            ),
            "source_attempt_status_counts": dict(sorted(outcome_counts.items())),
        },
        "outputs": [
            output_metadata(train_path, train_rows),
            output_metadata(validation_path, validation_rows),
        ],
    }
    write_json(manifest_path, output_manifest)

    label_case_ids: defaultdict[str, set[int]] = defaultdict(set)
    label_trajectory_counts: Counter[str] = Counter()
    fault_type_case_ids: defaultdict[str, set[int]] = defaultdict(set)
    fault_type_trajectory_counts: Counter[str] = Counter()
    fault_type_labels: defaultdict[str, set[str]] = defaultdict(set)
    for item in processed:
        if not item["selected"]:
            continue
        item_fault_types: set[str] = set()
        for answer_label in item["actual_result_items"]:
            fault_type = answer_label_fault_type(answer_label)
            label_case_ids[answer_label].add(item["case_id"])
            label_trajectory_counts[answer_label] += 1
            fault_type_case_ids[fault_type].add(item["case_id"])
            fault_type_labels[fault_type].add(answer_label)
            item_fault_types.add(fault_type)
        if len(item_fault_types) != 1:
            raise ValueError("a selected trajectory spans multiple fault types")
        for fault_type in item_fault_types:
            fault_type_trajectory_counts[fault_type] += 1

    accepted_durations = [
        value for values in accepted_durations_by_case.values() for value in values
    ]
    average_accepted_duration = (
        sum(accepted_durations) / len(accepted_durations)
        if accepted_durations
        else None
    )
    report_lines = [
        "# Accepted-only 100×10 轨迹过滤与 SFT 转换报告",
        "",
        f"- 来源实验：`{label(experiment_root)}`",
        f"- 来源模型有效 attempt：{source_attempt_count}",
        f"- accepted 候选：{len(processed)}",
        f"- 通过独立复核与证据清洁检查：{len(train_rows) + len(validation_rows)}",
        f"- 候选中排除：{split_counts['excluded']}",
        f"- 非 accepted 的模型有效 attempt：{filtered_nonaccepted}",
        f"- 训练集：{len(train_rows)} 条，{len(train_case_ids)} 个题号",
        f"- 验证集：{len(validation_rows)} 条，{len(validation_case_ids)} 个题号",
        f"- 验证题号：{'、'.join(str(case_id) for case_id in validation_case_ids)}",
        "- 训练/验证题号交集：0",
        (
            f"- accepted 轨迹平均生成耗时：{average_accepted_duration:.3f} 秒"
            if average_accepted_duration is not None
            else "- accepted 轨迹平均生成耗时：无有效记录"
        ),
        "",
        "## 统计口径",
        "",
        "- 归档只记录模型有效结果：accepted、incorrect 和 format_error。基础设施失败与中断不进入归档计数、报表或训练数据。",
        "- 每条 accepted 候选重新核对 metadata、独立 judgment、参考答案、最终事件、文件哈希和前置证据清洁性。",
        "- 训练/验证按 `case_id` 整题隔离。除 `全局STP未使能` 外，每类从满 10 条且模型有效 attempt 成功率为 100% 的题中按题号降序选择 2 题。",
        "- `全局STP未使能` 没有 100% 成功率候选，按显式回退规则从满 10 条题中依次按成功率、题号降序选择 q12、q2。",
        "- 所有样本均标记为 `draft`，正式训练前仍需领域审核。",
        "",
        "## 来源 attempt 状态",
        "",
        "| 状态 | 数量 |",
        "| --- | ---: |",
        *[
            f"| {status} | {count} |"
            for status, count in sorted(outcome_counts.items())
        ],
        "",
        "## 按答案 label 统计",
        "",
        "| 答案 label | 题目数量 | 轨迹数量 |",
        "| --- | ---: | ---: |",
        *[
            f"| `{answer_label}` | {len(label_case_ids[answer_label])} | {label_trajectory_counts[answer_label]} |"
            for answer_label in sorted(label_case_ids)
        ],
        "",
        "## 按故障类型合并统计",
        "",
        "故障类型取答案 label 第一个分号后的部分，忽略设备节点。",
        "",
        "| 故障类型 | 设备级 label 数 | 题目数量 | 轨迹数量 |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| `{fault_type}` | {len(fault_type_labels[fault_type])} | {len(fault_type_case_ids[fault_type])} | {fault_type_trajectory_counts[fault_type]} |"
            for fault_type in sorted(fault_type_case_ids)
        ],
        "",
        "## 每个 label 的 100% 成功率候选题",
        "",
        "成功率只使用模型有效 attempt，公式为 `accepted / (accepted + incorrect + format_error)`；候选题还必须有 10 条入选轨迹。",
        "",
        "| 故障类型 | 来源题数 | 满10条题数 | 100%成功率题数 | 合格题号 |",
        "| --- | ---: | ---: | ---: | --- |",
        *[
            (
                f"| `{fault_type}` | {len(source_cases_by_fault_type[fault_type])} | "
                f"{len(full_cases_by_fault_type[fault_type])} | "
                f"{len(perfect_cases_by_fault_type[fault_type])} | "
                f"{', '.join(str(case_id) for case_id in sorted(perfect_cases_by_fault_type[fault_type])) or '—'} |"
            )
            for fault_type in all_fault_types
        ],
        "",
        "## 验证集划分",
        "",
        "| 故障类型 | 验证题号 | 成功率 | 验证轨迹 |",
        "| --- | --- | --- | ---: |",
        *[
            f"| `{fault_type}` | {', '.join(str(case_id) for case_id in validation_cases_by_fault_type[fault_type])} | {', '.join(f'{success_rate_by_case[case_id]:.2%}' for case_id in validation_cases_by_fault_type[fault_type])} | {sum(selected_counts_by_case[case_id] for case_id in validation_cases_by_fault_type[fault_type])} |"
            for fault_type in sorted(validation_cases_by_fault_type)
        ],
        "",
        "## 逐题统计",
        "",
        "`Attempt` 只统计 accepted + incorrect + format_error；`错误` 为 incorrect + format_error。",
        "",
        "| 题号 | Attempt | 成功 | 错误 | SFT | 划分 | 终态 |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
        *[
            (
                f"| {int(source_by_index[row_index]['id'])} | "
                f"{int(state_by_row[row_index].get('accepted_count', 0)) + int(state_by_row[row_index].get('total_wrong', 0))} | "
                f"{int(state_by_row[row_index].get('accepted_count', 0))} | "
                f"{int(state_by_row[row_index].get('total_wrong', 0))} | "
                f"{selected_counts_by_case[int(source_by_index[row_index]['id'])]} | "
                f"{next((item['split'] for item in processed if item['row_index'] == row_index and item['selected']), 'excluded')} | "
                f"{state_by_row[row_index].get('status')} |"
            )
            for row_index in sorted(source_by_index)
        ],
        "",
        "## 候选排除原因",
        "",
        (
            f"无；{len(processed)} 条 accepted 候选全部通过复核。"
            if not exclusion_counts
            else "\n".join(
                f"- {reason}: {count}"
                for reason, count in sorted(exclusion_counts.items())
            )
        ),
    ]
    write_text(filter_report_path, "\n".join(report_lines))

    print(f"Source attempts: {source_attempt_count}")
    print(f"Accepted candidates: {len(processed)}")
    print(f"Selected fully correct trajectories: {len(train_rows) + len(validation_rows)}")
    print(f"Excluded accepted candidates: {split_counts['excluded']}")
    print(f"Train: {len(train_rows)} across {len(train_case_ids)} cases")
    print(
        f"Validation: {len(validation_rows)} across "
        f"{len(validation_case_ids)} cases ({validation_case_ids})"
    )


if __name__ == "__main__":
    main()
