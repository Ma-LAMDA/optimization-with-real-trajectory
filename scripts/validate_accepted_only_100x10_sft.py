#!/usr/bin/env python3
"""Validate an accepted-only 100x10 date-scoped SFT archive."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from convert_100x10_accepted_to_sft import reference_options
from validate_100x10_sft import (
    ROOT,
    answer_label_fault_type,
    check_row,
    load_json,
    load_jsonl,
    normalized_bytes,
    stable_digest,
)


DEFAULT_DATA_ROOT = ROOT / "data" / "2026-08-04"
CURATION_NAME = "accepted_trajectory_selection.json"
FILTER_REPORT_NAME = "FILTER_REPORT.md"
TRAIN_NAME = "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_NAME = "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check_source(path: Path, expected_digest: str, *, name: str) -> None:
    if not path.is_file() or stable_digest(path) != expected_digest:
        raise ValueError(f"{name} provenance mismatch")


def main() -> None:
    options = parse_args()
    data_root = options.data_root.resolve()
    raw_dir = data_root / "raw"
    curation_path = data_root / "curation" / CURATION_NAME
    report_path = data_root / "curation" / FILTER_REPORT_NAME
    train_path = data_root / "sft" / TRAIN_NAME
    validation_path = data_root / "sft" / VALIDATION_NAME
    manifest_path = data_root / "sft" / MANIFEST_NAME

    curation = load_json(curation_path)
    manifest = load_json(manifest_path)
    train_rows = load_jsonl(train_path)
    validation_rows = load_jsonl(validation_path)
    curation_digest = stable_digest(curation_path)
    if curation.get("schema_version") != "codex-ip-accepted-trajectory-curation.v4":
        raise ValueError("unexpected curation schema")
    if manifest.get("schema_version") != "qwen36-reasoning-decision-sft.v6":
        raise ValueError("unexpected manifest schema")
    if (
        manifest.get("curation_file") != curation_path.relative_to(ROOT).as_posix()
        or manifest.get("curation_sha256_lf_normalized") != curation_digest
    ):
        raise ValueError("manifest curation provenance mismatch")

    source_pairs = (
        ("source_dataset", "source_dataset_sha256_lf_normalized"),
        ("source_accepted_index", "source_accepted_index_sha256_lf_normalized"),
        ("source_state", "source_state_sha256_lf_normalized"),
        ("source_manifest", "source_manifest_sha256_lf_normalized"),
    )
    for path_key, digest_key in source_pairs:
        manifest_path_value = manifest.get(path_key)
        digest = manifest.get(digest_key)
        if not isinstance(manifest_path_value, str) or not isinstance(digest, str):
            raise ValueError(f"manifest is missing {path_key}")
        check_source(
            resolve_repo_path(manifest_path_value), digest, name=path_key
        )
        if (
            curation.get(path_key) != manifest_path_value
            or curation.get(digest_key) != digest
        ):
            raise ValueError(f"curation {path_key} provenance mismatch")

    source_state = load_json(resolve_repo_path(manifest["source_state"]))
    source_manifest = load_json(resolve_repo_path(manifest["source_manifest"]))
    accepted_index = load_json(resolve_repo_path(manifest["source_accepted_index"]))
    source_rows = load_jsonl(resolve_repo_path(manifest["source_dataset"]))
    if source_state.get("status") != "completed":
        raise ValueError("source state is not completed")
    if (
        source_manifest.get("status") != "completed"
        or source_manifest.get("attempt_retention") != "accepted_only"
        or source_manifest.get("target_correct_per_sample") != 10
    ):
        raise ValueError("source manifest is not a completed accepted-only 100x10 run")

    identifiers: set[str] = set()
    source_ids: set[str] = set()
    train_cases: set[int] = set()
    validation_cases: set[int] = set()
    selected_counts_by_case: Counter[int] = Counter()
    fault_types_by_case: defaultdict[int, set[str]] = defaultdict(set)
    for split_name, rows, cases in (
        ("train", train_rows, train_cases),
        ("validation", validation_rows, validation_cases),
    ):
        for row in rows:
            identifier, source_id, case_id = check_row(
                row,
                expected_split=split_name,
                curation_path=curation_path,
                curation_digest=curation_digest,
            )
            if identifier in identifiers or source_id in source_ids:
                raise ValueError("duplicate SFT sample or source trajectory")
            identifiers.add(identifier)
            source_ids.add(source_id)
            cases.add(case_id)
            selected_counts_by_case[case_id] += 1
            fault_types_by_case[case_id].update(
                answer_label_fault_type(answer_label)
                for answer_label in row["metadata"]["actual_result_items"]
            )

    if train_cases & validation_cases:
        raise ValueError("train and validation case groups overlap")
    all_cases = train_cases | validation_cases
    if set(fault_types_by_case) != all_cases or any(
        len(fault_types_by_case[case_id]) != 1 for case_id in all_cases
    ):
        raise ValueError("each selected case must belong to one fault type")
    case_fault_types = {
        case_id: next(iter(fault_types_by_case[case_id])) for case_id in all_cases
    }

    split = manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("manifest split metadata is missing")
    per_type = split.get("validation_case_count_per_fault_type")
    required = split.get("required_selected_trajectories_per_validation_case")
    if (
        split.get("strategy")
        != "leave_n_full_cases_out_per_fault_type_by_success_rate"
        or split.get("group_key") != "case_id"
        or split.get("stratification_key") != "fault_type"
        or split.get("success_rate_formula")
        != "accepted/(accepted+incorrect+format_error)"
        or split.get("required_success_rate") != 1.0
        or split.get("selection_rule")
        != "success_rate_equals_1_then_max_case_id"
        or split.get("fallback_selection_rule")
        != "max_success_rate_then_max_case_id"
        or not isinstance(per_type, int)
        or per_type < 1
        or not isinstance(required, int)
        or required < 1
    ):
        raise ValueError("manifest split strategy is malformed")

    relaxed_fault_types = split.get("relaxed_fault_types")
    if relaxed_fault_types != ["全局STP未使能"]:
        raise ValueError("unexpected relaxed fault-type policy")

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
        source_cases_by_fault_type[next(iter(source_fault_types))].append(case_id)

    state_by_case = {
        int(item["original_id"]): item
        for item in source_state.get("samples", [])
        if isinstance(item, dict)
    }
    if set(state_by_case) != {
        int(source_row["id"]) for source_row in source_rows
    }:
        raise ValueError("source state does not cover every source case")
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
        if selected_counts_by_case[case_id] == required:
            full_cases_by_fault_type[fault_type].append(case_id)
            if success_rate_by_case[case_id] == 1.0:
                perfect_cases_by_fault_type[fault_type].append(case_id)
    expected_validation_cases_by_fault_type: dict[str, list[int]] = {}
    for fault_type in sorted(source_cases_by_fault_type):
        if fault_type in relaxed_fault_types:
            ranked = sorted(
                full_cases_by_fault_type[fault_type],
                key=lambda case_id: (success_rate_by_case[case_id], case_id),
                reverse=True,
            )
        else:
            ranked = sorted(perfect_cases_by_fault_type[fault_type], reverse=True)
        if len(ranked) < per_type:
            raise ValueError(
                f"fault type has too few eligible validation cases: {fault_type}"
            )
        expected_validation_cases_by_fault_type[fault_type] = ranked[:per_type]
    actual_validation_cases_by_fault_type: defaultdict[str, list[int]] = defaultdict(list)
    for case_id in sorted(validation_cases, reverse=True):
        actual_validation_cases_by_fault_type[case_fault_types[case_id]].append(case_id)
    actual_validation_mapping = {
        fault_type: case_ids
        for fault_type, case_ids in sorted(actual_validation_cases_by_fault_type.items())
    }
    if actual_validation_mapping != expected_validation_cases_by_fault_type:
        raise ValueError("validation cases are not the deterministic full-query holdout")

    expected_validation_ids = sorted(validation_cases)
    expected_train_ids = sorted(train_cases)
    if (
        split.get("train") != len(train_rows)
        or split.get("validation") != len(validation_rows)
        or split.get("train_case_ids") != expected_train_ids
        or split.get("validation_case_ids") != expected_validation_ids
        or split.get("validation_cases_by_fault_type")
        != expected_validation_cases_by_fault_type
        or split.get("source_query_counts_by_fault_type")
        != {
            fault_type: len(case_ids)
            for fault_type, case_ids in sorted(source_cases_by_fault_type.items())
        }
        or split.get("full_query_counts_by_fault_type")
        != {
            fault_type: len(full_cases_by_fault_type[fault_type])
            for fault_type in sorted(source_cases_by_fault_type)
        }
        or split.get("perfect_success_query_counts_by_fault_type")
        != {
            fault_type: len(perfect_cases_by_fault_type[fault_type])
            for fault_type in sorted(source_cases_by_fault_type)
        }
        or split.get("perfect_success_case_ids_by_fault_type")
        != {
            fault_type: sorted(perfect_cases_by_fault_type[fault_type])
            for fault_type in sorted(source_cases_by_fault_type)
        }
        or split.get("case_groups_disjoint") is not True
        or len(validation_cases) != per_type * len(expected_validation_cases_by_fault_type)
    ):
        raise ValueError("manifest split counts or inventory mismatch")
    expected_curation_split = {
        key: value for key, value in split.items() if key not in {"train", "validation"}
    }
    if curation.get("split") != expected_curation_split:
        raise ValueError("curation split metadata mismatch")

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
    raw_paths = {path.relative_to(ROOT).as_posix() for path in raw_files}
    trajectory_raw_paths = {
        item["raw_file"] for item in trajectories if isinstance(item, dict)
    }
    if raw_paths != trajectory_raw_paths:
        raise ValueError("raw normalized trajectory inventory mismatch")
    for item in trajectories:
        raw_path = resolve_repo_path(item["raw_file"])
        if stable_digest(raw_path) != item["raw_sha256_lf_normalized"]:
            raise ValueError(f"raw trajectory hash mismatch: {raw_path}")

    accepted_total = sum(
        int(sample.get("accepted_count", 0))
        for sample in accepted_index.get("samples", [])
        if isinstance(sample, dict)
    )
    outcome_counts = source_state.get("outcome_counts")
    recorded_outcome_counts = {
        status: int(outcome_counts.get(status, 0))
        for status in ("accepted", "format_error", "incorrect")
    } if isinstance(outcome_counts, dict) else {}
    recorded_attempt_count = sum(recorded_outcome_counts.values())
    if (
        not isinstance(outcome_counts, dict)
        or int(outcome_counts.get("accepted", -1)) != accepted_total
        or accepted_total != len(trajectories)
        or manifest.get("accepted_candidate_count") != len(trajectories)
        or manifest.get("selected_trajectory_count") != len(source_ids)
        or manifest.get("excluded_candidate_count")
        != len(trajectories) - len(selected)
        or manifest.get("source_attempt_count") != recorded_attempt_count
        or manifest.get("source_attempt_count_scope")
        != "model_valid_outcomes_only"
        or manifest.get("filtered_nonaccepted_attempt_count")
        != recorded_attempt_count - accepted_total
        or manifest.get("selection", {}).get("source_attempt_status_counts")
        != recorded_outcome_counts
        or curation.get("source_attempt_status_counts")
        != recorded_outcome_counts
        or counts.get("source_attempt_count_scope")
        != "model_valid_outcomes_only"
        or counts.get("source_attempts") != recorded_attempt_count
        or counts.get("filtered_nonaccepted_attempts")
        != recorded_attempt_count - accepted_total
    ):
        raise ValueError("source, manifest, and curation totals disagree")

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
        output_path = ROOT / output["path"]
        content = normalized_bytes(output_path)
        if (
            output.get("samples") != len(expected_outputs[output["path"]])
            or output.get("normalized_bytes") != len(content)
            or output.get("sha256_lf_normalized")
            != hashlib.sha256(content).hexdigest()
        ):
            raise ValueError(f"output metadata mismatch: {output_path}")

    report = report_path.read_text(encoding="utf-8")
    if "## 验证集划分" not in report:
        raise ValueError("filter report is missing validation split section")
    if "## 每个 label 的 100% 成功率候选题" not in report:
        raise ValueError("filter report is missing perfect-success candidates")
    for fault_type in sorted(source_cases_by_fault_type):
        perfect_case_ids = sorted(perfect_cases_by_fault_type[fault_type])
        expected_candidate_row = (
            f"| `{fault_type}` | {len(source_cases_by_fault_type[fault_type])} | "
            f"{len(full_cases_by_fault_type[fault_type])} | "
            f"{len(perfect_case_ids)} | "
            f"{', '.join(str(case_id) for case_id in perfect_case_ids) or '—'} |"
        )
        if expected_candidate_row not in report:
            raise ValueError(f"filter report candidate row mismatch: {fault_type}")
    for fault_type, case_ids in expected_validation_cases_by_fault_type.items():
        expected_row = (
            f"| `{fault_type}` | {', '.join(str(case_id) for case_id in case_ids)} | "
            f"{', '.join(f'{success_rate_by_case[case_id]:.2%}' for case_id in case_ids)} | "
            f"{required * len(case_ids)} |"
        )
        if expected_row not in report:
            raise ValueError(f"filter report split row mismatch: {fault_type}")

    print(f"Validation passed for {len(source_ids)} fully correct trajectories")
    print(f"- accepted candidates: {len(trajectories)}")
    print(f"- train: {len(train_rows)} across {len(train_cases)} cases")
    print(
        f"- validation: {len(validation_rows)} across "
        f"{len(validation_cases)} cases ({sorted(validation_cases)})"
    )
    print(
        f"- fault types: {len(expected_validation_cases_by_fault_type)}, "
        f"{per_type} full queries each; relaxed: {relaxed_fault_types}"
    )
    print("- train/validation case overlap: 0")


if __name__ == "__main__":
    main()
