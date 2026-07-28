#!/usr/bin/env python3
"""Convert the latest complete Codex experiment run into grouped SFT splits."""

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
RUNS_ROOT = (
    ROOT
    / "experiments"
    / "2026-07-27-ip_codex_train0629_14x10"
    / "results"
    / "runs"
)
DATA_ROOT = ROOT / "data" / "2026-07-28"
RAW_DIR = DATA_ROOT / "raw"
CURATION_DIR = DATA_ROOT / "curation"
SFT_DIR = DATA_ROOT / "sft"
CURATION_FILE = CURATION_DIR / "trajectory_selection.json"
TRAIN_FILE = SFT_DIR / "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_FILE = SFT_DIR / "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_FILE = SFT_DIR / "manifest.json"
DEFAULT_EXCLUDED_CASE_IDS = (25, 26, 27, 28)

SYSTEM = (
    "你是一名网络故障分析专家。请根据题目和当前已知证据逐步分析。"
    "信息不足时，说明下一步需要核验的事实以及核验目的；证据充分时，比较候选根因并作出决策。"
    "不得补充题目未提供的事实。先在 <think>...</think> 中给出简洁、可复核的思考，"
    "再输出当前计划、阶段判断或最终结论。"
)
DECISION_STAGE = (
    "当前证据已完成收敛。请给出最小根因集合，并严格遵守题目要求的答案格式。"
)
RESULT_ITEMS = re.compile(r'"([^"\r\n]+;[^"\r\n]+)"')
WINDOWS_PATH = re.compile(r"[A-Za-z]:\\")
FORBIDDEN_EVIDENCE_MARKERS = (
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
        "--run-dir",
        type=Path,
        help="Complete experiment run. Defaults to the latest successful run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DATA_ROOT,
        help="Date-scoped data directory.",
    )
    parser.add_argument(
        "--validation-case-id",
        type=int,
        help="Case held out as validation. Defaults to the highest case id.",
    )
    parser.add_argument(
        "--exclude-case-id",
        type=int,
        action="append",
        dest="excluded_case_ids",
        help=(
            "Case excluded from both splits. May be repeated. "
            "Defaults to 25, 26, 27, and 28."
        ),
    )
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def result_items(value: str) -> list[str]:
    return RESULT_ITEMS.findall(normalize_text(value))


def canonical_result(items: list[str]) -> str:
    lines = [
        f'"{item}",' if index < len(items) - 1 else f'"{item}"'
        for index, item in enumerate(items)
    ]
    return "\n".join(["<result>", "[", *lines, "]", "</result>"])


def discover_latest_complete_run(runs_root: Path) -> Path:
    candidates: list[tuple[str, Path]] = []
    for manifest_path in runs_root.rglob("manifest.json"):
        folder = manifest_path.parent
        manifest = load_json(manifest_path)
        runs = manifest.get("runs")
        if (
            manifest.get("status") != "succeeded"
            or not isinstance(runs, list)
            or not runs
            or any(
                not isinstance(run, dict) or run.get("status") != "succeeded"
                for run in runs
            )
            or manifest.get("required_successful_trajectories") != len(runs)
        ):
            continue
        started_at = manifest.get("started_at")
        candidates.append(
            (started_at if isinstance(started_at, str) else folder.name, folder)
        )
    if not candidates:
        raise ValueError(f"No complete successful runs found under {runs_root}")
    return max(candidates, key=lambda item: (item[0], item[1].name))[1]


def load_agent_messages(events_path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if (
                isinstance(event, dict)
                and event.get("type") == "item.completed"
                and isinstance(event.get("item"), dict)
                and event["item"].get("type") == "agent_message"
                and isinstance(event["item"].get("text"), str)
            ):
                messages.append(
                    {
                        "event_line": line_number,
                        "item_id": event["item"].get("id"),
                        "text": normalize_text(event["item"]["text"]),
                    }
                )
    return messages


def evidence_is_clean(value: str) -> tuple[bool, list[str]]:
    lowered = value.lower()
    hits = [
        marker
        for marker in FORBIDDEN_EVIDENCE_MARKERS
        if marker.lower() in lowered
    ]
    if WINDOWS_PATH.search(value):
        hits.append("absolute_windows_path")
    return not hits, hits


def decision_reasoning(items: list[str]) -> str:
    descriptions = []
    for item in items:
        node, cause = item.split(";", maxsplit=1)
        descriptions.append(f"{node} 的“{cause}”")
    if len(descriptions) == 1:
        return (
            f"当前证据已经把能直接解释症状的异常收敛到{descriptions[0]}。"
            "其他观察未形成独立根因，因此按最小集合原则只保留这一项。"
        )
    joined = "、".join(descriptions)
    return (
        f"当前证据已经把能直接解释症状的异常收敛到{joined}。"
        "这些异常共同构成最小根因集合，未保留缺少独立证据的候选。"
    )


def build_user_prompt(source_record: Mapping[str, Any], evidence: str) -> str:
    question = source_record.get("question")
    output_format = source_record.get("output_format")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Source record is missing question text")
    if not isinstance(output_format, str) or not output_format.strip():
        raise ValueError("Source record is missing output format")
    return (
        "根据题目描述回答问题：\n\n"
        f"## 题目\n\n{question.strip()}\n\n"
        f"## 答案格式约束\n\n{output_format.strip()}\n\n"
        f"## 当前任务阶段\n\n{DECISION_STAGE}\n\n"
        f"## 当前已知证据\n\n{evidence.strip()}"
    )


def raw_path(raw_dir: Path, case_id: int, repeat_index: int) -> Path:
    return (
        raw_dir
        / f"q{case_id:04d}"
        / f"run_{repeat_index:02d}"
        / "conversation_trajectory.json"
    )


def output_metadata(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": label(path),
        "samples": len(rows),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def main() -> None:
    options = parse_args()
    run_dir = (
        options.run_dir.resolve()
        if options.run_dir
        else discover_latest_complete_run(RUNS_ROOT)
    )
    output_root = options.output_root.resolve()
    raw_dir = output_root / "raw"
    curation_dir = output_root / "curation"
    sft_dir = output_root / "sft"
    curation_path = curation_dir / CURATION_FILE.name
    train_path = sft_dir / TRAIN_FILE.name
    validation_path = sft_dir / VALIDATION_FILE.name
    manifest_path = sft_dir / MANIFEST_FILE.name

    source_manifest_path = run_dir / "manifest.json"
    manifest = load_json(source_manifest_path)
    runs = manifest.get("runs")
    if (
        manifest.get("status") != "succeeded"
        or not isinstance(runs, list)
        or not runs
        or any(
            not isinstance(run, dict) or run.get("status") != "succeeded"
            for run in runs
        )
    ):
        raise ValueError(f"Run is not complete and successful: {run_dir}")

    case_ids = sorted(
        {
            int(run["case_id"])
            for run in runs
            if isinstance(run, dict) and isinstance(run.get("case_id"), int)
        }
    )
    if not case_ids:
        raise ValueError("Source manifest does not contain case ids")
    excluded_case_ids = sorted(
        set(
            options.excluded_case_ids
            if options.excluded_case_ids is not None
            else DEFAULT_EXCLUDED_CASE_IDS
        )
    )
    unknown_excluded = sorted(set(excluded_case_ids) - set(case_ids))
    if unknown_excluded:
        raise ValueError(
            f"Excluded cases are not present in the source run: {unknown_excluded}"
        )
    user_eligible_case_ids = sorted(set(case_ids) - set(excluded_case_ids))
    if len(user_eligible_case_ids) < 2:
        raise ValueError("At least two eligible cases are required for a split")
    validation_case_id = (
        options.validation_case_id
        if options.validation_case_id is not None
        else max(user_eligible_case_ids)
    )
    if validation_case_id not in user_eligible_case_ids:
        raise ValueError(
            f"Validation case {validation_case_id} is not eligible; "
            f"eligible cases are {user_eligible_case_ids}"
        )

    processed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    ordered_runs = sorted(
        runs, key=lambda run: (int(run["case_id"]), int(run["repeat_index"]))
    )
    for run in ordered_runs:
        case_id = int(run["case_id"])
        repeat_index = int(run["repeat_index"])
        source_id = f"q{case_id:04d}_run_{repeat_index:02d}"
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source trajectory {source_id}")
        seen_ids.add(source_id)

        relative_run_dir = run.get("directory")
        successful_attempt = run.get("successful_attempt")
        if not isinstance(relative_run_dir, str) or not isinstance(
            successful_attempt, int
        ):
            raise ValueError(f"{source_id}: malformed run manifest entry")
        trajectory_dir = run_dir / relative_run_dir
        attempt_dir = trajectory_dir / f"attempt_{successful_attempt:03d}"
        source_record_path = trajectory_dir / "source_record.json"
        run_json_path = trajectory_dir / "run.json"
        events_path = attempt_dir / "events.jsonl"
        final_answer_path = attempt_dir / "final_answer.txt"
        required = (
            source_record_path,
            run_json_path,
            events_path,
            final_answer_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"{source_id}: missing source files: {missing}")

        source_record = load_json(source_record_path)
        if int(source_record.get("id", -1)) != case_id:
            raise ValueError(f"{source_id}: source record id does not match")
        agent_messages = load_agent_messages(events_path)
        final_answer = normalize_text(
            final_answer_path.read_text(encoding="utf-8")
        )
        expected_answer = normalize_text(str(source_record.get("answer", "")))
        actual_items = result_items(final_answer)
        expected_items = result_items(expected_answer)
        final_message_matches = (
            bool(agent_messages)
            and agent_messages[-1]["text"] == final_answer
        )
        has_evidence_message = len(agent_messages) >= 2
        evidence = (
            agent_messages[-2]["text"] if has_evidence_message else ""
        )
        clean_evidence, evidence_marker_hits = evidence_is_clean(evidence)
        exact_answer_match = bool(expected_items) and actual_items == expected_items

        exclusion_reasons: list[str] = []
        if case_id in excluded_case_ids:
            exclusion_reasons.append("case_excluded_by_user")
        if not exact_answer_match:
            exclusion_reasons.append("final_answer_differs_from_reference")
        if not final_message_matches:
            exclusion_reasons.append("final_event_differs_from_saved_answer")
        if not has_evidence_message:
            exclusion_reasons.append("missing_pre_final_evidence_message")
        if not clean_evidence:
            exclusion_reasons.append("evidence_contains_operation_markers")
        selected = not exclusion_reasons
        split = (
            "validation"
            if selected and case_id == validation_case_id
            else "train"
            if selected
            else "excluded"
        )

        raw_file = raw_path(raw_dir, case_id, repeat_index)
        successful_attempt_data = next(
            (
                attempt
                for attempt in run.get("attempts", [])
                if isinstance(attempt, dict)
                and attempt.get("attempt_index") == successful_attempt
            ),
            {},
        )
        raw_document = {
            "schema_version": "codex-ip-normalized-trajectory.v1",
            "id": source_id,
            "case_id": case_id,
            "repeat_index": repeat_index,
            "source_record": source_record,
            "agent_messages": agent_messages,
            "final_answer": final_answer,
            "expected_result_items": expected_items,
            "actual_result_items": actual_items,
            "answer_matches_reference": exact_answer_match,
            "source": {
                "experiment_run": label(run_dir),
                "run_directory": label(trajectory_dir),
                "run_json": label(run_json_path),
                "run_json_sha256": digest(run_json_path),
                "events_file": label(events_path),
                "events_sha256": digest(events_path),
                "final_answer_file": label(final_answer_path),
                "final_answer_sha256": digest(final_answer_path),
                "source_record_file": label(source_record_path),
                "source_record_sha256": digest(source_record_path),
                "successful_attempt": successful_attempt,
                "thread_id": successful_attempt_data.get("thread_id"),
                "duration_seconds": successful_attempt_data.get(
                    "duration_seconds"
                ),
            },
        }
        write_json(raw_file, raw_document)
        processed.append(
            {
                "id": source_id,
                "case_id": case_id,
                "repeat_index": repeat_index,
                "selected": selected,
                "split": split,
                "review_status": "draft",
                "selection_reason": (
                    "accepted_exact_reference_answer_and_clean_evidence"
                    if selected
                    else "excluded"
                ),
                "exclusion_reasons": exclusion_reasons,
                "evidence_marker_hits": evidence_marker_hits,
                "expected_result_items": expected_items,
                "actual_result_items": actual_items,
                "source_answer_format_normalized": (
                    exact_answer_match
                    and final_answer != canonical_result(expected_items)
                ),
                "evidence": evidence if selected else None,
                "evidence_message_index": (
                    len(agent_messages) - 2 if has_evidence_message else None
                ),
                "final_message_index": (
                    len(agent_messages) - 1 if agent_messages else None
                ),
                "raw_file": label(raw_file),
                "raw_sha256": digest(raw_file),
                "events_file": label(events_path),
                "events_sha256": digest(events_path),
            }
        )

    quality_excluded_case_ids = sorted(
        {
            item["case_id"]
            for item in processed
            if item["case_id"] not in excluded_case_ids
            and item["exclusion_reasons"]
        }
    )
    for item in processed:
        if item["case_id"] not in quality_excluded_case_ids:
            continue
        if "case_not_100_percent_eligible" not in item["exclusion_reasons"]:
            item["exclusion_reasons"].append("case_not_100_percent_eligible")
        item["selected"] = False
        item["split"] = "excluded"
        item["selection_reason"] = "excluded"
        item["evidence"] = None
    eligible_case_ids = sorted(
        set(case_ids) - set(excluded_case_ids) - set(quality_excluded_case_ids)
    )
    if validation_case_id not in eligible_case_ids:
        raise ValueError(
            f"Validation case {validation_case_id} did not achieve 100% "
            "trajectory eligibility"
        )

    split_counts = Counter(item["split"] for item in processed)
    exclusion_counts = Counter(
        reason
        for item in processed
        for reason in item["exclusion_reasons"]
    )
    selected_case_ids = {
        item["case_id"] for item in processed if item["selected"]
    }
    train_case_ids = sorted(
        {
            item["case_id"]
            for item in processed
            if item["split"] == "train"
        }
    )
    validation_case_ids = sorted(
        {
            item["case_id"]
            for item in processed
            if item["split"] == "validation"
        }
    )
    if validation_case_ids != [validation_case_id]:
        raise ValueError(
            f"Validation case {validation_case_id} has no accepted trajectories"
        )
    if set(train_case_ids) & set(validation_case_ids):
        raise ValueError("Train and validation case ids overlap")
    case_quality = []
    for case_id in case_ids:
        case_rows = [
            item for item in processed if item["case_id"] == case_id
        ]
        exact_answers = sum(
            "final_answer_differs_from_reference"
            not in item["exclusion_reasons"]
            for item in case_rows
        )
        selected_rows = [item for item in case_rows if item["selected"]]
        selected_split = (
            selected_rows[0]["split"] if selected_rows else "excluded"
        )
        case_quality.append(
            {
                "case_id": case_id,
                "trajectories": len(case_rows),
                "exact_reference_answers": exact_answers,
                "accuracy": round(exact_answers / len(case_rows), 6),
                "user_excluded": case_id in excluded_case_ids,
                "quality_excluded": case_id in quality_excluded_case_ids,
                "selected_trajectories": len(selected_rows),
                "split": selected_split,
            }
        )

    curation_document = {
        "schema_version": "codex-ip-trajectory-curation.v1",
        "source_run": label(run_dir),
        "source_manifest": label(source_manifest_path),
        "source_manifest_sha256": digest(source_manifest_path),
        "selection": {
            "required_run_status": "succeeded",
            "excluded_case_ids": excluded_case_ids,
            "quality_excluded_case_ids": quality_excluded_case_ids,
            "required_case_eligibility_rate": 1.0,
            "answer_filter": "exact_ordered_result_items_match_reference",
            "require_final_event_match": True,
            "require_clean_pre_final_evidence": True,
            "review_status": "draft",
        },
        "split": {
            "strategy": "leave_one_case_out",
            "group_key": "case_id",
            "validation_case_ids": validation_case_ids,
            "train_case_ids": train_case_ids,
            "case_groups_disjoint": True,
        },
        "counts": {
            "raw": len(processed),
            "selected": split_counts["train"] + split_counts["validation"],
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "excluded": split_counts["excluded"],
        },
        "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        "source_case_ids": case_ids,
        "eligible_case_ids": eligible_case_ids,
        "selected_case_ids": sorted(selected_case_ids),
        "case_quality": case_quality,
        "trajectories": processed,
    }
    write_json(curation_path, curation_document)
    curation_sha256 = digest(curation_path)

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for item in processed:
        if not item["selected"]:
            continue
        raw_file = ROOT / item["raw_file"]
        raw_document = load_json(raw_file)
        source_record = raw_document["source_record"]
        expected_items = item["expected_result_items"]
        answer = canonical_result(expected_items)
        row = {
            "id": f"{item['id']}_decision",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        source_record, str(item["evidence"])
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"<think>\n{decision_reasoning(expected_items)}\n"
                        f"</think>\n\n{answer}"
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
                "repeat_index": item["repeat_index"],
                "source_file": item["raw_file"],
                "source_sha256": item["raw_sha256"],
                "source_event_file": item["events_file"],
                "source_event_sha256": item["events_sha256"],
                "annotation_file": label(curation_path),
                "annotation_sha256": curation_sha256,
                "source_message_index": item["final_message_index"],
                "evidence_message_indices": [item["evidence_message_index"]],
                "evidence_count": 1,
                "expected_result_items": expected_items,
                "reference_answer_match": True,
                "source_answer_format_normalized": item[
                    "source_answer_format_normalized"
                ],
            },
        }
        if item["split"] == "validation":
            validation_rows.append(row)
        else:
            train_rows.append(row)

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    output_manifest = {
        "schema_version": "qwen36-reasoning-decision-sft.v3",
        "source_run": label(run_dir),
        "source_manifest": label(source_manifest_path),
        "source_manifest_sha256": digest(source_manifest_path),
        "raw_trajectory_count": len(processed),
        "selected_trajectory_count": len(train_rows) + len(validation_rows),
        "excluded_trajectory_count": split_counts["excluded"],
        "curation_file": label(curation_path),
        "curation_sha256": curation_sha256,
        "target_type_counts": {
            "train": {"decision": len(train_rows)},
            "validation": {"decision": len(validation_rows)},
        },
        "split": {
            "strategy": "leave_one_case_out",
            "group_key": "case_id",
            "train": len(train_rows),
            "validation": len(validation_rows),
            "train_case_ids": train_case_ids,
            "validation_case_ids": validation_case_ids,
            "case_groups_disjoint": True,
        },
        "selection": {
            "included": len(train_rows) + len(validation_rows),
            "excluded": split_counts["excluded"],
            "excluded_case_ids": excluded_case_ids,
            "quality_excluded_case_ids": quality_excluded_case_ids,
            "required_case_eligibility_rate": 1.0,
            "case_accuracy": {
                str(item["case_id"]): item["accuracy"]
                for item in case_quality
            },
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        },
        "outputs": [
            output_metadata(train_path, train_rows),
            output_metadata(validation_path, validation_rows),
        ],
    }
    write_json(manifest_path, output_manifest)

    print(
        f"Normalized {len(processed)} trajectories from {run_dir.name}"
    )
    print(
        f"- selected exact-answer trajectories: "
        f"{len(train_rows) + len(validation_rows)}"
    )
    print(f"- user-excluded cases: {excluded_case_ids}")
    print(f"- quality-excluded cases: {quality_excluded_case_ids}")
    print(f"- excluded trajectories: {split_counts['excluded']}")
    print(f"- train: {len(train_rows)} across {len(train_case_ids)} cases")
    print(
        f"- validation: {len(validation_rows)} from case "
        f"{validation_case_id}"
    )


if __name__ == "__main__":
    main()
