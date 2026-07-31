#!/usr/bin/env python3
"""Filter independently correct 100x10 trajectories and build grouped SFT splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    ROOT
    / "experiments"
    / "2026-07-28-ip_codex_train0629_100x10"
)
REPORT_DIR = EXPERIMENT_ROOT / "results" / "report"
RUNS_DIR = EXPERIMENT_ROOT / "results" / "runs"
DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "2026-07-31"
ACCEPTED_INDEX = REPORT_DIR / "accepted_index.json"
FINAL_AUDIT = REPORT_DIR / "final_audit.json"
STATE_FILE = REPORT_DIR / "state.json"
CURATION_NAME = "accepted_trajectory_selection.json"
FILTER_REPORT_NAME = "FILTER_REPORT.md"
TRAIN_NAME = "qwen3_6_27b_reasoning_decision_train.jsonl"
VALIDATION_NAME = "qwen3_6_27b_reasoning_decision_validation.jsonl"
MANIFEST_NAME = "manifest.json"
ATTEMPT_STATUSES = {
    "accepted",
    "rejected",
    "interrupted",
    "infrastructure_failure",
}

SYSTEM = (
    "你是一名网络故障分析专家。请根据题目和当前已知证据逐步分析。"
    "信息不足时，说明下一步需要核验的事实以及核验目的；证据充分时，比较候选根因并作出决策。"
    "不得补充题目未提供的事实。先在 <think>...</think> 中给出简洁、可复核的思考，"
    "再输出当前计划、阶段判断或最终结论。"
)
DECISION_STAGE = (
    "当前证据已完成收敛。请给出最小根因集合，并严格遵守题目要求的答案格式。"
)
RESULT_RE = re.compile(r"<result>\s*([\s\S]*?)\s*</result>")
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
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="100x10 experiment directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET,
        help="Immutable source JSONL used only for independent answer checks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Date-scoped output data directory.",
    )
    parser.add_argument(
        "--validation-case-id",
        type=int,
        help="Held-out case id. Defaults to the highest eligible case id.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def stable_digest(path: Path) -> str:
    return hashlib.sha256(normalized_bytes(path)).hexdigest()


def label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


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


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def format_duration(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def parse_result(text: str) -> list[str] | None:
    matches = RESULT_RE.findall(normalize_text(text))
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return sorted(set(value))


def reference_options(answer_text: str) -> list[list[str]]:
    value = json.loads(answer_text)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(option, list) for option in value)
        and all(all(isinstance(item, str) for item in option) for option in value)
    ):
        return [sorted(set(option)) for option in value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("source answer must be a JSON array of strings or alternatives")
    return [sorted(set(value))]


def canonical_result(items: list[str]) -> str:
    lines = [
        f'"{item}",' if index < len(items) - 1 else f'"{item}"'
        for index, item in enumerate(items)
    ]
    return "\n".join(["<result>", "[", *lines, "]", "</result>"])


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
        raise ValueError("source record is missing question")
    if not isinstance(output_format, str) or not output_format.strip():
        raise ValueError("source record is missing output_format")
    return (
        "根据题目描述回答问题：\n\n"
        f"## 题目\n\n{question.strip()}\n\n"
        f"## 答案格式约束\n\n{output_format.strip()}\n\n"
        f"## 当前任务阶段\n\n{DECISION_STAGE}\n\n"
        f"## 当前已知证据\n\n{evidence.strip()}"
    )


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


def output_metadata(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    content = normalized_bytes(path)
    return {
        "path": label(path),
        "samples": len(rows),
        "normalized_bytes": len(content),
        "sha256_lf_normalized": hashlib.sha256(content).hexdigest(),
    }


def raw_path(raw_dir: Path, case_id: int, success_slot: int) -> Path:
    return (
        raw_dir
        / f"q{case_id:04d}"
        / f"run_{success_slot:02d}"
        / "conversation_trajectory.json"
    )


def candidate_reasons(
    *,
    attempt_dir: Path,
    index_item: Mapping[str, Any],
    sample: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    required_names = (
        "metadata.json",
        "judgment.json",
        "source_record.json",
        "events.jsonl",
        "final_answer.txt",
    )
    missing = [name for name in required_names if not (attempt_dir / name).is_file()]
    if missing:
        return [f"missing_artifact:{name}" for name in missing], {}

    metadata = load_json(attempt_dir / "metadata.json")
    judgment = load_json(attempt_dir / "judgment.json")
    safe_record = load_json(attempt_dir / "source_record.json")
    final_text = normalize_text(
        (attempt_dir / "final_answer.txt").read_text(encoding="utf-8")
    )
    agent_messages = load_agent_messages(attempt_dir / "events.jsonl")
    prediction = parse_result(final_text)
    references = reference_options(str(source_row.get("answer", "")))
    reasons: list[str] = []

    if metadata.get("status") != "accepted":
        reasons.append("metadata_not_accepted")
    if metadata.get("generation_status") != "completed":
        reasons.append("generation_not_completed")
    if metadata.get("model") != "gpt-5.6-sol":
        reasons.append("unexpected_model")
    if metadata.get("attempt_index") != index_item.get("attempt_index"):
        reasons.append("attempt_index_mismatch")
    if metadata.get("thread_id") != index_item.get("thread_id"):
        reasons.append("thread_id_mismatch")
    if (
        judgment.get("judge_status") != "completed"
        or judgment.get("parsed") is not True
        or judgment.get("correct") is not True
    ):
        reasons.append("judgment_not_strictly_correct")

    final_path = attempt_dir / "final_answer.txt"
    final_digest = stable_digest(final_path)
    if judgment.get("final_answer_sha256") != final_digest:
        reasons.append("judgment_final_hash_mismatch")
    if metadata.get("sha256", {}).get("final_answer") != final_digest:
        reasons.append("metadata_final_hash_mismatch")
    if metadata.get("sha256", {}).get("events") != stable_digest(
        attempt_dir / "events.jsonl"
    ):
        reasons.append("metadata_events_hash_mismatch")
    if metadata.get("sha256", {}).get("source_record") != stable_digest(
        attempt_dir / "source_record.json"
    ):
        reasons.append("metadata_source_record_hash_mismatch")
    if prediction is None or prediction not in references:
        reasons.append("independent_exact_reference_match_failed")
    if not agent_messages or agent_messages[-1]["text"] != final_text:
        reasons.append("final_event_differs_from_saved_answer")
    if len(agent_messages) < 2:
        reasons.append("missing_pre_final_evidence_message")

    evidence = agent_messages[-2]["text"] if len(agent_messages) >= 2 else ""
    clean_evidence, evidence_hits = evidence_is_clean(evidence)
    if not clean_evidence:
        reasons.append("evidence_contains_operation_markers")

    expected_original_id = str(sample.get("original_id"))
    if (
        str(safe_record.get("original_id")) != expected_original_id
        or str(source_row.get("id")) != expected_original_id
        or safe_record.get("row_index") != sample.get("row_index")
    ):
        reasons.append("source_identity_mismatch")
    if (
        safe_record.get("question") != source_row.get("question")
        or safe_record.get("output_format") != source_row.get("output_format")
    ):
        reasons.append("safe_source_content_mismatch")
    if "answer" in safe_record or safe_record.get("contains_ground_answer") is not False:
        reasons.append("safe_source_contains_ground_answer")

    return reasons, {
        "metadata": metadata,
        "judgment": judgment,
        "source_record": safe_record,
        "final_text": final_text,
        "prediction": prediction,
        "reference_options": references,
        "agent_messages": agent_messages,
        "evidence": evidence,
        "evidence_hits": sorted(set(evidence_hits)),
    }


def main() -> None:
    options = parse_args()
    experiment_root = options.experiment_root.resolve()
    report_dir = experiment_root / "results" / "report"
    runs_dir = experiment_root / "results" / "runs"
    accepted_index_path = report_dir / ACCEPTED_INDEX.name
    final_audit_path = report_dir / FINAL_AUDIT.name
    state_path = report_dir / STATE_FILE.name
    dataset_path = options.dataset.resolve()
    output_root = options.output_root.resolve()
    raw_dir = output_root / "raw"
    curation_dir = output_root / "curation"
    sft_dir = output_root / "sft"
    curation_path = curation_dir / CURATION_NAME
    filter_report_path = curation_dir / FILTER_REPORT_NAME
    train_path = sft_dir / TRAIN_NAME
    validation_path = sft_dir / VALIDATION_NAME
    manifest_path = sft_dir / MANIFEST_NAME

    accepted_index = load_json(accepted_index_path)
    final_audit = load_json(final_audit_path)
    state = load_json(state_path)
    source_rows = load_jsonl(dataset_path)
    source_by_index = {
        index: row for index, row in enumerate(source_rows, start=1)
    }
    if final_audit.get("passed") is not True:
        raise ValueError("source experiment final audit did not pass")
    if final_audit.get("accepted_total") != sum(
        int(sample.get("accepted_count", 0))
        for sample in accepted_index.get("samples", [])
    ):
        raise ValueError("accepted index count does not match final audit")
    if state.get("status") != "completed":
        raise ValueError("source experiment state is not completed")

    attempt_status_counts: Counter[str] = Counter()
    attempt_status_counts_by_case: defaultdict[int, Counter[str]] = defaultdict(
        Counter
    )
    attempt_durations_by_case: defaultdict[int, list[float]] = defaultdict(list)
    successful_attempt_durations_by_case: defaultdict[int, list[float]] = (
        defaultdict(list)
    )
    for metadata_path in runs_dir.glob("q*_r*/attempt_*/metadata.json"):
        metadata = load_json(metadata_path)
        run_key = metadata_path.relative_to(runs_dir).parts[0]
        run_match = re.fullmatch(r"q(\d+)_r\d+", run_key)
        if run_match is None:
            raise ValueError(f"cannot determine case id from {metadata_path}")
        case_id = int(run_match.group(1))
        status = str(metadata.get("status", "unknown"))
        if status not in ATTEMPT_STATUSES:
            raise ValueError(f"{metadata_path}: unexpected attempt status {status!r}")
        if int(metadata.get("original_id", -1)) != case_id:
            raise ValueError(
                f"{metadata_path}: original_id does not match its case directory"
            )
        metadata_row_index = int(metadata.get("row_index", -1))
        metadata_source_row = source_by_index.get(metadata_row_index)
        if (
            metadata_source_row is None
            or int(metadata_source_row.get("id", -1)) != case_id
        ):
            raise ValueError(
                f"{metadata_path}: row_index does not resolve to its case id"
            )
        attempt_status_counts[status] += 1
        attempt_status_counts_by_case[case_id][status] += 1
        duration = metadata.get("duration_seconds")
        if duration is None:
            continue
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
            or not math.isfinite(duration)
        ):
            raise ValueError(
                f"{metadata_path}: duration_seconds must be non-negative or null"
            )
        attempt_durations_by_case[case_id].append(float(duration))
        if status == "accepted":
            successful_attempt_durations_by_case[case_id].append(float(duration))

    processed: list[dict[str, Any]] = []
    seen_attempt_paths: set[str] = set()
    seen_event_paths: set[str] = set()
    seen_thread_ids: set[str] = set()
    samples = accepted_index.get("samples")
    if not isinstance(samples, list):
        raise ValueError("accepted index samples are missing")

    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("accepted index sample is malformed")
        row_index = int(sample["row_index"])
        case_id = int(sample["original_id"])
        source_row = source_by_index.get(row_index)
        if source_row is None:
            raise ValueError(f"row {row_index}: source record is missing")
        mapping = sample.get("mapping")
        if not isinstance(mapping, dict):
            raise ValueError(f"row {row_index}: accepted mapping is malformed")
        expected_keys = [
            f"success_{index:02d}"
            for index in range(1, int(sample["accepted_count"]) + 1)
        ]
        if list(mapping) != expected_keys:
            raise ValueError(f"row {row_index}: success slots are not sequential")

        for success_key, index_item in mapping.items():
            if not isinstance(index_item, dict):
                raise ValueError(f"row {row_index} {success_key}: malformed index item")
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

            attempt_dir = experiment_root / attempt_relative
            reasons, details = candidate_reasons(
                attempt_dir=attempt_dir,
                index_item=index_item,
                sample=sample,
                source_row=source_row,
            )
            success_slot = int(index_item["success_slot"])
            source_id = f"q{case_id:04d}_success_{success_slot:02d}"
            raw_file = raw_path(raw_dir, case_id, success_slot)
            selected = not reasons

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
                    "source_record_file": label(attempt_dir / "source_record.json"),
                    "source_record_sha256_lf_normalized": (
                        stable_digest(attempt_dir / "source_record.json")
                        if (attempt_dir / "source_record.json").is_file()
                        else None
                    ),
                    "thread_id": thread_id,
                    "duration_seconds": details.get("metadata", {}).get(
                        "duration_seconds"
                    ),
                },
            }
            write_json(raw_file, raw_document)
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
                    "reference_answer_options": details.get(
                        "reference_options", []
                    ),
                    "source_answer_format_normalized": (
                        selected
                        and details.get("final_text")
                        != canonical_result(details["prediction"])
                    ),
                    "raw_file": label(raw_file),
                    "raw_sha256_lf_normalized": stable_digest(raw_file),
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

    eligible_case_ids = sorted(
        {item["case_id"] for item in processed if item["selected"]}
    )
    if len(eligible_case_ids) < 2:
        raise ValueError("at least two cases with eligible trajectories are required")
    validation_case_id = (
        int(options.validation_case_id)
        if options.validation_case_id is not None
        else max(eligible_case_ids)
    )
    if validation_case_id not in eligible_case_ids:
        raise ValueError(
            f"validation case {validation_case_id} has no eligible trajectory"
        )
    for item in processed:
        if not item["selected"]:
            continue
        item["split"] = (
            "validation"
            if item["case_id"] == validation_case_id
            else "train"
        )

    train_case_ids = sorted(
        {item["case_id"] for item in processed if item["split"] == "train"}
    )
    validation_case_ids = sorted(
        {
            item["case_id"]
            for item in processed
            if item["split"] == "validation"
        }
    )
    if validation_case_ids != [validation_case_id]:
        raise ValueError("validation split must contain exactly one case group")
    if set(train_case_ids) & set(validation_case_ids):
        raise ValueError("train and validation case ids overlap")

    split_counts = Counter(item["split"] for item in processed)
    exclusion_counts = Counter(
        reason
        for item in processed
        for reason in item["exclusion_reasons"]
    )
    state_by_row = {
        int(item["row_index"]): item
        for item in state.get("samples", [])
        if isinstance(item, dict)
    }
    case_quality = []
    case_report_rows = []
    for row_index, source_row in source_by_index.items():
        case_id = int(source_row["id"])
        case_items = [
            item for item in processed if item["row_index"] == row_index
        ]
        selected_items = [item for item in case_items if item["selected"]]
        state_item = state_by_row.get(row_index, {})
        status_counts = attempt_status_counts_by_case[case_id]
        source_attempts = sum(status_counts.values())
        state_attempts = int(state_item.get("total_attempts", 0))
        if source_attempts != state_attempts:
            raise ValueError(
                f"case {case_id}: metadata attempt count {source_attempts} "
                f"does not match state count {state_attempts}"
            )
        successful_attempts = status_counts["accepted"]
        if successful_attempts != len(case_items):
            raise ValueError(
                f"case {case_id}: accepted metadata count {successful_attempts} "
                f"does not match accepted index count {len(case_items)}"
            )
        durations = attempt_durations_by_case[case_id]
        successful_durations = successful_attempt_durations_by_case[case_id]
        case_quality.append(
            {
                "case_id": case_id,
                "row_index": row_index,
                "source_attempts": source_attempts,
                "accepted_candidates": len(case_items),
                "selected_trajectories": len(selected_items),
                "terminal_status": state_item.get("status"),
                "split": (
                    selected_items[0]["split"]
                    if selected_items
                    else "excluded"
                ),
            }
        )
        case_report_rows.append(
            {
                "case_id": case_id,
                "attempts": source_attempts,
                "successes": successful_attempts,
                "duration_count": len(durations),
                "average_duration_seconds": (
                    sum(durations) / len(durations) if durations else None
                ),
                "successful_average_duration_seconds": (
                    sum(successful_durations) / len(successful_durations)
                    if successful_durations
                    else None
                ),
                "rejected": status_counts["rejected"],
                "interrupted": status_counts["interrupted"],
                "infrastructure_failure": status_counts[
                    "infrastructure_failure"
                ],
                "selected": len(selected_items),
                "terminal_status": state_item.get("status"),
                "split": (
                    selected_items[0]["split"]
                    if selected_items
                    else "excluded"
                ),
            }
        )

    curation_document = {
        "schema_version": "codex-ip-accepted-trajectory-curation.v2",
        "source_experiment": label(experiment_root),
        "source_dataset": label(dataset_path),
        "source_dataset_sha256_lf_normalized": stable_digest(dataset_path),
        "source_accepted_index": label(accepted_index_path),
        "source_accepted_index_sha256_lf_normalized": stable_digest(
            accepted_index_path
        ),
        "source_final_audit": label(final_audit_path),
        "source_final_audit_sha256_lf_normalized": stable_digest(
            final_audit_path
        ),
        "selection": {
            "required_source_audit_passed": True,
            "required_metadata_status": "accepted",
            "required_judgment_correct": True,
            "answer_filter": "independent_exact_fault_set_match_with_alternatives",
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
            "source_records": len(source_rows),
            "source_attempts": sum(attempt_status_counts.values()),
            "accepted_candidates": len(processed),
            "selected": split_counts["train"] + split_counts["validation"],
            "train": split_counts["train"],
            "validation": split_counts["validation"],
            "excluded_candidates": split_counts["excluded"],
            "filtered_nonaccepted_attempts": (
                sum(attempt_status_counts.values()) - len(processed)
            ),
        },
        "source_attempt_status_counts": dict(sorted(attempt_status_counts.items())),
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
        raw_document = load_json(ROOT / item["raw_file"])
        source_record = raw_document["source_record"]
        actual_items = item["actual_result_items"]
        answer = canonical_result(actual_items)
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
                        f"<think>\n{decision_reasoning(actual_items)}\n"
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
        "schema_version": "qwen36-reasoning-decision-sft.v4",
        "source_experiment": label(experiment_root),
        "source_dataset": label(dataset_path),
        "source_dataset_sha256_lf_normalized": stable_digest(dataset_path),
        "source_accepted_index": label(accepted_index_path),
        "source_accepted_index_sha256_lf_normalized": stable_digest(
            accepted_index_path
        ),
        "source_final_audit": label(final_audit_path),
        "source_final_audit_sha256_lf_normalized": stable_digest(
            final_audit_path
        ),
        "source_attempt_count": sum(attempt_status_counts.values()),
        "accepted_candidate_count": len(processed),
        "selected_trajectory_count": len(train_rows) + len(validation_rows),
        "excluded_candidate_count": split_counts["excluded"],
        "filtered_nonaccepted_attempt_count": (
            sum(attempt_status_counts.values()) - len(processed)
        ),
        "curation_file": label(curation_path),
        "curation_sha256_lf_normalized": curation_digest,
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
            "excluded_candidates": split_counts["excluded"],
            "filtered_nonaccepted_attempts": (
                sum(attempt_status_counts.values()) - len(processed)
            ),
            "candidate_exclusion_reason_counts": dict(
                sorted(exclusion_counts.items())
            ),
            "source_attempt_status_counts": dict(
                sorted(attempt_status_counts.items())
            ),
        },
        "outputs": [
            output_metadata(train_path, train_rows),
            output_metadata(validation_path, validation_rows),
        ],
    }
    write_json(manifest_path, output_manifest)

    all_attempt_durations = [
        duration
        for durations in attempt_durations_by_case.values()
        for duration in durations
    ]
    average_attempt_duration = (
        sum(all_attempt_durations) / len(all_attempt_durations)
        if all_attempt_durations
        else None
    )
    all_successful_attempt_durations = [
        duration
        for durations in successful_attempt_durations_by_case.values()
        for duration in durations
    ]
    average_successful_attempt_duration = (
        sum(all_successful_attempt_durations)
        / len(all_successful_attempt_durations)
        if all_successful_attempt_durations
        else None
    )
    success_count_distribution = Counter(
        item["successes"] for item in case_report_rows
    )
    label_case_ids: defaultdict[str, set[int]] = defaultdict(set)
    label_trajectory_counts: Counter[str] = Counter()
    for item in processed:
        if not item["selected"]:
            continue
        for answer_label in item["actual_result_items"]:
            label_case_ids[answer_label].add(item["case_id"])
            label_trajectory_counts[answer_label] += 1
    report_lines = [
        "# 100×10 完全正确轨迹过滤与 SFT 转换报告",
        "",
        f"- 来源实验：`{label(experiment_root)}`",
        f"- 来源 attempt：{sum(attempt_status_counts.values())}",
        f"- accepted 且独立判题完全正确的候选：{len(processed)}",
        f"- 通过 SFT 完整性与证据清洁检查：{len(train_rows) + len(validation_rows)}",
        f"- 候选中排除：{split_counts['excluded']}",
        f"- 非 accepted attempt 过滤：{sum(attempt_status_counts.values()) - len(processed)}",
        f"- 训练集：{len(train_rows)} 条，{len(train_case_ids)} 个题号",
        f"- 验证集：{len(validation_rows)} 条，题号 {validation_case_id}",
        "- 训练/验证题号交集：0",
        (
            f"- 全部 attempt 平均耗时：{average_attempt_duration:.3f} 秒"
            f"（{len(all_attempt_durations)}/"
            f"{sum(attempt_status_counts.values())} 条有有效耗时）"
            if average_attempt_duration is not None
            else "- 全部 attempt 平均耗时：无有效耗时"
        ),
        (
            f"- 成功 attempt 平均耗时："
            f"{average_successful_attempt_duration:.3f} 秒"
            f"（{len(all_successful_attempt_durations)}/"
            f"{attempt_status_counts['accepted']} 条有有效耗时）"
            if average_successful_attempt_duration is not None
            else "- 成功 attempt 平均耗时：无有效耗时"
        ),
        "",
        "## 统计口径",
        "",
        "- `Attempt`：题号目录下存在 `metadata.json` 的独立尝试；"
        "全局及逐题数量均与实验 `state.json` 复核一致。",
        "- `成功`：来源 attempt 状态为 `accepted`；`SFT` 列才表示进入候选后继续"
        "通过独立判题、参考答案严格集合匹配、最终事件一致性、文件哈希和"
        "前置证据清洁检查的最终保留数。",
        "- `平均耗时`：该题所有 `duration_seconds` 非空且非负 attempt 的算术平均，"
        "包含 accepted、rejected 和 infrastructure failure；"
        "`成功平均耗时` 只统计 accepted attempt。",
        "- `duration_seconds` 由 runner 的单调时钟记录，覆盖 Codex 子进程执行及其"
        "输出实时落盘，不包含退出后的事件解析、审计整理和随后启动的独立判题，"
        "因此不是完整端到端耗时。",
        "- 缺失耗时的 interrupted attempt 不以 0 计入。",
        "- `有效耗时`：以 `有耗时记录数/Attempt` 展示平均值的实际分母；"
        "`SFT` 是最终保留并转换的轨迹数。",
        "",
        "## 来源 attempt 状态",
        "",
        "| 状态 | 数量 |",
        "| --- | ---: |",
        *[
            f"| {status} | {count} |"
            for status, count in sorted(attempt_status_counts.items())
        ],
        "",
        "## 按答案 label 统计题目与轨迹",
        "",
        "仅统计最终进入 SFT 的严格正确轨迹。`题目数量` 是包含该 label 的去重题号数，"
        "`轨迹数量` 是包含该 label 的保留轨迹数。多标签答案会分别计入各 label，"
        "因此各行不能直接相加；"
        f"去重总计为 {len(eligible_case_ids)} 题、"
        f"{len(train_rows) + len(validation_rows)} 条轨迹。",
        "",
        "| 答案 label | 题目数量 | 轨迹数量 |",
        "| --- | ---: | ---: |",
        *[
            f"| `{answer_label}` | {len(label_case_ids[answer_label])} | "
            f"{label_trajectory_counts[answer_label]} |"
            for answer_label in sorted(label_case_ids)
        ],
        (
            f"| **去重总计** | **{len(eligible_case_ids)}** | "
            f"**{len(train_rows) + len(validation_rows)}** |"
        ),
        "",
        "## 按成功次数统计题目数量",
        "",
        "| 每题成功次数 | 题目数量 | 成功轨迹小计 |",
        "| ---: | ---: | ---: |",
        *[
            f"| {successes} | {question_count} | "
            f"{successes * question_count} |"
            for successes, question_count in sorted(
                success_count_distribution.items()
            )
        ],
        (
            f"| **总计** | **{len(case_report_rows)}** | "
            f"**{attempt_status_counts['accepted']}** |"
        ),
        "",
        "## 逐题过滤统计",
        "",
        "| 题号 | Attempt | 成功 | 成功率 | 平均耗时（秒） | "
        "成功平均耗时（秒） | 有效耗时 | Rejected | Interrupted | "
        "Infrastructure failure | SFT | 划分 | 终态 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | --- | --- |",
        *[
            (
                f"| {item['case_id']} | {item['attempts']} | "
                f"{item['successes']} | "
                f"{item['successes'] / item['attempts']:.2%} | "
                f"{format_duration(item['average_duration_seconds'])} | "
                f"{format_duration(item['successful_average_duration_seconds'])} | "
                f"{item['duration_count']}/{item['attempts']} | "
                f"{item['rejected']} | {item['interrupted']} | "
                f"{item['infrastructure_failure']} | {item['selected']} | "
                f"{item['split']} | {item['terminal_status']} |"
            )
            for item in case_report_rows
        ],
        (
            f"| **总计** | **{sum(attempt_status_counts.values())}** | "
            f"**{attempt_status_counts['accepted']}** | "
            f"**{attempt_status_counts['accepted'] / sum(attempt_status_counts.values()):.2%}** | "
            f"**{format_duration(average_attempt_duration)}** | "
            f"**{format_duration(average_successful_attempt_duration)}** | "
            f"**{len(all_attempt_durations)}/"
            f"{sum(attempt_status_counts.values())}** | "
            f"**{attempt_status_counts['rejected']}** | "
            f"**{attempt_status_counts['interrupted']}** | "
            f"**{attempt_status_counts['infrastructure_failure']}** | "
            f"**{len(train_rows) + len(validation_rows)}** | — | — |"
        ),
        "",
        "## 候选排除原因",
        "",
        (
            f"无；{len(processed)} 条 accepted 候选全部通过独立答案复核、"
            "最终事件一致性、判题哈希和证据清洁检查。"
            if not exclusion_counts
            else "\n".join(
                f"- {reason}: {count}"
                for reason, count in sorted(exclusion_counts.items())
            )
        ),
    ]
    write_text(filter_report_path, "\n".join(report_lines))

    print(f"Source attempts: {sum(attempt_status_counts.values())}")
    print(f"Accepted candidates: {len(processed)}")
    print(f"Selected fully correct trajectories: {len(train_rows) + len(validation_rows)}")
    print(f"Excluded accepted candidates: {split_counts['excluded']}")
    print(f"Filtered non-accepted attempts: {sum(attempt_status_counts.values()) - len(processed)}")
    print(f"Train: {len(train_rows)} across {len(train_case_ids)} cases")
    print(f"Validation: {len(validation_rows)} from case {validation_case_id}")


if __name__ == "__main__":
    main()
