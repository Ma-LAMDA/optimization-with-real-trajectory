#!/usr/bin/env python3
"""Select one accepted trajectory per 0804 case and build weighted multi-turn SFT."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-04"
SOURCE_CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
BEST_SELECTION = DATA_ROOT / "curation" / "best_trajectory_per_case.json"
SFT_DIR = DATA_ROOT / "sft"
TRAIN_OUTPUT = SFT_DIR / "qwen3_6_27b_reasoning_trajectory_best1_train.jsonl"
VALIDATION_OUTPUT = SFT_DIR / "qwen3_6_27b_reasoning_trajectory_best1_validation.jsonl"
MANIFEST_OUTPUT = SFT_DIR / "reasoning_trajectory_best1_manifest.json"

THINKING_LOSS_SCALE = 0.4
TARGET_LOSS_SCALE = 1.0
HISTORY_LOSS_SCALE = 0.0
MAX_ACTIONS_PER_STAGE = 6
CURRENT_OUTPUT_LIMIT = 1800
HISTORICAL_OUTPUT_LIMIT = 600

SYSTEM = (
    "你是一名网络故障分析专家，正在沿一条逐步排障轨迹工作。只能使用题目、此前已经输出的"
    "分析和当前消息序列中已经返回的工具结果，不得使用未来命令、未来结果或隐藏答案。"
    "证据不足时继续选择最有区分度的实际取证动作；证据收敛后输出最小根因集合。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "读取归档网络设备配置和运行状态。",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]

FAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "全局STP未使能": ("全局stp", "stp enable", "protocol status", "stp未使能"),
    "STP BPDU被过滤": ("bpdu", "bpdu-filter", "bpdu过滤", "bpdu被过滤"),
    "存在IP路由环路": ("ip路由环路", "路由环路", "routing loop", "loop"),
    "存在MPLS标签环路": ("mpls标签环路", "标签环路", "mpls loop", "lsp loop"),
    "VRRP Master角色规划不合理": ("vrrp master", "master角色", "master规划"),
    "VRRP工作在非抢占模式": ("非抢占", "non-preempt", "preempt", "抢占模式"),
}

PROTOCOL_TERMS = (
    "arp", "bgp", "isis", "mpls", "lsp", "ldp", "ospf", "srv6", "stp",
    "mstp", "bpdu", "vrrp", "vpn", "route", "routing", "interface", "vlan",
    "eth-trunk", "loop", "preempt", "master", "backup", "cost", "10.",
)

HOUSEKEEPING_MARKERS = (
    "Get-ChildItem -Recurse",
    "Get-Location",
    "git status",
    "Format-Hex",
    "Get-FileHash",
)


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def digest_text(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": len(rows),
        "bytes": len(text.encode("utf-8")),
        "sha256_lf_normalized": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def parse_events(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for event_line, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        item = event.get("item") if event.get("type") == "item.completed" else None
        if not isinstance(item, dict):
            continue
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            messages.append(
                {
                    "event_line": event_line,
                    "item_id": str(item.get("id")),
                    "text": normalize_text(item["text"]),
                }
            )
        elif item.get("type") == "command_execution":
            commands.append(
                {
                    "event_line": event_line,
                    "item_id": str(item.get("id")),
                    "command": normalize_text(str(item.get("command", ""))),
                    "output": normalize_text(str(item.get("aggregated_output", ""))),
                    "exit_code": item.get("exit_code"),
                    "status": item.get("status"),
                }
            )
    return messages, commands


def fault_parts(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    devices: list[str] = []
    reasons: list[str] = []
    for item in raw["actual_result_items"]:
        device, reason = str(item).split(";", 1)
        devices.append(device.strip())
        reasons.append(reason.strip())
    return devices, reasons


def contains_any(text: str, values: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(value.lower() in lowered for value in values if value)


def reason_is_explicit(text: str, reasons: list[str]) -> bool:
    for reason in reasons:
        aliases = (reason,) + FAULT_ALIASES.get(reason, ())
        if contains_any(text, aliases):
            return True
    return False


def candidate_quality(
    annotation: dict[str, Any], raw: dict[str, Any], messages: list[dict[str, Any]], commands: list[dict[str, Any]]
) -> dict[str, Any]:
    pre_final = "\n".join(message["text"] for message in messages[:-1])
    evidence = str(annotation.get("evidence", ""))
    devices, reasons = fault_parts(raw)
    device_hits = sum(device.lower() in pre_final.lower() for device in devices)
    evidence_device_hits = sum(device.lower() in evidence.lower() for device in devices)
    reason_hit = reason_is_explicit(pre_final, reasons)
    evidence_reason_hit = reason_is_explicit(evidence, reasons)
    successful_commands = [
        command
        for command in commands
        if command["status"] == "completed" and command["exit_code"] in (0, None)
    ]
    unique_commands = {command["command"] for command in successful_commands}
    duplicate_commands = len(successful_commands) - len(unique_commands)
    failed_commands = len(commands) - len(successful_commands)
    message_count = len(messages)
    command_count = len(commands)
    duration = float(raw.get("source", {}).get("duration_seconds") or 0.0)

    # Evidence grounding dominates; efficiency only breaks ties among grounded trajectories.
    score = (
        120.0 * float(reason_hit)
        + 80.0 * device_hits / max(1, len(devices))
        + 50.0 * float(evidence_reason_hit)
        + 35.0 * evidence_device_hits / max(1, len(devices))
        + 12.0 * float(3 <= message_count <= 6)
        - 2.5 * max(0, message_count - 6)
        - 0.045 * command_count
        - 0.20 * duplicate_commands
        - 0.60 * failed_commands
        - 0.0015 * duration
    )
    return {
        "score": round(score, 6),
        "pre_final_reason_explicit": reason_hit,
        "pre_final_device_hits": device_hits,
        "expected_device_count": len(devices),
        "curated_evidence_reason_explicit": evidence_reason_hit,
        "curated_evidence_device_hits": evidence_device_hits,
        "message_count": message_count,
        "command_count": command_count,
        "successful_command_count": len(successful_commands),
        "duplicate_command_count": duplicate_commands,
        "failed_command_count": failed_commands,
        "duration_seconds": duration,
    }


def command_target(command: str) -> str:
    normalized = re.sub(r"\\+", "/", command)
    match = re.search(r"CampusNetwork_\d+/([^/'\"]+)/([^/'\"]+)", normalized)
    if match:
        return f"{match.group(1)}/{match.group(2)}".lower()
    return digest_text(command)[:16]


def keyword_set(text: str, devices: list[str], reasons: list[str]) -> set[str]:
    lowered = text.lower()
    keywords: set[str] = set()
    for value in devices + reasons:
        if value:
            keywords.add(value.lower())
    for reason in reasons:
        keywords.update(alias.lower() for alias in FAULT_ALIASES.get(reason, ()))
    keywords.update(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?\b", lowered))
    keywords.update(
        token.lower()
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_.:/-]{2,}\b", text)
        if "campusnetwork" not in token.lower()
    )
    keywords.update(term for term in PROTOCOL_TERMS if term in lowered)
    return {keyword for keyword in keywords if len(keyword) >= 2}


def command_score(
    command: dict[str, Any], keywords: set[str], devices: list[str], reasons: list[str], final_segment: bool
) -> float:
    if command["status"] != "completed" or command["exit_code"] not in (0, None):
        return -math.inf
    cmd = command["command"].lower()
    output = command["output"].lower()
    combined = f"{cmd}\n{output}"
    score = 0.0
    for keyword in keywords:
        if keyword in cmd:
            score += 4.0
        elif keyword in output:
            score += 1.2
    if contains_any(cmd, PROTOCOL_TERMS):
        score += 2.0
    if final_segment and contains_any(cmd, devices):
        score += 18.0
    if final_segment and reason_is_explicit(combined, reasons):
        score += 12.0
    if any(marker.lower() in cmd for marker in HOUSEKEEPING_MARKERS):
        score -= 6.0
    if "get-content" in cmd or "select-string" in cmd:
        score += 1.5
    if not command["output"]:
        score -= 0.5
    return score


def select_segment_commands(
    candidates: list[dict[str, Any]], target_text: str, devices: list[str], reasons: list[str], final_segment: bool
) -> list[dict[str, Any]]:
    keywords = keyword_set(target_text, devices, reasons)
    deduplicated: dict[str, dict[str, Any]] = {}
    for command in candidates:
        deduplicated.setdefault(command["command"], command)
    scored = [
        # Quantize before ranking.  command_score sums matches from a set of
        # keywords; without quantization, hash-seed-dependent floating-point
        # accumulation can reorder commands that are semantically tied.
        (round(command_score(command, keywords, devices, reasons, final_segment), 6), command)
        for command in deduplicated.values()
    ]
    scored = [(score, command) for score, command in scored if score > 0 and math.isfinite(score)]
    scored.sort(key=lambda item: (-item[0], int(item[1]["event_line"])))

    selected: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    for score, command in scored:
        target = command_target(command["command"])
        if target in used_targets:
            continue
        selected.append({**command, "selection_score": round(score, 3), "target": target})
        used_targets.add(target)
        if len(selected) >= MAX_ACTIONS_PER_STAGE:
            break
    if len(selected) < min(2, len(scored)):
        chosen_ids = {command["item_id"] for command in selected}
        for score, command in scored:
            if command["item_id"] in chosen_ids:
                continue
            selected.append(
                {**command, "selection_score": round(score, 3), "target": command_target(command["command"])}
            )
            if len(selected) >= min(2, len(scored)):
                break
    selected.sort(key=lambda command: int(command["event_line"]))
    return selected


def excerpt_output(output: str, keywords: set[str], limit: int) -> tuple[str, bool]:
    if not output:
        return "(empty output)", False
    lines = output.splitlines()
    selected: set[int] = set()
    lowered_keywords = tuple(keyword.lower() for keyword in keywords if keyword)
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))
    if not selected:
        selected.update(range(min(30, len(lines))))
        selected.update(range(max(0, len(lines) - 5), len(lines)))
    rendered: list[str] = []
    previous: int | None = None
    for index in sorted(selected):
        if previous is not None and index != previous + 1:
            rendered.append("... [unselected lines omitted] ...")
        rendered.append(lines[index])
        previous = index
    text = "\n".join(rendered)
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n... [excerpt truncated] ..."
    return text, len(selected) != len(lines) or len(text) < len(output)


def split_stage_message(text: str) -> tuple[str, str]:
    parts = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*|\n+", text) if part.strip()]
    if len(parts) >= 2:
        return "".join(parts[:-1]), parts[-1]
    return text, "继续执行能够最大区分剩余候选根因的实际取证动作。"


def build_initial_user(raw: dict[str, Any]) -> str:
    source = raw["source_record"]
    return "\n\n".join(
        [
            str(source["question"]),
            "请根据当前已经返回的证据逐步排障；证据不足时继续调用工具，不要提前输出最终答案。",
            "答案格式要求：",
            str(source["output_format"]),
        ]
    )


def build_stage_data(
    raw: dict[str, Any], messages: list[dict[str, Any]], commands: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    devices, reasons = fault_parts(raw)
    stages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        is_final = index == len(messages) - 1
        if is_final:
            thinking = "前序证据已经完成收敛；按最小根因集合原则，只输出被证据直接支持的故障项。"
            conclusion = normalize_text(str(raw["final_answer"]))
            selected_commands: list[dict[str, Any]] = []
            target_text = conclusion
        else:
            thinking, conclusion = split_stage_message(message["text"])
            lower = int(message["event_line"])
            upper = int(messages[index + 1]["event_line"])
            candidates = [
                command for command in commands if lower < int(command["event_line"]) < upper
            ]
            next_text = messages[index + 1]["text"]
            selected_commands = select_segment_commands(
                candidates,
                f"{message['text']}\n{next_text}",
                devices,
                reasons,
                final_segment=index == len(messages) - 2,
            )
            target_text = next_text
        keywords = keyword_set(target_text, devices, reasons)
        actions: list[dict[str, Any]] = []
        for action_index, command in enumerate(selected_commands, 1):
            excerpt, excerpted = excerpt_output(command["output"], keywords, CURRENT_OUTPUT_LIMIT)
            actions.append(
                {
                    "action_id": f"S{index + 1:02d}-A{action_index:02d}",
                    "item_id": command["item_id"],
                    "event_line": command["event_line"],
                    "command": command["command"],
                    "output_excerpt": excerpt,
                    "output_is_excerpt": excerpted,
                    "exit_code": command["exit_code"],
                    "status": command["status"],
                    "selection_score": command["selection_score"],
                    "target": command["target"],
                    "command_sha256_lf_normalized": digest_text(command["command"]),
                    "full_output_sha256_lf_normalized": digest_text(command["output"]),
                    "excerpt_sha256_lf_normalized": digest_text(excerpt),
                }
            )
        stages.append(
            {
                "source_message_index": index,
                "source_message_event_line": message["event_line"],
                "source_message_item_id": message["item_id"],
                "thinking": thinking,
                "conclusion": conclusion,
                "is_final": is_final,
                "actions": actions,
            }
        )
    return stages


def tool_call_messages(actions: list[dict[str, Any]], loss_scale: float) -> list[dict[str, Any]]:
    return [
        {
            "role": "tool_call",
            "content": json.dumps(
                {"name": "powershell", "arguments": {"command": action["command"]}},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "loss_scale": loss_scale,
        }
        for action in actions
    ]


def tool_response_messages(actions: list[dict[str, Any]], compact: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for action in actions:
        output = action["output_excerpt"]
        if compact and len(output) > HISTORICAL_OUTPUT_LIMIT:
            output = output[:HISTORICAL_OUTPUT_LIMIT].rstrip() + "\n... [historical excerpt truncated] ..."
        messages.append(
            {
                "role": "tool_response",
                "content": json.dumps(
                    {
                        "status": action["status"],
                        "exit_code": action["exit_code"],
                        "output": output,
                        "historical_compaction": compact,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return messages


def build_messages(raw: dict[str, Any], stages: list[dict[str, Any]], target_index: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_initial_user(raw)},
    ]
    for stage_index in range(target_index + 1):
        stage = stages[stage_index]
        current = stage_index == target_index
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>\n{stage['thinking']}\n</think>\n\n",
                "loss_scale": THINKING_LOSS_SCALE if current else HISTORY_LOSS_SCALE,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": stage["conclusion"],
                "loss_scale": TARGET_LOSS_SCALE if current else HISTORY_LOSS_SCALE,
            }
        )
        messages.extend(
            tool_call_messages(
                stage["actions"], TARGET_LOSS_SCALE if current else HISTORY_LOSS_SCALE
            )
        )
        if not current:
            messages.extend(
                tool_response_messages(
                    stage["actions"], compact=stage_index + 1 < target_index
                )
            )
    return messages


def public_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in action.items() if key != "output_excerpt"}
        for action in actions
    ]


def build_sft_rows(
    annotation: dict[str, Any], raw: dict[str, Any], messages: list[dict[str, Any]], commands: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    stages = build_stage_data(raw, messages, commands)
    rows: list[dict[str, Any]] = []
    for target_index, stage in enumerate(stages):
        if stage["is_final"]:
            target_type = "decision"
        elif not stage["actions"]:
            # Some source trajectories explicitly state that the evidence has
            # converged and then answer in the next message.  This is a useful
            # stop-investigating target, not a missing or fabricated tool call.
            target_type = "decision_ready"
        elif target_index == 0:
            target_type = "planning"
        else:
            target_type = "reasoning"
        row = {
            "id": f"{raw['id']}_step_{target_index + 1:02d}",
            "tools": json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":")),
            "messages": build_messages(raw, stages, target_index),
            "metadata": {
                "dataset_type": "reasoning_trajectory_best1_step",
                "target_type": target_type,
                "review_status": "auto_curated_draft",
                "split": annotation["split"],
                "case_id": raw["case_id"],
                "row_index": raw["row_index"],
                "trajectory_id": raw["id"],
                "success_slot": raw["success_slot"],
                "attempt_index": raw["attempt_index"],
                "step_index": target_index + 1,
                "step_count": len(stages),
                "source_file": annotation["raw_file"],
                "source_event_file": annotation["events_file"],
                "source_event_sha256_lf_normalized": annotation["events_sha256_lf_normalized"],
                "source_message_index": stage["source_message_index"],
                "source_message_event_line": stage["source_message_event_line"],
                "source_message_item_id": stage["source_message_item_id"],
                "thinking_source": "split_from_original_visible_agent_message" if not stage["is_final"] else "minimal_final_bridge",
                "thinking_is_original_hidden_chain_of_thought": False,
                "current_actions": public_actions(stage["actions"]),
                "current_action_count": len(stage["actions"]),
                "evidence_converged_without_next_tool_call": target_type == "decision_ready",
                "future_event_leakage_checked": True,
                "final_answer_visible": stage["is_final"],
                "actual_result_items": raw["actual_result_items"] if stage["is_final"] else None,
                "reference_answer_match": bool(raw["answer_matches_reference"]) if stage["is_final"] else None,
                "loss_policy": {
                    "thinking": THINKING_LOSS_SCALE,
                    "conclusion_or_result": TARGET_LOSS_SCALE,
                    "tool_calls": TARGET_LOSS_SCALE,
                    "history": HISTORY_LOSS_SCALE,
                    "tool_responses": "context_only",
                },
            },
        }
        rows.append(row)
    return rows


def validate_raw(annotation: dict[str, Any], raw: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    if not annotation.get("selected") or annotation.get("review_status") != "draft":
        raise ValueError(f"{annotation['id']}: source trajectory is not an admitted draft")
    if not raw.get("answer_matches_reference") or not raw.get("independent_judgment", {}).get("correct"):
        raise ValueError(f"{annotation['id']}: source answer is not independently correct")
    if not messages or messages[-1]["text"] != normalize_text(str(raw["final_answer"])):
        raise ValueError(f"{annotation['id']}: final event mismatch")
    if digest_file(ROOT / annotation["raw_file"]) != annotation["raw_sha256_lf_normalized"]:
        raise ValueError(f"{annotation['id']}: raw hash mismatch")
    if digest_file(ROOT / annotation["events_file"]) != annotation["events_sha256_lf_normalized"]:
        raise ValueError(f"{annotation['id']}: event hash mismatch")


def main() -> None:
    source = load_json(SOURCE_CURATION)
    annotations = source["trajectories"]
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_case[int(annotation["case_id"])].append(annotation)

    chosen: list[dict[str, Any]] = []
    chosen_payloads: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
    for case_id in sorted(by_case):
        candidates: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = []
        for annotation in by_case[case_id]:
            raw_path = ROOT / annotation["raw_file"]
            raw = load_json(raw_path)
            messages, commands = parse_events(ROOT / annotation["events_file"])
            validate_raw(annotation, raw, messages)
            quality = candidate_quality(annotation, raw, messages, commands)
            candidates.append((annotation, raw, messages, commands, quality))
        candidates.sort(
            key=lambda item: (
                -float(item[4]["score"]),
                int(item[1]["attempt_index"]),
                int(item[1]["success_slot"]),
            )
        )
        annotation, raw, messages, commands, quality = candidates[0]
        candidate_scores = [
            {
                "trajectory_id": candidate_raw["id"],
                "attempt_index": candidate_raw["attempt_index"],
                "success_slot": candidate_raw["success_slot"],
                **candidate_quality_value,
            }
            for _, candidate_raw, _, _, candidate_quality_value in candidates
        ]
        chosen.append(
            {
                "case_id": case_id,
                "split": annotation["split"],
                "selected_trajectory_id": raw["id"],
                "selected_raw_file": annotation["raw_file"],
                "selected_events_file": annotation["events_file"],
                "selection_method": "grounding_first_then_efficiency_v1",
                "quality": quality,
                "candidate_scores": candidate_scores,
            }
        )
        chosen_payloads.append((annotation, raw, messages, commands))

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    trajectories_by_split = defaultdict(int)
    cases_by_split: dict[str, set[int]] = defaultdict(set)
    for annotation, raw, messages, commands in chosen_payloads:
        rows = build_sft_rows(annotation, raw, messages, commands)
        split = annotation["split"]
        trajectories_by_split[split] += 1
        cases_by_split[split].add(int(raw["case_id"]))
        (train_rows if split == "train" else validation_rows).extend(rows)

    train_cases = cases_by_split["train"]
    validation_cases = cases_by_split["validation"]
    if train_cases & validation_cases:
        raise ValueError("train/validation case leakage")
    expected_train = set(source["split"]["train_case_ids"])
    expected_validation = set(source["split"]["validation_case_ids"])
    if train_cases != expected_train or validation_cases != expected_validation:
        raise ValueError("generated cases differ from the frozen 0804 split")

    selection_document = {
        "schema_version": "0804-best-accepted-trajectory-per-case.v1",
        "source_curation": SOURCE_CURATION.relative_to(ROOT).as_posix(),
        "source_curation_sha256_lf_normalized": digest_file(SOURCE_CURATION),
        "selection_method": {
            "name": "grounding_first_then_efficiency_v1",
            "priority": [
                "root-cause wording is explicit before the final answer",
                "fault device is explicit before the final answer",
                "curated evidence aligns with device and reason",
                "3-6 visible reasoning checkpoints",
                "fewer commands, duplicates, failures, and shorter duration",
            ],
            "manual_review_status": "not_reviewed_fast_run",
        },
        "counts": {
            "eligible_cases": len(chosen),
            "train_cases": len(train_cases),
            "validation_cases": len(validation_cases),
            "selected_trajectories": len(chosen),
        },
        "cases": chosen,
    }
    write_json(BEST_SELECTION, selection_document)

    train_output = write_jsonl(TRAIN_OUTPUT, train_rows)
    validation_output = write_jsonl(VALIDATION_OUTPUT, validation_rows)
    manifest = {
        "schema_version": "qwen36-0804-best1-reasoning-trajectory-sft.v1",
        "status": "auto_curated_draft_fast_run",
        "scope": "data/2026-08-04 only",
        "source_curation": SOURCE_CURATION.relative_to(ROOT).as_posix(),
        "source_curation_sha256_lf_normalized": digest_file(SOURCE_CURATION),
        "best_selection": BEST_SELECTION.relative_to(ROOT).as_posix(),
        "best_selection_sha256_lf_normalized": digest_file(BEST_SELECTION),
        "split": {
            "group_key": "case_id",
            "train_case_count": len(train_cases),
            "validation_case_count": len(validation_cases),
            "train_validation_case_intersection": [],
            "validation_case_ids": sorted(validation_cases),
        },
        "counts": {
            "selected_train_trajectories": trajectories_by_split["train"],
            "selected_validation_trajectories": trajectories_by_split["validation"],
            "train_sft_rows": len(train_rows),
            "validation_sft_rows": len(validation_rows),
        },
        "conversion": {
            "one_trajectory_per_case": True,
            "one_sample_per_visible_reasoning_checkpoint": True,
            "original_message_policy": "split into low-weight thinking and strict stage conclusion",
            "command_selection": "top causally relevant successful unique commands per stage",
            "max_actions_per_stage": MAX_ACTIONS_PER_STAGE,
            "zero_action_nonfinal_checkpoint": (
                "kept as decision_ready when the source evidence has converged; "
                "no unrelated tool call is fabricated"
            ),
            "irrelevant_duplicate_failed_commands": "omitted",
            "infrastructure_failures_and_interruptions": "not present and never archived",
        },
        "loss_policy": {
            "thinking": THINKING_LOSS_SCALE,
            "stage_conclusion_or_result": TARGET_LOSS_SCALE,
            "current_tool_calls": TARGET_LOSS_SCALE,
            "historical_assistant_and_tool_calls": HISTORY_LOSS_SCALE,
            "tool_responses": "context_only",
            "required_cli": "--loss_scale default --is_binary_loss_scale false",
        },
        "training_profile": {
            "max_length": 16384,
            "truncation_strategy": "delete",
            "preserve_thinking": True,
            "add_non_thinking_prefix": False,
            "tokenizer_preflight_required": True,
        },
        "outputs": {"train": train_output, "validation": validation_output},
    }
    write_json(MANIFEST_OUTPUT, manifest)
    print(f"Selected {len(chosen)} trajectories: train={len(train_cases)}, validation={len(validation_cases)}")
    print(f"Generated SFT rows: train={len(train_rows)}, validation={len(validation_rows)}")
    print(f"Selection: {BEST_SELECTION.relative_to(ROOT).as_posix()}")
    print(f"Manifest: {MANIFEST_OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
