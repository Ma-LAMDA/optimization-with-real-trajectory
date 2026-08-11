#!/usr/bin/env python3
"""Build the 0809 error-aware SFT release from the frozen 0805 parent.

The converter deliberately does not consume Agent evaluation trajectories.  It
derives all new counterfactual supervision from the 72 frozen training cases,
keeps the 12 validation cases untouched, and replaces (rather than appends)
one base-core exposure per query and epoch so the 0805 row/optimizer budget is
preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import convert_0807_evidence_gated_reasoning_sft as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-08-09"
PARENT_ROOT = ROOT / "data" / "2026-08-05"
FROZEN_VALIDATION_CASES = {2, 12, 19, 20, 29, 38, 65, 71, 85, 86, 99, 100}
EPOCHS = 5
COUNTERFACTUAL_EXPOSURES_PER_QUERY_PER_EPOCH = 1

PARENT_SFT = PARENT_ROOT / "sft"
PARENT_TRAIN = PARENT_SFT / "qwen3_6_27b_reasoning_causal_path_train.jsonl"
PARENT_CORE = PARENT_SFT / "qwen3_6_27b_reasoning_causal_path_train_core.jsonl"
PARENT_ENDPOINT_POOL = (
    PARENT_SFT / "qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl"
)
PARENT_VALIDATION = (
    PARENT_SFT / "qwen3_6_27b_reasoning_causal_path_validation.jsonl"
)
PARENT_MANIFEST = PARENT_SFT / "reasoning_causal_path_manifest.json"
DESIGN_AUDIT_REPORT = (
    ROOT / "data" / "2026-08-09" / "curation" / "DESIGN_AUDIT_INPUT.md"
)

LABEL_NEIGHBORS: dict[str, list[str]] = {
    "全局STP未使能": ["BGP配置错误", "L3VPN配置错误", "端口未使能STP"],
    "STP BPDU被过滤": ["缺失静态路由", "NAT内网接口属性配置错误或者缺失", "端口未使能STP"],
    "存在IP路由环路": ["存在MPLS标签环路", "BGP配置错误", "L3VPN配置错误"],
    "存在MPLS标签环路": ["存在IP路由环路", "L3VPN配置错误"],
    "VRRP工作在非抢占模式": ["VRRP Master角色规划不合理"],
    "VRRP Master角色规划不合理": ["VRRP工作在非抢占模式"],
}

LABEL_ACTION_REJECTIONS: dict[str, list[str]] = {
    "全局STP未使能": ["跨故障族搜索BGP/L3VPN", "在未读取全局STP状态前全盘grep"],
    "STP BPDU被过滤": ["转向静态路由/NAT", "重复枚举无关设备配置"],
    "存在IP路由环路": ["转向MPLS/L3VPN", "未闭合三个next-hop便提前换协议"],
    "存在MPLS标签环路": ["转向IP路由/L3VPN", "未闭合三段标签路径便提前换协议"],
    "VRRP工作在非抢占模式": ["把Preempt状态误映射为Master角色", "读取无关路由协议"],
    "VRRP Master角色规划不合理": ["把角色/转发链误映射为nonpreempt", "读取无关路由协议"],
}

BOUNDARY_RULES: dict[str, str] = {
    "全局STP未使能": "全局Protocol Status为Disabled才映射到全局STP未使能；BGP/L3VPN状态和端口级STP不是同一事实。",
    "STP BPDU被过滤": "端口配置stp port bpdu-filter enable直接映射到BPDU被过滤；它不等价于路由、NAT或泛化STP故障。",
    "存在IP路由环路": "必须把同一前缀的三条静态路由下一跳逐一绑定到三台设备的接口地址，并证明有向next-hop闭环；仅有不同下一跳不能证明环路。",
    "存在MPLS标签环路": "必须同时证明三段static-LSP的next-hop设备有向闭环及out-label到下一跳in-label的有序闭环；仅比较标签多重集合不充分。",
    "VRRP工作在非抢占模式": "Preempt : NO只映射到VRRP非抢占；它不证明Master角色规划错误。",
    "VRRP Master角色规划不合理": "必须从源主机/24确定业务VLAN，再把该VLAN的VRRP Master、MST实例映射和同实例ALTE/discarding逐设备闭合；它不等价于Preempt : NO。",
}

ACTION_RELEVANT_PATTERNS: dict[str, str] = {
    "全局STP未使能": r"display_stp|current-configuration|interface_brief",
    "STP BPDU被过滤": r"display_stp|current-configuration|logbuffer",
    "存在IP路由环路": r"display_ip_routing-table|ip_route|route-static|display_current-configuration|display_interface_",
    "存在MPLS标签环路": r"display_mpls_lsp|static-lsp|display_current-configuration|display_interface_",
    "VRRP工作在非抢占模式": r"display_vrrp",
    "VRRP Master角色规划不合理": r"display_vrrp|display_stp|ip_route_show",
}

ACTION_EVIDENCE_FOCUS: dict[str, str] = {
    "全局STP未使能": "全局 STP 运行状态与生效配置",
    "STP BPDU被过滤": "上联接口的 BPDU filter 生效配置与 STP 端口角色",
    "存在IP路由环路": "静态路由下一跳、出接口和接口地址归属",
    "存在MPLS标签环路": "静态 LSP 入/出标签、下一跳和接口归属",
    "VRRP工作在非抢占模式": "VRRP 抢占状态",
    "VRRP Master角色规划不合理": "源 VLAN、VRRP Master、MST 映射和端口角色",
}

FACT_KIND_REQUIREMENTS: dict[str, str] = {
    "protocol_status_disabled": "全局 Protocol Status: Disabled",
    "bpdu_filter_current_configuration": "current configuration 中的 BPDU filter 生效行",
    "static_route_directed_next_hop": "同一前缀静态路由的有向 next-hop",
    "next_hop_interface_owner": "next-hop 地址对应的接口 IP owner",
    "static_lsp_directed_label_hop": "static-LSP 的 next-hop 与入/出标签链",
    "vrrp_preempt_no": "VRRP Preempt: NO",
    "source_host_ipv4_vlan": "源主机 /24 对应业务 VLAN",
    "vrrp_master": "业务 Vlanif 的 VRRP Master",
    "mst_vlan_instance_mapping": "业务 VLAN 到 MST instance 的显式映射",
    "mst_alternate_discarding": "同一 MST instance 的 ALTE/discarding 端口角色",
}

PREMATURE_CONVERGENCE_PHRASES = (
    "证据已经形成闭环",
    "证据已形成闭环",
    "关键区分证据已出现",
    "决定性证据已出现",
    "唯一异常",
    "唯一的异常",
    "决定性异常",
    "根因应归",
)


def action_command_is_label_relevant(label: str, command: str) -> bool:
    lowered = command.lower()
    if label == "存在IP路由环路" and "display_current-configuration" in lowered:
        return lowered.startswith("cat ") or bool(
            re.search(r"route-static|150\.0\.0\.1|160\.0\.0\.1", lowered)
        )
    if label == "存在MPLS标签环路" and "display_current-configuration" in lowered:
        return lowered.startswith("cat ") or bool(
            re.search(r"static-lsp|in-label|out-label", lowered)
        )
    return bool(re.search(ACTION_RELEVANT_PATTERNS[label], command, re.I))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def resolve_data_root(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    path = path.resolve()
    expected = (ROOT / "data" / "2026-08-09").resolve()
    if path != expected:
        raise ValueError(f"0809 converter only writes {expected}; received {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
        newline="\n",
    )
    return output_record(path, len(materialized))


def lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def digest_file(path: Path) -> str:
    return hashlib.sha256(lf_bytes(path)).hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def output_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256_lf_normalized": digest_file(path),
    }


def rewrite_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rewrite_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_identity(item) for item in value]
    if isinstance(value, str):
        return value.replace("data/2026-08-05", "data/2026-08-09")
    return value


def raw_tree(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, digest_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_digest(records: dict[str, tuple[int, str]]) -> str:
    payload = "".join(
        f"{name}\t{size}\t{digest}\n"
        for name, (size, digest) in sorted(records.items())
    )
    return digest_text(payload)


def rewrite_parent_row(row: dict[str, Any]) -> dict[str, Any]:
    rewritten = rewrite_identity(copy.deepcopy(row))
    metadata = rewritten.setdefault("metadata", {})
    metadata.update(
        {
            "release_id": "0809",
            "release_data_root": "data/2026-08-09",
            "parent_release": "0805",
            "parent_row_id": row["id"],
            "parent_row_sha256": digest_text(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            ),
            "error_mining_agent_trajectory_used": False,
            "validation_case_used_for_augmentation": False,
        }
    )
    return rewritten


def suppress_unsafe_parent_training_tool_calls(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Delete unsafe inherited call/response pairs from the visible history.

    A zero-loss tool exchange still conditions every later supervised token.  It
    therefore cannot remain merely because its own syntax is masked.  Calls to
    another/unresolved snapshot and saved_configs globs are removed together
    with their FIFO-matched responses; exact current-snapshot calls survive.
    """

    snapshot = expected_snapshot(row["messages"])
    suppressed: list[dict[str, Any]] = []
    messages = row["messages"]
    remove_indices: set[int] = set()
    pending: list[tuple[int, bool]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool_call":
            original_loss = float(message.get("loss_scale", 0) or 0)
            payload = json.loads(message["content"])
            command = payload["arguments"]["cmd"]
            folders = set(re.findall(r"saved_configs/([^/\s]+)", command))
            path_glob = bool(re.search(r"saved_configs/[^\s]*(?:\*|\?|\[)", command))
            reasons: list[str] = []
            if folders != {snapshot}:
                reasons.append("snapshot_mismatch_or_unresolved")
            if path_glob:
                reasons.append("saved_configs_path_glob")
            unsafe = bool(reasons)
            pending.append((index, unsafe))
            if unsafe:
                remove_indices.add(index)
                suppressed.append(
                    {
                        "command": command,
                        "original_loss_scale": original_loss,
                        "reasons": reasons,
                        "command_sha256": digest_text(command),
                    }
                )
        elif role == "tool_response" and pending:
            _, unsafe = pending.pop(0)
            if unsafe:
                remove_indices.add(index)
    row["messages"] = [
        message for index, message in enumerate(messages) if index not in remove_indices
    ]
    metadata = row["metadata"]
    metadata["0809_parent_tool_path_gate"] = {
        "expected_snapshot": snapshot,
        "removed_call_count": len(suppressed),
        "removed_message_count": len(remove_indices),
        "suppressed_calls": suppressed,
        "policy": "remove cross-snapshot, unresolved-snapshot and saved_configs path-glob call/response pairs from supervised histories",
    }
    return row


def case_id(row: dict[str, Any]) -> int:
    return int(row["metadata"]["case_id"])


def normalize_result_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(f"unexpected actual_result_items: {value!r}")


def result_label(item: str) -> str:
    if ";" not in item:
        raise ValueError(f"result item lacks device separator: {item!r}")
    return item.split(";", 1)[1]


def result_device(item: str) -> str:
    return item.split(";", 1)[0]


def zero_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned = copy.deepcopy(messages)
    for message in cloned:
        if "loss_scale" in message:
            message["loss_scale"] = 0.0
    return cloned


def action_visible_grounding_issues(row: dict[str, Any]) -> list[str]:
    """Find factual/convergence claims unsupported before the first loss token.

    Action replay trains the model at the current observation boundary.  A
    precise fact may therefore be asserted only when its decisive literal is
    already present in a tool response before the first positive-loss message.
    Explicitly tentative wording is allowed because it teaches hypothesis
    formation rather than leaking a future observation.
    """

    first_positive = next(
        (
            index
            for index, message in enumerate(row["messages"])
            if float(message.get("loss_scale", 0) or 0) > 0
        ),
        len(row["messages"]),
    )
    visible = "\n".join(
        message.get("content", "")
        for message in row["messages"][:first_positive]
        if message.get("role") == "tool_response"
    )
    positive = "\n".join(
        message.get("content", "")
        for message in row["messages"]
        if message.get("role") == "assistant"
        and float(message.get("loss_scale", 0) or 0) > 0
    )
    if not positive:
        return []

    tentative = "待验证" in positive or "尚不足以确认" in positive
    issues: list[str] = []
    if not tentative:
        claims_disabled = bool(
            re.search(r"Protocol\s+Status\s*:\s*Disabled", positive, re.I)
            or re.search(r"全局.{0,24}STP.{0,24}(?:未使能|未启用|Disabled)", positive, re.I)
            or re.search(r"(?:缺少|没有).{0,20}`?stp\s+enable`?", positive, re.I)
        )
        if claims_disabled and not re.search(
            r"Protocol\s+Status\s*:\s*Disabled", visible, re.I
        ):
            issues.append("unsupported_global_stp_disabled")

        claims_bpdu_filter = bool(
            re.search(r"stp\s+port\s+bpdu-filter\s+enable", positive, re.I)
            or (re.search(r"BPDU", positive, re.I) and "过滤" in positive)
        )
        if claims_bpdu_filter and not re.search(
            r"stp\s+port\s+bpdu-filter\s+enable", visible, re.I
        ):
            issues.append("unsupported_bpdu_filter_enabled")

        claims_nonpreempt = bool(
            re.search(r"Preempt\s*:\s*NO", positive, re.I)
            or re.search(r"preempt\s+disable", positive, re.I)
            or "非抢占" in positive
        )
        if claims_nonpreempt and not re.search(
            r"Preempt\s*:\s*NO|preempt\s+disable", visible, re.I
        ):
            issues.append("unsupported_vrrp_nonpreempt")

        claims_stp_enabled = bool(
            re.search(r"(?:配置|启用|存在|包含).{0,20}`?stp\s+(?:global\s+)?enable`?", positive, re.I)
        )
        if claims_stp_enabled and not re.search(
            r"(?m)^\s*stp\s+(?:global\s+)?enable\s*$", visible, re.I
        ):
            issues.append("unsupported_stp_enabled")

    formed_loop_claim = re.search(
        r"(?:已经|已)\s*(?:确认|证明)?\s*(?:形成|构成)\s*(?:IP|MPLS)?\s*(?:路由|标签|转发)?\s*环路",
        positive,
        re.I,
    )
    if any(phrase in positive for phrase in PREMATURE_CONVERGENCE_PHRASES) or formed_loop_claim:
        issues.append("premature_convergence_assertion")
    return sorted(set(issues))


def action_target_devices(row: dict[str, Any]) -> list[str]:
    devices: set[str] = set()
    trailing_calls: list[dict[str, Any]] = []
    for message in reversed(row["messages"]):
        if message.get("role") != "tool_call":
            break
        trailing_calls.append(message)
    for message in reversed(trailing_calls):
        payload = json.loads(message["content"])
        command = payload["arguments"]["cmd"]
        devices.update(re.findall(r"saved_configs/[^/\s]+/([^/\s]+)/", command))
    return sorted(devices)


def normalize_visible_literal(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def structural_action_binding_status(
    row: dict[str, Any], endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute label closure only from tool responses before first loss."""

    first_positive = next(
        (
            index
            for index, message in enumerate(row["messages"])
            if float(message.get("loss_scale", 0) or 0) > 0
        ),
        len(row["messages"]),
    )
    visible_responses = [
        message.get("content", "")
        for message in row["messages"][:first_positive]
        if message.get("role") == "tool_response"
    ]
    visible = normalize_visible_literal("\n".join(visible_responses))
    candidates: list[dict[str, Any]] = []
    for endpoint in endpoint_rows:
        binding = endpoint["metadata"]["evidence_binding"]
        facts = binding["facts"]
        matched = [
            fact
            for fact in facts
            if normalize_visible_literal(str(fact["output_line"])) in visible
        ]
        candidates.append(
            {
                "binding": binding,
                "matched_facts": matched,
                "missing_facts": [fact for fact in facts if fact not in matched],
                "complete": len(matched) == len(facts),
            }
        )
    if not candidates:
        raise ValueError(f"{row['id']}: no strict endpoint binding for action gate")
    best = max(
        candidates,
        key=lambda item: (
            int(item["complete"]),
            len(item["matched_facts"]),
            -len(item["binding"]["facts"]),
            item["binding"]["sha256"],
        ),
    )
    return {
        **best,
        "candidate_binding_count": len(candidates),
        "visible_tool_response_count": len(visible_responses),
        "visible_tool_responses_sha256": digest_text(
            "\n".join(visible_responses)
        ),
    }


def compact_fact_requirement(fact: dict[str, Any]) -> str:
    kind = str(fact["fact_kind"])
    requirement = FACT_KIND_REQUIREMENTS.get(kind, kind)
    return f"{fact['device']}的{requirement}"


def repair_action_visible_grounding_parent_row(
    row: dict[str, Any], decision: dict[str, Any], endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Make every selected incomplete action an auditable pending hypothesis."""

    status = structural_action_binding_status(row, endpoint_rows)
    if status["complete"]:
        raise ValueError(
            f"{row['id']}: complete structural binding reached action-pending repair"
        )
    issues = action_visible_grounding_issues(row)
    metadata = row["metadata"]
    labels, _ = labels_and_items(decision)
    label = labels[0]
    devices = action_target_devices(row)
    device_text = "、".join(devices) if devices else "当前路径相关设备"
    matched_text = "；".join(
        f"{fact['device']}：{compact(fact['output_line'], 160)}"
        for fact in status["matched_facts"][:4]
    )
    if not matched_text:
        matched_text = "尚未出现该故障族的决定性字面事实"
    missing_text = "、".join(
        compact_fact_requirement(fact) for fact in status["missing_facts"]
    )
    original = [
        {
            "content": message.get("content", ""),
            "loss_scale": float(message.get("loss_scale", 0) or 0),
        }
        for message in row["messages"]
        if message.get("role") == "assistant"
        and float(message.get("loss_scale", 0) or 0) > 0
    ]
    original_sha256 = digest_text(
        json.dumps(original, ensure_ascii=False, separators=(",", ":"))
    )
    for message in row["messages"]:
        if message.get("role") != "assistant" or float(
            message.get("loss_scale", 0) or 0
        ) <= 0:
            continue
        if "<think>" in message.get("content", ""):
            message["content"] = (
                "<think>\n"
                f"当前可见事实为：{matched_text}。严格门禁仅满足"
                f"{len(status['matched_facts'])}/{len(status['binding']['facts'])} 项，"
                f"仍缺少{missing_text}，因此“{label}”只能保留为待验证假设。"
                f"下一步通过{device_text}的{ACTION_EVIDENCE_FOCUS[label]}补齐或否定这些缺失事实；"
                "新回显返回前不宣称闭环、唯一根因或最终标签。\n"
                "</think>\n\n"
            )
        else:
            message["content"] = (
                f"待验证假设：“{label}”（当前严格事实门禁"
                f"{len(status['matched_facts'])}/{len(status['binding']['facts'])}）。"
                f"执行以下只读动作核对{missing_text}；结果返回前不输出最终答案。"
            )

    remaining = action_visible_grounding_issues(row)
    if remaining:
        raise ValueError(
            f"{row['id']}: action grounding repair left issues {remaining}"
        )
    metadata["0809_action_visible_grounding_gate"] = {
        "checked": True,
        "repaired": True,
        "structural_binding_complete": False,
        "candidate_binding_count": status["candidate_binding_count"],
        "best_binding_sha256": status["binding"]["sha256"],
        "best_binding_expected_fact_count": len(status["binding"]["facts"]),
        "best_binding_matched_fact_count": len(status["matched_facts"]),
        "matched_fact_descriptors": [
            compact_fact_requirement(fact) for fact in status["matched_facts"]
        ],
        "missing_fact_descriptors": [
            compact_fact_requirement(fact) for fact in status["missing_facts"]
        ],
        "visible_tool_response_count": status["visible_tool_response_count"],
        "visible_tool_responses_sha256": status["visible_tool_responses_sha256"],
        "original_issue_codes": issues,
        "original_positive_assistant_sha256": original_sha256,
        "original_positive_assistant_messages": original,
        "remaining_issues": [],
        "pending_hypothesis_supervised": True,
        "policy": "for every selected action, recompute complete label-family closure from strict endpoint facts before first positive loss; incomplete states supervise only matched facts, missing requirements and a pending hypothesis",
    }
    return row


def unsupervised_context(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in row["messages"]:
        if float(message.get("loss_scale", 0) or 0) > 0:
            continue
        cloned = copy.deepcopy(message)
        if "loss_scale" in cloned:
            cloned["loss_scale"] = 0.0
        messages.append(cloned)
    if not messages or messages[0].get("role") != "system":
        raise ValueError(f"{row['id']}: missing system context")
    return messages


def compact(value: Any, limit: int = 520) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def decision_for_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["metadata"]["target_type"] == "decision"]
    if not candidates:
        raise ValueError(f"q{case_id(rows[0])}: no decision row")
    return sorted(
        candidates,
        key=lambda row: (
            -int(row["metadata"].get("path_cluster_size", 0) or 0),
            len(row["messages"]),
            row["id"],
        ),
    )[0]


def endpoint_row(
    rows: list[dict[str, Any]], decision: dict[str, Any], target_type: str
) -> dict[str, Any]:
    cluster_id = decision["metadata"]["path_cluster_id"]
    matches = [
        row
        for row in rows
        if row["metadata"]["target_type"] == target_type
        and row["metadata"]["path_cluster_id"] == cluster_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{decision['id']}: expected one {target_type} in {cluster_id}, found {len(matches)}"
        )
    return matches[0]


def common_counterfactual_metadata(
    source: dict[str, Any], target_type: str, objective: str
) -> dict[str, Any]:
    metadata = copy.deepcopy(source["metadata"])
    metadata.update(
        {
            "dataset_type": "0809_agent_error_aware_counterfactual",
            "target_type": target_type,
            "source_target_type": source["metadata"]["target_type"],
            "release_id": "0809",
            "release_data_root": "data/2026-08-09",
            "parent_release": "0805",
            "counterfactual_objective": objective,
            "counterfactual_source_row_id": source["id"],
            "training_sampling_role": "counterfactual_pool",
            "error_mining_agent_trajectory_used": False,
            "validation_case_used_for_augmentation": False,
            "future_event_leakage_policy": "target-specific literal/causal gate; never infer a pass from metadata alone",
            "synthetic_stage": True,
            "review_status": "auto_generated_evidence_bound_requires_domain_review",
            "endpoint_schedule_epoch": None,
            "endpoint_schedule_slot": None,
        }
    )
    return metadata


def labels_and_items(decision: dict[str, Any]) -> tuple[list[str], list[str]]:
    items = normalize_result_items(decision["metadata"]["actual_result_items"])
    labels = sorted({result_label(item) for item in items})
    if len(labels) != 1 or labels[0] not in LABEL_NEIGHBORS:
        raise ValueError(f"{decision['id']}: unsupported label set {labels}")
    return labels, items


def tool_observations(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return visible tool-output lines with their originating command/device.

    A command may contain a path glob, so a filename-prefixed output line is a
    stronger device locator than the command itself.  Empty output, transport
    headers, table headers and the historical-truncation marker are never facts.
    """

    pending: list[str] = []
    observations: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool_call":
            payload = json.loads(message["content"])
            pending.append(payload["arguments"]["cmd"])
            continue
        if role != "tool_response":
            continue
        command = pending.pop(0) if pending else ""
        command_device = re.search(r"saved_configs/[^/]+/([^/]+)/", command)
        command_snapshot = re.search(r"saved_configs/([^/]+)/", command)
        for raw_line in message.get("content", "").splitlines():
            line = raw_line.strip()
            if not line or line == "... [historical excerpt truncated] ...":
                continue
            output_device = re.search(r"saved_configs/[^/]+/([^/]+)/", line)
            output_snapshot = re.search(r"saved_configs/([^/]+)/", line)
            device = (
                output_device.group(1)
                if output_device
                else command_device.group(1)
                if command_device
                else ""
            )
            observations.append(
                {
                    "command": command,
                    "snapshot": (
                        output_snapshot.group(1)
                        if output_snapshot
                        else command_snapshot.group(1)
                        if command_snapshot
                        else ""
                    ),
                    "device": device,
                    "output_line": line,
                }
            )
    return observations


def expected_snapshot(messages: list[dict[str, Any]]) -> str:
    user_text = "\n".join(
        message.get("content", "")
        for message in messages
        if message.get("role") == "user"
    )
    match = re.search(r"CampusNetwork(?:-for-perf|_\d+)", user_text)
    if not match:
        raise ValueError("unable to resolve the case snapshot from the user prompt")
    return match.group(0)


def real_ip_static_line(line: str) -> bool:
    return bool(
        re.search(r"(?:^|:\d+:)ip route-static\s+\S+\s+\S+\s+\S+", line, re.I)
        or re.search(
            r"^\S+/\d+\s+Static\s+\d+\s+\d+\s+\S+\s+\d+\.\d+\.\d+\.\d+",
            line,
            re.I,
        )
    )


def real_mpls_static_line(line: str) -> bool:
    return bool(
        (
            re.search(r"(?:^|:\d+:)static-lsp\s+", line, re.I)
            and re.search(r"\bin-label\s+\d+", line, re.I)
            and re.search(r"\bout-label\s+\d+", line, re.I)
            and re.search(r"\bnexthop\s+\d+\.\d+\.\d+\.\d+", line, re.I)
        )
        or re.search(r"^-/-\s+\d+/\d+\s+-/\S+", line)
    )


def ip_static_route_pair(line: str) -> tuple[str, str] | None:
    configured = re.search(
        r"(?:^|:\d+:)ip route-static\s+(\d+\.\d+\.\d+\.\d+)\s+\S+\s+(\d+\.\d+\.\d+\.\d+)",
        line,
        re.I,
    )
    if configured:
        return configured.group(1), configured.group(2)
    runtime = re.search(
        r"^(\d+\.\d+\.\d+\.\d+)/\d+\s+Static\s+\d+\s+\d+\s+\S+\s+(\d+\.\d+\.\d+\.\d+)",
        line,
        re.I,
    )
    return (runtime.group(1), runtime.group(2)) if runtime else None


def mpls_label_pair(line: str) -> tuple[int, int] | None:
    configured = re.search(
        r"\bin-label\s+(\d+).*\bout-label\s+(\d+)", line, re.I
    )
    if configured:
        return int(configured.group(1)), int(configured.group(2))
    runtime = re.search(r"^-/-\s+(\d+)/(\d+)\s+-/\S+", line)
    return (int(runtime.group(1)), int(runtime.group(2))) if runtime else None


def interface_ipv4(line: str) -> str | None:
    match = re.search(
        r"(?:^|:\d+:)ip address\s+(\d+\.\d+\.\d+\.\d+)(?:\s+\S+)?",
        line,
        re.I,
    )
    return match.group(1) if match and "route-static" not in line.lower() else None


def mpls_static_hop(line: str) -> tuple[int, str, int] | None:
    match = re.search(
        r"\bstatic-lsp\s+(?:ingress|transit)\s+\S+.*?"
        r"\bin-label\s+(\d+).*?\bnexthop\s+(\d+\.\d+\.\d+\.\d+)"
        r".*?\bout-label\s+(\d+)",
        line,
        re.I,
    )
    return (int(match.group(1)), match.group(2), int(match.group(3))) if match else None


def source_host_and_vlan(
    messages: list[dict[str, Any]], observations: list[dict[str, str]]
) -> tuple[str, int, dict[str, str]] | None:
    question = "\n".join(
        message.get("content", "")
        for message in messages
        if message.get("role") == "user"
    )
    host_match = re.search(r"\b([A-Za-z][A-Za-z0-9_-]+)\s+ping", question, re.I)
    if not host_match:
        return None
    host = host_match.group(1).lower()
    for observation in observations:
        if observation["device"].lower() != host:
            continue
        address = re.search(
            r"\binet(?:\s+addr:|\s+)(\d+\.\d+\.(\d+)\.\d+)(?:/(\d+))?",
            observation["output_line"],
            re.I,
        )
        if address and address.group(3) == "24" and address.group(1).startswith("10.1."):
            return host, int(address.group(2)), observation
    return None


def select_directed_cycle(
    observations: list[dict[str, str]], answer_devices: set[str], *, mpls: bool
) -> list[dict[str, Any]] | None:
    """Bind forwarding hops to exact next-hop owners and prove a directed cycle."""

    normalized_devices = {device.lower(): device for device in answer_devices}
    owners: dict[str, list[dict[str, str]]] = defaultdict(list)
    for observation in observations:
        address = interface_ipv4(observation["output_line"])
        if address:
            owners[address].append(observation)

    candidates: dict[str, list[tuple[dict[str, str], Any, str, dict[str, str]]]] = defaultdict(list)
    for observation in observations:
        device_key = observation["device"].lower()
        if device_key not in normalized_devices:
            continue
        parsed: Any = (
            mpls_static_hop(observation["output_line"])
            if mpls
            else ip_static_route_pair(observation["output_line"])
        )
        if not parsed:
            continue
        next_hop = parsed[1]
        owner_matches = [
            owner
            for owner in owners.get(next_hop, [])
            if owner["device"].lower() != device_key
            and owner["device"].lower() in normalized_devices
        ]
        owner_devices = {owner["device"].lower() for owner in owner_matches}
        if len(owner_devices) != 1:
            continue
        owner_key = next(iter(owner_devices))
        candidates[device_key].append((observation, parsed, owner_key, owner_matches[0]))

    device_keys = sorted(normalized_devices)
    if len(device_keys) != 3 or any(not candidates[device] for device in device_keys):
        return None
    for combination in product(*(candidates[device] for device in device_keys)):
        successor = {
            device: selected[2] for device, selected in zip(device_keys, combination)
        }
        if set(successor.values()) != set(device_keys):
            continue
        start = device_keys[0]
        if successor.get(successor.get(successor.get(start, ""), ""), "") != start:
            continue
        if any(successor[device] == device for device in device_keys):
            continue
        by_device = dict(zip(device_keys, combination))
        ordered_devices = [start, successor[start], successor[successor[start]]]
        ordered = [by_device[device] for device in ordered_devices]
        if not mpls:
            if len({selected[1][0] for selected in ordered}) != 1:
                continue
        elif any(
            ordered[index][1][2] != ordered[(index + 1) % 3][1][0]
            for index in range(3)
        ):
            continue
        facts: list[dict[str, Any]] = []
        for observation, parsed, owner_key, owner in ordered:
            fact = copy.deepcopy(observation)
            fact.update(
                {
                    "fact_kind": (
                        "static_lsp_directed_label_hop"
                        if mpls
                        else "static_route_directed_next_hop"
                    ),
                    "next_hop_owner": normalized_devices[owner_key],
                    "parsed_transition": parsed,
                }
            )
            facts.append(fact)
        for _, _, _, owner in ordered:
            fact = copy.deepcopy(owner)
            fact["fact_kind"] = "next_hop_interface_owner"
            facts.append(fact)
        return facts
    return None


def vrrp_master_vlan(observation: dict[str, str]) -> int | None:
    line_match = re.search(
        r"\bMaster\b.*\bVlanif(\d+)", observation["output_line"], re.I
    )
    if line_match:
        return int(line_match.group(1))
    if re.search(r"State\s*:\s*Master", observation["output_line"], re.I):
        command_match = re.search(r"Vlanif(\d+)", observation["command"], re.I)
        if command_match:
            return int(command_match.group(1))
    return None


def mapped_mst_instance(line: str, vlan: int) -> int | None:
    match = re.search(r"\binstance\s+(\d+)\s+vlan\s+(.+)$", line, re.I)
    if not match:
        return None
    vlan_tokens = {int(value) for value in re.findall(r"\d+", match.group(2))}
    return int(match.group(1)) if vlan in vlan_tokens else None


def direct_evidence_binding(
    row: dict[str, Any], decision: dict[str, Any]
) -> dict[str, Any] | None:
    """Select only label-specific facts literally visible before the target.

    This deliberately returns ``None`` for partial loop/role histories.  A
    synthetic stop, boundary, minimal-set or elimination row is safer to omit
    than to teach a conclusion from a table header or from the final answer.
    """

    labels, items = labels_and_items(decision)
    label = labels[0]
    answer_devices = {result_device(item) for item in items}
    snapshot = expected_snapshot(row["messages"])
    observations = [
        observation
        for observation in tool_observations(unsupervised_context(row))
        if observation["snapshot"] == snapshot
    ]
    selected: list[dict[str, str]] = []
    qualification = ""

    def select_per_device(predicate: Any, fact_kind: str) -> bool:
        for device in sorted(answer_devices):
            matches = [
                observation
                for observation in observations
                if observation["device"] == device
                and predicate(observation["output_line"])
            ]
            if not matches:
                return False
            fact = copy.deepcopy(matches[0])
            fact["fact_kind"] = fact_kind
            selected.append(fact)
        return True

    if label == "全局STP未使能":
        if not select_per_device(
            lambda line: bool(
                re.search(r"Protocol Status\s*:\s*Disabled", line, re.I)
            ),
            "protocol_status_disabled",
        ):
            return None
        qualification = "each answer device has a literal Protocol Status: Disabled line"
    elif label == "STP BPDU被过滤":
        for device in sorted(answer_devices):
            matches = [
                observation
                for observation in observations
                if observation["device"].lower() == device.lower()
                and "display_current-configuration" in observation["command"].lower()
                and re.search(
                    r"(?:^|:\d+:)stp port bpdu-filter enable\s*$",
                    observation["output_line"],
                    re.I,
                )
            ]
            if not matches:
                return None
            fact = copy.deepcopy(matches[0])
            fact["fact_kind"] = "bpdu_filter_current_configuration"
            selected.append(fact)
        qualification = "each answer device has a literal stp port bpdu-filter enable line from current configuration"
    elif label == "存在IP路由环路":
        cycle = select_directed_cycle(observations, answer_devices, mpls=False)
        if not cycle:
            return None
        selected.extend(cycle)
        qualification = "three same-prefix static routes bind to exact interface-IP owners and form one directed three-device next-hop cycle"
    elif label == "存在MPLS标签环路":
        cycle = select_directed_cycle(observations, answer_devices, mpls=True)
        if not cycle:
            return None
        selected.extend(cycle)
        qualification = "three configured static-LSP hops bind to exact next-hop owners and close both the directed device cycle and ordered out-label-to-next-in-label chain"
    elif label == "VRRP工作在非抢占模式":
        if not select_per_device(
            lambda line: bool(re.search(r"Preempt\s*:\s*NO", line, re.I)),
            "vrrp_preempt_no",
        ):
            return None
        qualification = "each answer device has a literal Preempt: NO line"
    elif label == "VRRP Master角色规划不合理":
        source = source_host_and_vlan(row["messages"], observations)
        if source is None:
            return None
        _, source_vlan, source_observation = source
        source_fact = copy.deepcopy(source_observation)
        source_fact["fact_kind"] = "source_host_ipv4_vlan"
        source_fact["source_vlan"] = source_vlan
        selected.append(source_fact)
        for device in sorted(answer_devices):
            role_chain = None
            for master in observations:
                if master["device"] != device or "ospf" in master["command"].lower():
                    continue
                vlan = vrrp_master_vlan(master)
                if vlan is None or vlan != source_vlan:
                    continue
                for mapping in observations:
                    if mapping["device"] != device:
                        continue
                    instance = mapped_mst_instance(mapping["output_line"], vlan)
                    if instance is None:
                        continue
                    alternate = next(
                        (
                            observation
                            for observation in observations
                            if observation["device"] == device
                            and re.search(
                                rf"^{instance}\s+\S+\s+ALTE\s+discarding\b",
                                observation["output_line"],
                                re.I,
                            )
                        ),
                        None,
                    )
                    if alternate:
                        role_chain = (master, mapping, alternate)
                        break
                if role_chain:
                    break
            if not role_chain:
                return None
            for fact, kind in (
                (role_chain[0], "vrrp_master"),
                (role_chain[1], "mst_vlan_instance_mapping"),
                (role_chain[2], "mst_alternate_discarding"),
            ):
                cloned = copy.deepcopy(fact)
                cloned["fact_kind"] = kind
                selected.append(cloned)
        qualification = "source-host /24 identifies the affected VLAN; each answer device has Master on that Vlanif, an explicit VLAN-to-MST-instance mapping and ALTE/discarding in that same instance"
    else:
        raise ValueError(f"{decision['id']}: unsupported direct-evidence label {label}")

    canonical = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
    return {
        "source_row_id": row["id"],
        "snapshot": snapshot,
        "label": label,
        "qualification_rule": qualification,
        "facts": selected,
        "fact_count": len(selected),
        "sha256": digest_text(canonical),
    }


def rendered_evidence(binding: dict[str, Any], limit: int = 900) -> str:
    facts = [
        f"{fact['device']}：{compact(fact['output_line'], 220)}"
        for fact in binding["facts"]
    ]
    return compact("；".join(facts), limit)


def result_block(items: list[str]) -> str:
    return "<result>\n[\n" + ",\n".join(
        json.dumps(item, ensure_ascii=False) for item in items
    ) + "\n]\n</result>"


def evidence_package(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    decisions = {
        row["metadata"]["path_cluster_id"]: row
        for row in rows
        if row["metadata"]["target_type"] == "decision"
    }
    for evidence in rows:
        if evidence["metadata"]["target_type"] != "evidence_summary":
            continue
        decision = decisions[evidence["metadata"]["path_cluster_id"]]
        binding = direct_evidence_binding(evidence, decision)
        if binding:
            candidates.append((evidence, decision, binding))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[2]["fact_count"],
            int(item[1]["metadata"].get("path_cluster_size", 0) or 0),
            -len(item[0]["messages"]),
            item[0]["id"],
        ),
    )


def build_action_row(
    source: dict[str, Any], decision: dict[str, Any], ordinal: int
) -> dict[str, Any]:
    labels, items = labels_and_items(decision)
    messages = zero_history(source["messages"])
    all_trailing_tools: list[dict[str, Any]] = []
    for message in reversed(messages):
        if message.get("role") == "tool_call":
            all_trailing_tools.append(message)
            continue
        break
    if not all_trailing_tools or len(all_trailing_tools) > 2:
        raise ValueError(f"{source['id']}: action replay requires one or two trailing calls")
    trailing_tools: list[dict[str, Any]] = []
    for message in all_trailing_tools:
        payload = json.loads(message["content"])
        if set(payload) != {"name", "arguments"} or payload["name"] != "exec_command":
            raise ValueError(f"{source['id']}: invalid tool protocol")
        if set(payload["arguments"]) != {"cmd"}:
            raise ValueError(f"{source['id']}: tool arguments are not cmd-only")
        if not action_command_is_label_relevant(
            labels[0], payload["arguments"]["cmd"]
        ):
            continue
        message["loss_scale"] = 0.05
        trailing_tools.append(message)
    if not trailing_tools:
        raise ValueError(f"{source['id']}: no label-relevant action remains")
    messages = messages[: len(messages) - len(all_trailing_tools)] + list(
        reversed(trailing_tools)
    )
    assistant_before_tools = [
        message
        for message in messages[: len(messages) - len(trailing_tools)]
        if message.get("role") == "assistant"
    ]
    if assistant_before_tools:
        assistant_before_tools[-1]["loss_scale"] = 0.25
    if len(assistant_before_tools) >= 2 and "<think>" in assistant_before_tools[-2].get("content", ""):
        assistant_before_tools[-2]["loss_scale"] = 0.35
    valid_calls = [json.loads(message["content"]) for message in reversed(trailing_tools)]
    commands = [call["arguments"]["cmd"] for call in valid_calls]
    if any(any(character in command for character in "*?[") for command in commands):
        raise ValueError(f"{source['id']}: selected action contains a path glob")
    metadata = common_counterfactual_metadata(
        source, "counterfactual_action_selection", "select_1_to_2_high_information_actions"
    )
    grounding_gate = source["metadata"].get("0809_action_visible_grounding_gate")
    if not grounding_gate or grounding_gate.get("checked") is not True:
        raise ValueError(f"{source['id']}: missing action visible-grounding gate")
    if grounding_gate.get("remaining_issues"):
        raise ValueError(
            f"{source['id']}: unresolved action grounding issues "
            f"{grounding_gate['remaining_issues']}"
        )
    metadata.update(
        {
            "actual_result_items": items,
            "correct_labels": labels,
            "selected_valid_tool_calls": valid_calls,
            "selected_action_commands": commands,
            "selected_action_count": len(commands),
            "selected_action_ordinal_within_query": ordinal,
            "selected_action_path_policy": "current_snapshot_direct_path_only_no_glob",
            "complete_binding_before_positive_tool_call": False,
            "action_visible_grounding_checked": True,
            "action_visible_grounding_remaining_issues": [],
            "rejected_action_categories": LABEL_ACTION_REJECTIONS[labels[0]],
            "protocol_hard_negatives_metadata_only": [
                {"name": "exec_command", "arguments": {"command": commands[0]}},
                {"name": "exec_command", "arguments": {"cmd": commands[0] + "\""}},
                {"tool": "exec_command", "cmd": commands[0]},
            ],
            "protocol_hard_negatives_supervised": False,
            "loss_policy": {
                "thinking": 0.35,
                "action_rationale": 0.25,
                "selected_tool_calls": 0.05,
                "history": 0.0,
                "tool_responses": "context_only",
            },
            "synthetic_assistant_control_marker_used": False,
        }
    )
    return {
        "id": f"0809_cf_action_q{case_id(source):04d}_a{ordinal:02d}",
        "tools": source["tools"],
        "messages": messages,
        "metadata": metadata,
    }


def build_stop_row(
    source: dict[str, Any], decision: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    labels, items = labels_and_items(decision)
    messages = unsupervised_context(source)
    evidence_text = rendered_evidence(binding)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    f"同一可见路径已经确认：{evidence_text}。"
                    f"这些原始回显满足“{labels[0]}”的完整证据门禁，继续跨设备或跨协议搜索不会改变当前最小集合，因此现在应停止调用工具。\n"
                    "</think>\n\n"
                ),
                "loss_scale": 0.45,
            },
            {
                "role": "assistant",
                "content": f"停止判断：{labels[0]}的决定性证据已经闭环，停止调用工具并进入最终回答。",
                "loss_scale": 0.6,
            },
        ]
    )
    metadata = common_counterfactual_metadata(
        source, "counterfactual_stop", "stop_at_first_complete_evidence_closure"
    )
    metadata.update(
        {
            "actual_result_items": items,
            "correct_labels": labels,
            "evidence_binding": binding,
            "rejected_continuation_categories": LABEL_ACTION_REJECTIONS[labels[0]],
            "rejected_continuation_is_training_target": False,
            "must_emit_no_tool_call": True,
            "loss_policy": {
                "thinking": 0.45,
                "stop_judgment": 0.6,
                "history": 0.0,
                "tool_calls": 0.0,
            },
        }
    )
    return {
        "id": f"0809_cf_stop_q{case_id(source):04d}",
        "tools": source["tools"],
        "messages": messages,
        "metadata": metadata,
    }


def build_label_boundary_row(
    evidence_source: dict[str, Any],
    decision: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    labels, items = labels_and_items(decision)
    label = labels[0]
    context = unsupervised_context(evidence_source)
    evidence_text = rendered_evidence(binding)
    rejected = LABEL_NEIGHBORS[label]
    context.extend(
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    f"当前直接证据为：{evidence_text}。{BOUNDARY_RULES[label]}"
                    f"因此保留标签“{label}”，并排除“{'、'.join(rejected)}”。\n"
                    "</think>\n\n"
                ),
                "loss_scale": 0.4,
            },
            {
                "role": "assistant",
                "content": (
                    f"标签边界判定：保留“{label}”；排除“{'、'.join(rejected)}”。"
                ),
                "loss_scale": 0.6,
            },
        ]
    )
    metadata = common_counterfactual_metadata(
        evidence_source,
        "counterfactual_label_boundary",
        "map_decisive_fact_to_label_and_reject_neighbors",
    )
    metadata.update(
        {
            "actual_result_items": items,
            "correct_labels": labels,
            "rejected_labels": rejected,
            "boundary_rule": BOUNDARY_RULES[label],
            "evidence_binding": binding,
            "loss_policy": {"thinking": 0.4, "boundary_decision": 0.6, "history": 0.0},
        }
    )
    return {
        "id": f"0809_cf_label_q{case_id(evidence_source):04d}",
        "tools": evidence_source["tools"],
        "messages": context,
        "metadata": metadata,
    }


def build_minimal_set_row(
    evidence_source: dict[str, Any],
    decision: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    labels, items = labels_and_items(decision)
    label = labels[0]
    context = unsupervised_context(evidence_source)
    rejected_label = LABEL_NEIGHBORS[label][0]
    if len(items) > 1:
        missing_member = items[:-1]
        extra_or_wrong = [*items, f"{result_device(items[0])};{rejected_label}"]
        hard_negatives = [missing_member, extra_or_wrong]
        negative_explanation = "缺少任一闭环成员会产生漏答；加入相邻标签会产生额外项。"
    else:
        hard_negatives = [[*items, f"{result_device(items[0])};{rejected_label}"]]
        negative_explanation = "加入没有直接决定性证据支持的相邻标签会产生额外项。"
    rendered_items = "、".join(f"“{item}”" for item in items)
    context.extend(
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    f"逐项核验后，只有{rendered_items}被当前决定性证据直接支持。"
                    f"{negative_explanation}因此必须保持完整且无多余项的最小集合。\n"
                    "</think>\n\n"
                ),
                "loss_scale": 0.4,
            },
            {
                "role": "assistant",
                "content": "最小集合判定：" + " | ".join(items),
                "loss_scale": 0.7,
            },
        ]
    )
    metadata = common_counterfactual_metadata(
        evidence_source,
        "counterfactual_minimal_set",
        "reject_false_positive_and_false_negative_result_sets",
    )
    metadata.update(
        {
            "actual_result_items": items,
            "correct_labels": labels,
            "positive_minimal_set": items,
            "hard_negative_sets_metadata_only": hard_negatives,
            "hard_negative_sets_supervised": False,
            "minimality_rule": negative_explanation,
            "evidence_binding": binding,
            "loss_policy": {"thinking": 0.4, "minimal_set": 0.7, "history": 0.0},
        }
    )
    return {
        "id": f"0809_cf_minset_q{case_id(evidence_source):04d}",
        "tools": evidence_source["tools"],
        "messages": context,
        "metadata": metadata,
    }


def build_evidence_elimination_row(
    evidence_source: dict[str, Any],
    decision: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    labels, items = labels_and_items(decision)
    label = labels[0]
    rejected = LABEL_NEIGHBORS[label][0]
    evidence_text = rendered_evidence(binding)
    context = unsupervised_context(evidence_source)
    context.extend(
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    f"可见回显“{evidence_text}”直接支持“{label}”。"
                    f"按照事实到标签的边界，候选“{rejected}”与该决定性事实不对应，应予排除。\n"
                    "</think>\n\n"
                ),
                "loss_scale": 0.35,
            },
            {
                "role": "assistant",
                "content": f"候选排除：排除“{rejected}”；绑定证据支持“{label}”。",
                "loss_scale": 0.5,
            },
        ]
    )
    metadata = common_counterfactual_metadata(
        evidence_source,
        "evidence_bound_elimination",
        "fill_case_level_elimination_coverage_without_final_answer_backfill",
    )
    metadata.update(
        {
            "actual_result_items": items,
            "correct_labels": labels,
            "rejected_labels": [rejected],
            "evidence_binding": binding,
            "elimination_derived_from_visible_source": False,
            "elimination_derived_from_verified_mapping": True,
            "derived_from_verified_final_answer": False,
            "direct_output_fact_gate_passed": True,
            "loss_policy": {"thinking": 0.35, "elimination": 0.5, "history": 0.0},
        }
    )
    return {
        "id": f"0809_cf_elimination_q{case_id(evidence_source):04d}",
        "tools": evidence_source["tools"],
        "messages": context,
        "metadata": metadata,
    }


def rebase_counterfactual_target(
    row: dict[str, Any], history_row: dict[str, Any], chain_position: int, history_policy: str
) -> dict[str, Any]:
    positive = [
        copy.deepcopy(message)
        for message in row["messages"]
        if float(message.get("loss_scale", 0) or 0) > 0
    ]
    if not positive:
        raise ValueError(f"{row['id']}: cannot chain a target without positive messages")
    row["messages"] = [*zero_history(history_row["messages"]), *positive]
    row["metadata"]["counterfactual_chain_position"] = chain_position
    row["metadata"]["counterfactual_chain_history_policy"] = history_policy
    return row


def choose_action_sources(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    endpoint_rows: list[dict[str, Any]],
    limit: int = 2,
    deduplicate_commands: bool = True,
) -> list[dict[str, Any]]:
    def commands(row: dict[str, Any]) -> tuple[str, ...]:
        trailing: list[dict[str, Any]] = []
        for message in reversed(row["messages"]):
            if message.get("role") != "tool_call":
                break
            trailing.append(message)
        return tuple(
            json.loads(message["content"])["arguments"]["cmd"]
            for message in reversed(trailing)
        )

    def commands_use_current_snapshot(row: dict[str, Any]) -> bool:
        snapshot = expected_snapshot(row["messages"])
        command_snapshots = {
            match
            for command in commands(row)
            for match in re.findall(r"saved_configs/([^/]+)/", command)
        }
        return command_snapshots == {snapshot}

    label = labels_and_items(decision)[0][0]

    def commands_are_label_relevant(row: dict[str, Any]) -> bool:
        return any(
            action_command_is_label_relevant(label, command)
            for command in commands(row)
        )

    candidates = [
        row
        for row in rows
        if row["metadata"]["target_type"] in {"planning", "reasoning"}
        and 1 <= int(row["metadata"].get("current_action_count", 0) or 0) <= 2
        and row["messages"][-1].get("role") == "tool_call"
        and not any(
            action.get("causal_semantics", {}).get("is_discovery")
            for action in row["metadata"].get("current_actions", [])
        )
        and not any(
            any(character in command for character in "*?[")
            for command in commands(row)
        )
        and commands_use_current_snapshot(row)
        and commands_are_label_relevant(row)
        # A positive tool call after complete decisive evidence is the exact
        # continuation failure seen in 0805/0807.  Such a state must become a
        # stop/final target, never an action replay.
        and not structural_action_binding_status(row, endpoint_rows)["complete"]
    ]
    if not candidates:
        raise ValueError(f"q{case_id(rows[0])}: no action source")

    def score(row: dict[str, Any]) -> tuple[float, int, int, str]:
        actions = row["metadata"].get("current_actions", [])
        priority = max(float(action.get("causal_priority", 0) or 0) for action in actions)
        discovery = sum(
            bool(action.get("causal_semantics", {}).get("is_discovery")) for action in actions
        )
        return (
            priority,
            -discovery,
            int(row["metadata"].get("step_index_in_compacted_path", 0) or 0),
            row["id"],
        )

    selected: list[dict[str, Any]] = []
    seen_commands: set[tuple[str, ...]] = set()
    for candidate in sorted(candidates, key=score, reverse=True):
        signature = commands(candidate)
        if deduplicate_commands and signature in seen_commands:
            continue
        selected.append(candidate)
        seen_commands.add(signature)
        if len(selected) == limit:
            break
    return selected


def build_strict_endpoint_group(
    evidence_source: dict[str, Any],
    decision: dict[str, Any],
    binding: dict[str, Any],
    origin: str,
    query_path_count: int,
) -> dict[str, Any]:
    """Build one reachable multi-objective bundle per real evidence path."""

    labels, items = labels_and_items(decision)
    label = labels[0]
    current_case = case_id(evidence_source)
    cluster_id = evidence_source["metadata"]["path_cluster_id"]
    path_tag = re.sub(r"[^a-zA-Z0-9]+", "_", cluster_id).strip("_")
    evidence_text = rendered_evidence(binding)
    neighbors = LABEL_NEIGHBORS[label]
    visible_commands: list[str] = []
    raw_context = unsupervised_context(evidence_source)
    for message in raw_context:
        if message.get("role") != "tool_call":
            continue
        command = json.loads(message["content"])["arguments"]["cmd"]
        if command not in visible_commands:
            visible_commands.append(command)
    raw_prefix_sha256 = digest_text(
        "\n".join(message.get("content", "") for message in raw_context)
    )
    path_weight = 1.0 / query_path_count

    common = rewrite_parent_row(evidence_source)["metadata"]
    common.update(
        {
            "dataset_type": "0809_strict_evidence_endpoint",
            "release_id": "0809",
            "release_data_root": "data/2026-08-09",
            "parent_release": "0805",
            "training_sampling_role": "strict_endpoint_pool",
            "strict_endpoint_gate_passed": True,
            "strict_endpoint_gate_version": "direct_evidence_binding.v5",
            "evidence_binding": binding,
            "actual_result_items": items,
            "correct_labels": labels,
            "rejected_labels": [],
            "unverified_neighbor_labels": neighbors,
            "candidate_negative_evidence_bindings": {},
            "candidate_negative_evidence_complete": False,
            "hypothesis_elimination_supervised": False,
            "candidate_scope_policy": "positive evidence confirms the current label but cannot prove an unchecked neighbor absent; unverified neighbors stay outside the current confirmed set without being called false",
            "endpoint_schedule_epoch": None,
            "endpoint_schedule_slot": None,
            "endpoint_schedule_role": None,
            "endpoint_chain_variant": "unified_reachable_bundle",
            "endpoint_group_id": cluster_id,
            "raw_evidence_path_id": f"q{current_case:04d}:{cluster_id}",
            "raw_first_positive_prefix_sha256": raw_prefix_sha256,
            "raw_evidence_path_origin": origin,
            "counts_as_real_evidence_path": True,
            "query_raw_evidence_path_count": query_path_count,
            "query_normalized_path_weight": round(path_weight, 8),
            "sampling_source_row_id": None,
            "synthetic_stage": True,
            "review_status": "auto_rebuilt_literal_evidence_bound_requires_domain_review",
            "synthetic_assistant_control_marker_used": False,
        }
    )

    messages = raw_context
    messages.extend(
        [
            {
                "role": "assistant",
                "content": (
                    "<think>\n"
                    f"决定性原始回显为：{evidence_text}。这些事实逐设备满足“{label}”的完整证据门禁。"
                    f"{BOUNDARY_RULES[label]}但该正证据本身不能推出相邻候选“{'、'.join(neighbors)}”不存在；"
                    "本路径没有这些候选各自的专属反证，所以它们保持未验证。"
                    f"当前已证实集合为{' | '.join(items)}，其中每个成员都有完整正证据；"
                    "未验证标签不加入当前集合，也不表述为不存在。该集合与已验证结果一致，停止工具调用并回答。\n"
                    "</think>\n\n"
                ),
                "loss_scale": round(0.2 * path_weight, 8),
            },
            {
                "role": "assistant",
                "content": (
                    f"证据归纳：{evidence_text}。该组事实支持“{label}”；"
                    f"相邻候选“{'、'.join(neighbors)}”缺少候选专属反证，保持未验证。"
                    f"当前已证实集合为{' | '.join(items)}；只对该集合完成证据闭环，停止调用工具。"
                ),
                "loss_scale": round(0.3 * path_weight, 8),
            },
            {
                "role": "assistant",
                "content": result_block(items),
                "loss_scale": round(1.0 * path_weight, 8),
            },
        ]
    )
    common.update(
        {
            "target_type": "endpoint_bundle",
            "endpoint_objectives": [
                "evidence_summary",
                "label_boundary",
                "candidate_scope_calibration",
                "minimal_set",
                "decision_ready",
                "decision",
            ],
            "endpoint_chain_position": 1,
            "endpoint_chain_history_policy": "single_reachable_completion_from_raw_context",
            "must_emit_no_tool_call": True,
            "positive_minimal_set": items,
            "minimal_set_scope": "current_evidence_confirmed_set_not_global_negative_proof",
            "loss_policy": {
                "thinking_evidence_boundary_minimal_stop": round(0.2 * path_weight, 8),
                "grounded_summary_and_stop": round(0.3 * path_weight, 8),
                "verified_final_answer": round(1.0 * path_weight, 8),
                "query_path_normalization": "1 / real evidence path count for this query",
                "history": 0.0,
            },
        }
    )
    return {
        "id": f"0809_endpoint_q{current_case:04d}_{path_tag}_bundle",
        "tools": evidence_source["tools"],
        "messages": messages,
        "metadata": common,
    }


def recover_same_query_endpoint_rows(case_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    """Rebuild endpoints from the frozen 0805 success events, never dev runs.

    The 0807 recovery implementation is reused as code only.  Its input here is
    the 0805 accepted-trajectory index and the exact 0805 raw/events paths, so
    0809 remains parented from 0805.  Recovery may combine real read-only facts
    across the ten successful trajectories of one query and one snapshot.
    """

    source = load_json(PARENT_ROOT / "curation" / "accepted_trajectory_selection.json")
    annotations_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source["trajectories"]:
        current_case = int(annotation["case_id"])
        if annotation.get("selected") and current_case in case_ids:
            annotations_by_case[current_case].append(annotation)
    recovered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    event_items_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def normalized_span(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def event_items(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
        event_file = str(annotation["events_file"])
        if event_file in event_items_cache:
            return event_items_cache[event_file]
        path = ROOT / event_file
        actual_hash = digest_file(path)
        expected_hash = str(annotation["events_sha256_lf_normalized"])
        if actual_hash != expected_hash:
            raise ValueError(f"stale 0805 source event hash: {event_file}")
        items: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "command_execution":
                items[str(item["id"])] = item
        event_items_cache[event_file] = items
        return items

    def fact_provenance(
        action_id: str,
        observation: str,
        annotations_by_trajectory: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        match = re.fullmatch(r"REC-(q\d{4}_success_\d+)-(item_\d+)", action_id)
        if not match:
            raise ValueError(f"unsupported recovery action id: {action_id}")
        trajectory_id, item_id = match.groups()
        annotation = annotations_by_trajectory.get(trajectory_id)
        if annotation is None:
            raise ValueError(f"{action_id}: trajectory is absent from frozen 0805 selection")
        item = event_items(annotation).get(item_id)
        if item is None:
            raise ValueError(f"{action_id}: item is absent from {annotation['events_file']}")
        if int(item.get("exit_code", 1)) != 0 or item.get("status") != "completed":
            raise ValueError(f"{action_id}: source action did not complete successfully")
        raw_output = str(item.get("aggregated_output") or "")
        normalized_observation = normalized_span(observation)
        matching_lines = [
            line
            for line in raw_output.splitlines()
            if normalized_span(line) == normalized_observation
        ]
        if not matching_lines:
            raise ValueError(f"{action_id}: normalized observation is absent from raw output")
        source_command = str(item.get("command") or "")
        return {
            "action_id": action_id,
            "source_event_file": str(annotation["events_file"]),
            "source_event_sha256_lf_normalized": str(
                annotation["events_sha256_lf_normalized"]
            ),
            "item_id": item_id,
            "source_command": source_command,
            "source_command_sha256_lf_normalized": digest_text(
                source_command.replace("\r\n", "\n").replace("\r", "\n")
            ),
            "raw_output_sha256_lf_normalized": digest_text(
                raw_output.replace("\r\n", "\n").replace("\r", "\n")
            ),
            "raw_output_line_or_span": matching_lines[0],
            "normalized_observation": normalized_observation,
            "normalization_policy": "collapse_whitespace_to_single_ascii_space_and_strip",
            "source_exit_code": int(item["exit_code"]),
            "source_status": str(item["status"]),
        }
    for current_case in sorted(case_ids):
        annotations = annotations_by_case[current_case]
        if len(annotations) != 10:
            raise ValueError(
                f"q{current_case}: expected ten frozen 0805 successful trajectories, found {len(annotations)}"
            )
        rows, _ = recovery.process_case(current_case, annotations)
        annotations_by_trajectory = {
            str(load_json(ROOT / annotation["raw_file"])["id"]): annotation
            for annotation in annotations
        }
        for row in rows:
            if row["metadata"]["target_type"] != "endpoint_bundle":
                continue
            cloned = suppress_unsafe_parent_training_tool_calls(copy.deepcopy(row))
            cloned["metadata"]["path_cluster_id"] = (
                "0805_same_query_recovery_" + str(row["metadata"]["path_cluster_id"])
            )
            selected_facts = cloned["metadata"]["endpoint_evidence_gate"][
                "selected_facts"
            ]
            provenance_records = [
                fact_provenance(
                    str(fact["action_id"]),
                    str(fact["observation"]),
                    annotations_by_trajectory,
                )
                for fact in selected_facts
            ]
            carrier_event_file = cloned["metadata"].get("source_event_file")
            source_event_files = sorted(
                {record["source_event_file"] for record in provenance_records}
            )
            cloned["metadata"]["carrier_bundle_source_event_file"] = carrier_event_file
            cloned["metadata"]["source_event_file"] = None
            cloned["metadata"]["source_event_sha256_lf_normalized"] = None
            cloned["metadata"]["source_event_files"] = source_event_files
            cloned["metadata"]["0809_recovery_source"] = {
                "release": "0805",
                "policy": "same-query same-snapshot successful-event recovery",
                "carrier_bundle_source_event_file": carrier_event_file,
                "source_event_files": source_event_files,
                "fact_action_provenance": provenance_records,
                "cross_trajectory_evidence_recovery": bool(
                    row["metadata"].get("cross_trajectory_evidence_recovery")
                ),
            }
            recovered[current_case].append(cloned)
    return recovered


def attach_recovery_fact_provenance(
    binding: dict[str, Any], evidence_source: dict[str, Any]
) -> dict[str, Any]:
    recovery_metadata = evidence_source["metadata"].get("0809_recovery_source")
    if not recovery_metadata:
        return binding
    available = list(recovery_metadata["fact_action_provenance"])
    for fact in binding["facts"]:
        normalized = re.sub(r"\s+", " ", str(fact["output_line"])).strip()
        match_index = next(
            (
                index
                for index, record in enumerate(available)
                if record["normalized_observation"] == normalized
            ),
            None,
        )
        if match_index is None:
            raise ValueError(
                f"{evidence_source['id']}: no raw-event provenance for {fact['output_line']!r}"
            )
        fact["source_action_provenance"] = available.pop(match_index)
    binding["provenance_policy"] = (
        "each normalized atomic fact independently resolves REC action -> frozen 0805 "
        "event file -> completed item -> raw output span"
    )
    binding["sha256"] = digest_text(
        json.dumps(binding["facts"], ensure_ascii=False, separators=(",", ":"))
    )
    return binding


def build_strict_endpoint_pool(
    parent_endpoint_rows: list[dict[str, Any]], train_case_ids: set[int]
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_endpoint_rows:
        grouped[(case_id(row), row["metadata"]["path_cluster_id"])].append(row)

    admitted_sources: dict[
        int, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]]
    ] = defaultdict(list)
    failed: list[dict[str, Any]] = []
    for (current_case, cluster_id), rows in sorted(grouped.items()):
        typed = {row["metadata"]["target_type"]: row for row in rows}
        if set(typed) != {"evidence_summary", "decision_ready", "decision"}:
            raise ValueError(f"q{current_case} {cluster_id}: malformed inherited endpoint group")
        binding = direct_evidence_binding(typed["evidence_summary"], typed["decision"])
        if binding is None:
            failed.append(
                {
                    "case_id": current_case,
                    "path_cluster_id": cluster_id,
                    "source_row_ids": sorted(row["id"] for row in rows),
                    "reason": "direct_evidence_binding_failed",
                }
            )
            continue
        admitted_sources[current_case].append(
            (typed["evidence_summary"], typed["decision"], binding, "inherited_0805")
        )

    inherited_pass_count = sum(len(sources) for sources in admitted_sources.values())
    missing_cases = train_case_ids - set(admitted_sources)
    recovered_by_case = recover_same_query_endpoint_rows(missing_cases)
    recovery_failures: list[int] = []
    for current_case in sorted(missing_cases):
        for recovered in recovered_by_case.get(current_case, []):
            binding = direct_evidence_binding(recovered, recovered)
            if binding is not None:
                binding = attach_recovery_fact_provenance(binding, recovered)
                admitted_sources[current_case].append(
                    (recovered, recovered, binding, "0805_same_query_recovery")
                )
        if not admitted_sources[current_case]:
            recovery_failures.append(current_case)
    if recovery_failures:
        raise ValueError(
            "strict same-query endpoint recovery failed for cases "
            + ",".join(map(str, recovery_failures))
        )

    strict_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_source_records: list[dict[str, Any]] = []
    for current_case in sorted(train_case_ids):
        sources = sorted(
            admitted_sources[current_case],
            key=lambda item: (
                item[3] != "inherited_0805",
                item[2]["sha256"],
                item[0]["id"],
            ),
        )
        for source in sources:
            evidence_source, decision, binding, origin = source
            strict_by_case[current_case].append(
                build_strict_endpoint_group(
                    evidence_source, decision, binding, origin, len(sources)
                )
            )
            selected_source_records.append(
                {
                    "case_id": current_case,
                    "objective": "unified_reachable_bundle",
                    "origin": origin,
                    "source_row_id": evidence_source["id"],
                    "binding_sha256": binding["sha256"],
                    "source_reused_for_distinct_objective": False,
                }
            )

    pool = [
        row
        for current_case in sorted(strict_by_case)
        for row in strict_by_case[current_case]
    ]
    audit = {
        "inherited_groups": len(grouped),
        "inherited_groups_passed_strict_gate": inherited_pass_count,
        "strict_groups": sum(len(groups) for groups in strict_by_case.values()),
        "removed_groups": len(failed),
        "strict_case_count": len(strict_by_case),
        "same_query_recovery_case_count": len(missing_cases),
        "same_query_recovery_case_ids": sorted(missing_cases),
        "objective_group_count": sum(len(groups) for groups in strict_by_case.values()),
        "unique_raw_evidence_path_count": sum(len(groups) for groups in strict_by_case.values()),
        "minimum_real_evidence_paths_per_case": 1,
        "single_evidence_path_case_ids": sorted(
            current_case
            for current_case, rows in strict_by_case.items()
            if len(rows) == 1
        ),
        "selected_source_records": selected_source_records,
        "removed_group_records": failed,
    }
    return pool, strict_by_case, audit


def repair_post_closure_parent_row(
    row: dict[str, Any], decision: dict[str, Any], endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Remove inherited continuation targets once decisive evidence is complete."""

    status = structural_action_binding_status(row, endpoint_rows)
    binding = status["binding"] if status["complete"] else None
    positive_tools = [
        message
        for message in row["messages"]
        if message.get("role") == "tool_call"
        and float(message.get("loss_scale", 0) or 0) > 0
    ]
    metadata = row["metadata"]
    if binding is None or not positive_tools:
        metadata["0809_post_closure_gate"] = {
            "complete_binding_before_positive_tool_call": bool(binding),
            "structural_gate_checked": True,
            "best_binding_sha256": status["binding"]["sha256"],
            "expected_fact_count": len(status["binding"]["facts"]),
            "matched_fact_count": len(status["matched_facts"]),
            "repaired": False,
        }
        return row

    labels, items = labels_and_items(decision)
    label = labels[0]
    source_target_type = metadata["target_type"]
    context = unsupervised_context(row)
    if source_target_type == "hypothesis_elimination":
        # Preserve the original evidence-based elimination language but remove
        # every subsequent call, then explicitly terminate the investigation.
        context.extend(
            copy.deepcopy(message)
            for message in row["messages"]
            if message.get("role") == "assistant"
            and float(message.get("loss_scale", 0) or 0) > 0
        )
        context.append(
            {
                "role": "assistant",
                "content": (
                    f"停止判断：{rendered_evidence(binding)}已完整支持“{label}”；"
                    "当前候选排除完成后无需再调用工具，进入最终回答。"
                ),
                "loss_scale": 0.6,
            }
        )
    else:
        context.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        "<think>\n"
                        f"当前历史已经包含完整决定性证据：{rendered_evidence(binding)}。"
                        f"这些事实满足“{label}”的全部设备/闭环条件，继续执行原计划中的工具调用只会扩大搜索，"
                        "不会改变最小根因集合，因此应立即停止。\n"
                        "</think>\n\n"
                    ),
                    "loss_scale": 0.5,
                },
                {
                    "role": "assistant",
                    "content": f"停止判断：{label}的决定性证据已经闭环，取消后续调用并进入最终回答。",
                    "loss_scale": 0.65,
                },
            ]
        )
        metadata["target_type"] = "post_closure_stop_repair"

    metadata.update(
        {
            "actual_result_items": items,
            "correct_labels": labels,
            "source_target_type_before_post_closure_repair": source_target_type,
            "evidence_binding": binding,
            "must_emit_no_tool_call": True,
            "0809_post_closure_gate": {
                "complete_binding_before_positive_tool_call": True,
                "structural_gate_checked": True,
                "best_binding_sha256": status["binding"]["sha256"],
                "expected_fact_count": len(status["binding"]["facts"]),
                "matched_fact_count": len(status["matched_facts"]),
                "repaired": True,
                "removed_positive_tool_call_count": len(positive_tools),
                "removed_commands": [
                    json.loads(message["content"])["arguments"]["cmd"]
                    for message in positive_tools
                ],
                "policy": "recompute complete label-family closure from strict endpoint facts before first positive loss; preserve useful elimination when present, otherwise replace continuation reasoning; remove positive tools and append evidence-specific stop",
            },
        }
    )
    row["messages"] = context
    return row


def selected_action_sources_by_case(
    train_rows: list[dict[str, Any]],
    base_core_rows: list[dict[str, Any]],
    strict_endpoint_by_case: dict[int, list[dict[str, Any]]],
) -> dict[int, list[str]]:
    def message_signature(row: dict[str, Any]) -> str:
        current_case = case_id(row)
        simulated = repair_action_visible_grounding_parent_row(
            copy.deepcopy(row),
            decision_for_case(rows_by_case[current_case]),
            strict_endpoint_by_case[current_case],
        )
        prospective = build_action_row(
            simulated,
            decision_for_case(rows_by_case[current_case]),
            1,
        )["messages"]
        return digest_text(
            json.dumps(
                prospective, ensure_ascii=False, separators=(",", ":")
            )
        )

    rows_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    core_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        rows_by_case[case_id(row)].append(row)
    for row in base_core_rows:
        core_by_case[case_id(row)].append(row)
    selected_by_case: dict[int, list[str]] = {}
    selected: set[str] = set()
    selected_message_signatures: set[str] = set()
    for current_case in sorted(rows_by_case):
        decision = decision_for_case(rows_by_case[current_case])
        rows = choose_action_sources(
            core_by_case[current_case],
            decision,
            strict_endpoint_by_case[current_case],
            limit=99,
        )
        selected_by_case[current_case] = []
        for row in rows:
            if row["id"] in selected:
                raise ValueError(f"duplicate selected action source: {row['id']}")
            signature = message_signature(row)
            if signature in selected_message_signatures:
                continue
            selected.add(row["id"])
            selected_message_signatures.add(signature)
            selected_by_case[current_case].append(row["id"])

    # q23 loses the audited post-closure continuation.  Preserve the fixed
    # 308-row/144-step protocol with another real, incomplete q23 context,
    # even when its command signature duplicates a different visible prefix.
    fill_order = [23, *[current for current in sorted(rows_by_case) if current != 23]]
    for fill_case in fill_order:
        if len(selected) >= 308:
            break
        decision = decision_for_case(rows_by_case[fill_case])
        expanded = choose_action_sources(
            core_by_case[fill_case],
            decision,
            strict_endpoint_by_case[fill_case],
            limit=99,
            deduplicate_commands=False,
        )
        for row in expanded:
            if row["id"] in selected:
                continue
            signature = message_signature(row)
            if signature in selected_message_signatures:
                continue
            selected.add(row["id"])
            selected_message_signatures.add(signature)
            selected_by_case[fill_case].append(row["id"])
            if len(selected) == 308:
                break
    if len(selected) != 308:
        raise ValueError(
            f"structural action selection expected 308 incomplete contexts, found {len(selected)}"
        )
    return selected_by_case


def build_counterfactual_rows(
    train_rows: list[dict[str, Any]],
    base_core_rows: list[dict[str, Any]],
    strict_endpoint_by_case: dict[int, list[dict[str, Any]]],
    selected_action_ids_by_case: dict[int, list[str]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    set[int],
    set[int],
    set[int],
    dict[str, int],
]:
    """Select every distinct high-information reachable action context.

    Boundary, exclusion, minimal-set, stop and final supervision now live in
    the reachable endpoint bundle for each real evidence path.  Keeping a
    second counterfactual chain at the same raw prefix would recreate the
    off-policy/multimodal condition identified by the v4 audit.
    """
    rows_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    core_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        rows_by_case[case_id(row)].append(row)
    for row in base_core_rows:
        core_by_case[case_id(row)].append(row)
    train_cases = set(rows_by_case)
    if train_cases & FROZEN_VALIDATION_CASES:
        raise ValueError("validation cases leaked into the parent train pool")
    if len(train_cases) != 72:
        raise ValueError(f"expected 72 train cases, found {len(train_cases)}")

    real_elimination_cases = {
        case_id(row)
        for row in base_core_rows
        if row["metadata"]["target_type"] == "hypothesis_elimination"
    }
    missing_elimination_cases = train_cases - real_elimination_cases
    counterfactual_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    direct_evidence_cases: set[int] = set()
    direct_evidence_labels: Counter[str] = Counter()
    for current_case in sorted(train_cases):
        case_rows = rows_by_case[current_case]
        decision = decision_for_case(case_rows)
        package = evidence_package(case_rows)
        core_by_id = {row["id"]: row for row in core_by_case[current_case]}
        actions = [
            core_by_id[row_id]
            for row_id in selected_action_ids_by_case[current_case]
        ]
        counterfactual_by_case[current_case].extend(
            build_action_row(action, decision, ordinal)
            for ordinal, action in enumerate(actions, start=1)
        )
        if package is None:
            continue
        _, evidence_decision, _ = package
        direct_evidence_cases.add(current_case)
        label = labels_and_items(evidence_decision)[0][0]
        direct_evidence_labels[label] += 1
    elimination_sources_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_core_rows:
        if row["metadata"]["target_type"] != "hypothesis_elimination":
            continue
        decision = decision_for_case(rows_by_case[case_id(row)])
        label = labels_and_items(decision)[0][0]
        elimination_sources_by_label[label].append(row)
    replay_sources = [
        sorted(rows, key=lambda row: row["id"])[0]
        for _, rows in sorted(elimination_sources_by_label.items())
    ]
    for label in ("存在IP路由环路", "存在MPLS标签环路"):
        replay_sources.append(
            sorted(elimination_sources_by_label[label], key=lambda row: row["id"])[1]
        )
    if len(replay_sources) != 8 or len({row["id"] for row in replay_sources}) != 8:
        raise ValueError("elimination replay selection must contain eight distinct sources")
    for ordinal, source in enumerate(sorted(replay_sources, key=lambda row: row["id"]), start=1):
        replay = copy.deepcopy(source)
        replay["id"] = f"0809_elimination_replay_{ordinal:02d}_{source['id']}"
        replay["metadata"].update(
            {
                "dataset_type": "0809_real_elimination_sampling_replay",
                "target_type": "hypothesis_elimination_replay",
                "training_sampling_role": "explicit_same_target_budget_replay",
                "sampling_replay": True,
                "counts_as_new_semantic_target": False,
                "sampling_source_row_id": source["id"],
                "synthetic_assistant_control_marker_used": False,
            }
        )
        counterfactual_by_case[case_id(source)].append(replay)
    all_rows = [
        row
        for current_case in sorted(counterfactual_by_case)
        for row in counterfactual_by_case[current_case]
    ]
    if len(all_rows) != 316:
        raise ValueError(f"expected 308 actions + 8 elimination replays, found {len(all_rows)}")
    generated_elimination_cases = missing_elimination_cases & direct_evidence_cases
    return (
        all_rows,
        counterfactual_by_case,
        missing_elimination_cases,
        generated_elimination_cases,
        direct_evidence_cases,
        dict(sorted(direct_evidence_labels.items())),
    )


def build_core_schedules(
    base_core_rows: list[dict[str, Any]],
    counterfactual_by_case: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    counterfactual_rows = [
        row
        for current_case in sorted(counterfactual_by_case)
        for row in counterfactual_by_case[current_case]
    ]
    if len(base_core_rows) != 719 or len(counterfactual_rows) != 316:
        raise ValueError("core pool must be 719 base + 308 actions + 8 elimination replays")
    schedules: dict[int, list[dict[str, Any]]] = {}
    for epoch in range(1, EPOCHS + 1):
        scheduled = copy.deepcopy([*base_core_rows, *counterfactual_rows])
        for row in scheduled:
            if row["metadata"]["target_type"] in {
                "counterfactual_action_selection",
                "hypothesis_elimination_replay",
            }:
                row["metadata"]["counterfactual_schedule_epoch"] = epoch
                row["metadata"]["counterfactual_schedule_slot"] = row["id"]
        if len(scheduled) != 1035:
            raise ValueError(f"epoch {epoch}: core schedule has {len(scheduled)} rows")
        schedules[epoch] = sorted(scheduled, key=lambda row: row["id"])
    return schedules


def build_endpoint_schedules(
    strict_by_case: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Expose every one of the 116 real evidence paths exactly once."""

    source_rows = [
        row
        for current_case in sorted(strict_by_case)
        for row in strict_by_case[current_case]
    ]
    if len(source_rows) != 116:
        raise ValueError("endpoint schedule requires 116 real paths")
    schedules: dict[int, list[dict[str, Any]]] = {}
    for epoch in range(1, EPOCHS + 1):
        scheduled = copy.deepcopy(source_rows)
        slots_by_case: Counter[int] = Counter()
        for row in scheduled:
            current_case = case_id(row)
            slots_by_case[current_case] += 1
            row["metadata"].update(
                {
                    "endpoint_schedule_epoch": epoch,
                    "endpoint_schedule_slot": slots_by_case[current_case],
                    "endpoint_schedule_role": "real_evidence_path_bundle",
                }
            )
        if len(scheduled) != 116:
            raise ValueError(f"epoch {epoch}: endpoint schedule has {len(scheduled)} rows")
        if len({row["id"] for row in scheduled}) != len(scheduled):
            raise ValueError(f"epoch {epoch}: endpoint schedule contains duplicate row ids")
        schedules[epoch] = sorted(scheduled, key=lambda row: row["id"])
    return schedules


def estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    return cjk + math.ceil(max(0, len(text) - cjk) / 4)


def training_signal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weighted = 0.0
    raw = 0
    by_target: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    target_weight: Counter[str] = Counter()
    for row in rows:
        target = row["metadata"]["target_type"]
        by_target[target] += 1
        for message in row["messages"]:
            scale = float(message.get("loss_scale", 0) or 0)
            if scale <= 0:
                continue
            tokens = estimate_tokens(message.get("content", ""))
            raw += tokens
            contribution = tokens * scale
            weighted += contribution
            target_weight[target] += contribution
            role = message.get("role")
            if role == "tool_call":
                kind = "tool_call"
            elif "<think>" in message.get("content", ""):
                kind = "thinking"
            else:
                kind = "conclusion_or_answer"
            by_kind[kind] += contribution
    return {
        "method": "one CJK code point is approximately one token; other text is approximately four characters per token",
        "row_count": len(rows),
        "target_type_rows": dict(sorted(by_target.items())),
        "raw_target_token_estimate": raw,
        "weighted_token_estimate": round(weighted, 4),
        "weighted_message_kind_percent": {
            key: round(100 * value / weighted, 6) for key, value in sorted(by_kind.items())
        },
        "weighted_target_type_percent": {
            key: round(100 * value / weighted, 6)
            for key, value in sorted(target_weight.items())
        },
    }


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty_at_generation": bool(run("status", "--porcelain")),
    }


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    raw_root = data_root / "raw"
    parent_raw = PARENT_ROOT / "raw"
    if not raw_root.is_dir():
        raise ValueError(f"0809 raw archive is missing: {raw_root}")
    parent_raw_tree = raw_tree(parent_raw)
    release_raw_tree = raw_tree(raw_root)
    if parent_raw_tree != release_raw_tree:
        raise ValueError("0809 raw archive differs from its frozen 0805 parent")

    parent_train_rows = load_jsonl(PARENT_TRAIN)
    parent_core_rows = load_jsonl(PARENT_CORE)
    parent_endpoint_rows = [
        suppress_unsafe_parent_training_tool_calls(rewrite_parent_row(row))
        for row in load_jsonl(PARENT_ENDPOINT_POOL)
    ]
    parent_validation_rows = load_jsonl(PARENT_VALIDATION)
    train_rows = [
        suppress_unsafe_parent_training_tool_calls(rewrite_parent_row(row))
        for row in parent_train_rows
    ]
    base_core_rows = [
        suppress_unsafe_parent_training_tool_calls(rewrite_parent_row(row))
        for row in parent_core_rows
    ]
    decision_by_case_cluster = {
        (case_id(row), row["metadata"]["path_cluster_id"]): row
        for row in train_rows
        if row["metadata"]["target_type"] == "decision"
    }
    validation_rows = [rewrite_parent_row(row) for row in parent_validation_rows]
    train_case_ids = {case_id(row) for row in train_rows}
    if {case_id(row) for row in validation_rows} != FROZEN_VALIDATION_CASES:
        raise ValueError("0809 validation split differs from the frozen 0804/0805 split")
    endpoint_pool_rows, strict_endpoint_by_case, endpoint_gate_audit = (
        build_strict_endpoint_pool(parent_endpoint_rows, train_case_ids)
    )
    base_core_rows = [
        repair_post_closure_parent_row(
            row,
            decision_by_case_cluster[
                (case_id(row), row["metadata"]["path_cluster_id"])
            ],
            strict_endpoint_by_case[case_id(row)],
        )
        for row in base_core_rows
    ]
    selected_action_ids_by_case = selected_action_sources_by_case(
        train_rows, base_core_rows, strict_endpoint_by_case
    )
    selected_action_ids = {
        row_id for rows in selected_action_ids_by_case.values() for row_id in rows
    }
    base_core_rows = [
        repair_action_visible_grounding_parent_row(
            row,
            decision_by_case_cluster[
                (case_id(row), row["metadata"]["path_cluster_id"])
            ],
            strict_endpoint_by_case[case_id(row)],
        )
        if row["id"] in selected_action_ids
        else row
        for row in base_core_rows
    ]
    (
        counterfactual_rows,
        counterfactual_by_case,
        missing_elimination_cases,
        generated_elimination_cases,
        direct_evidence_cases,
        direct_evidence_labels,
    ) = build_counterfactual_rows(
        train_rows,
        base_core_rows,
        strict_endpoint_by_case,
        selected_action_ids_by_case,
    )
    core_schedules = build_core_schedules(base_core_rows, counterfactual_by_case)
    endpoint_schedules = build_endpoint_schedules(strict_endpoint_by_case)

    # Failed inherited endpoint groups are excluded from both the semantic pool
    # and schedules; keeping them only out of the schedule would still make the
    # release pool falsely advertise unsafe teacher-forced conclusions.
    semantic_rows = [*base_core_rows, *endpoint_pool_rows, *counterfactual_rows]
    sft = data_root / "sft"
    sft.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_semantic_pool": sft / "qwen3_6_27b_0809_train_semantic_pool.jsonl",
        "base_core_pool": sft / "qwen3_6_27b_0809_base_core_pool.jsonl",
        "counterfactual_pool": sft / "qwen3_6_27b_0809_counterfactual_pool.jsonl",
        "endpoint_pool": sft / "qwen3_6_27b_0809_endpoint_pool.jsonl",
        "validation": sft / "qwen3_6_27b_0809_validation.jsonl",
        "manifest": sft / "0809_agent_error_aware_manifest.json",
        "release_record": sft / "RELEASE_RECORD.json",
        "preflight": sft / "TARGET_TOKENIZER_PREFLIGHT.json",
    }
    outputs: dict[str, dict[str, Any]] = {
        "train_semantic_pool": write_jsonl(paths["train_semantic_pool"], semantic_rows),
        "base_core_pool": write_jsonl(paths["base_core_pool"], base_core_rows),
        "counterfactual_pool": write_jsonl(paths["counterfactual_pool"], counterfactual_rows),
        "endpoint_pool": write_jsonl(paths["endpoint_pool"], endpoint_pool_rows),
        "validation": write_jsonl(paths["validation"], validation_rows),
    }
    for epoch in range(1, EPOCHS + 1):
        core_path = sft / f"qwen3_6_27b_0809_core_epoch_{epoch:02d}.jsonl"
        endpoint_path = sft / f"qwen3_6_27b_0809_endpoint_epoch_{epoch:02d}.jsonl"
        outputs[f"core_epoch_{epoch:02d}"] = write_jsonl(
            core_path, core_schedules[epoch]
        )
        outputs[f"endpoint_epoch_{epoch:02d}"] = write_jsonl(
            endpoint_path, endpoint_schedules[epoch]
        )

    curation = data_root / "curation"
    parent_selection = load_json(PARENT_ROOT / "curation" / "accepted_trajectory_selection.json")
    accepted_selection = rewrite_identity(parent_selection)
    accepted_selection["schema_version"] = "0809-accepted-trajectory-selection-parented-from-0805.v1"
    accepted_selection["parent_release"] = "data/2026-08-05"
    accepted_selection["release_scope"] = "data/2026-08-09"
    write_json(curation / "accepted_trajectory_selection.json", accepted_selection)

    parent_clusters = load_json(PARENT_ROOT / "curation" / "causal_path_clusters_per_case.json")
    clusters = rewrite_identity(parent_clusters)
    clusters["parent_schema_version"] = parent_clusters.get("schema_version")
    clusters["schema_version"] = "0809-agent-error-aware-parent-cluster-selection.v1"
    clusters["parent_release"] = "data/2026-08-05"
    clusters["release_scope"] = "data/2026-08-09"
    clusters["counterfactual_policy"] = {
        "source": "frozen 72-case train split only",
        "agent_evaluation_trajectories_used": False,
        "validation_cases_used": False,
        "objectives_per_train_case": [
            "counterfactual_action_selection",
            "direct-output-gated counterfactual_stop when complete facts are visible",
            "direct-output-gated counterfactual_label_boundary when complete facts are visible",
            "direct-output-gated counterfactual_minimal_set when complete facts are visible",
        ],
        "missing_real_elimination_cases": sorted(missing_elimination_cases),
        "evidence_bound_elimination_generated_cases": sorted(
            generated_elimination_cases
        ),
        "synthetic_endpoint_eligible_cases": sorted(direct_evidence_cases),
        "synthetic_endpoint_eligibility_by_label": direct_evidence_labels,
        "omission_rule": "omit synthetic stop/boundary/minimal/elimination when the same visible path lacks label-specific direct output closure",
    }
    write_json(curation / "causal_path_clusters_per_case.json", clusters)
    write_json(curation / "endpoint_gate_audit.json", endpoint_gate_audit)

    parent_filter = (PARENT_ROOT / "curation" / "FILTER_REPORT.md").read_text(
        encoding="utf-8"
    )
    (curation / "FILTER_REPORT.md").write_text(
        "# 0809 父级轨迹过滤记录\n\n"
        "本文件继承 0805 的严格成功轨迹过滤结果；0809 不重新接纳失败轨迹，"
        "仅在冻结训练题上构造证据绑定的反事实监督。\n\n"
        + parent_filter.replace("2026-08-05", "2026-08-09").replace("0805", "0809"),
        encoding="utf-8",
        newline="\n",
    )

    tracked_paths = {
        "root_readme": ROOT / "README.md",
        "release_readme": data_root / "README.md",
        "reproducibility": data_root / "REPRODUCIBILITY.md",
        "design_audit_input": DESIGN_AUDIT_REPORT,
        "converter": ROOT / "scripts" / "convert_0809_agent_error_aware_sft.py",
        "recovery_dependency": ROOT / "scripts" / "convert_0807_evidence_gated_reasoning_sft.py",
        "validator": ROOT / "scripts" / "validate_0809_agent_error_aware_sft.py",
        "independent_validator": ROOT / "scripts" / "independent_validate_0809_sft.py",
        "audit": ROOT / "scripts" / "audit_0809_agent_error_aware_sft.py",
        "tokenizer_preflight": ROOT / "scripts" / "check_0809_target_tokenizer_preflight.py",
        "training_entry": ROOT / "scripts" / "train_qwen36_0809_agent_error_aware_5epoch.sh",
        "lr_plugin": ROOT / "scripts" / "qwen36_0809_fixed_stage_lr_plugin.py",
        "formal_config": ROOT / "config" / "qwen36_0809_formal_training.json",
        "agent_validation_launcher": ROOT / "scripts" / "run_agent_validation_resilient.sh",
        "agent_validation_retry_policy": ROOT / "scripts" / "agent_validation_retry_policy.sh",
        "agent_validation_early_stop": ROOT / "scripts" / "monitor_agent_validation_early_stop.py",
        "agent_validation_summarizer": ROOT / "scripts" / "summarize_agent_validation.py",
        "final_answer_scorer": ROOT / "scripts" / "final_answer_scoring.py",
        "parent_accepted_selection": PARENT_ROOT / "curation" / "accepted_trajectory_selection.json",
        "parent_causal_clusters": PARENT_ROOT / "curation" / "causal_path_clusters_per_case.json",
        "parent_filter_report": PARENT_ROOT / "curation" / "FILTER_REPORT.md",
        "release_accepted_selection": curation / "accepted_trajectory_selection.json",
        "release_causal_clusters": curation / "causal_path_clusters_per_case.json",
        "release_endpoint_gate_audit": curation / "endpoint_gate_audit.json",
        "release_filter_report": curation / "FILTER_REPORT.md",
    }
    missing_tracked = [name for name, path in tracked_paths.items() if not path.is_file()]
    if missing_tracked:
        raise ValueError(f"0809 release files are missing: {missing_tracked}")
    if not DESIGN_AUDIT_REPORT.is_file():
        raise ValueError(f"0809 design audit report is missing: {DESIGN_AUDIT_REPORT}")

    per_epoch_signal = {
        f"epoch_{epoch:02d}": training_signal(
            [*core_schedules[epoch], *endpoint_schedules[epoch]]
        )
        for epoch in range(1, EPOCHS + 1)
    }
    counterfactual_counts = Counter(
        row["metadata"]["target_type"] for row in counterfactual_rows
    )
    epoch3_counterfactual_ids = {
        row["id"]
        for epoch in range(1, 4)
        for row in core_schedules[epoch]
        if row["metadata"].get("counterfactual_schedule_epoch") == epoch
    }
    epoch3_counterfactual_counts = Counter(
        row["metadata"]["target_type"]
        for row in counterfactual_rows
        if row["id"] in epoch3_counterfactual_ids
    )
    suppressed_parent_calls = [
        call
        for row in base_core_rows
        for call in row["metadata"]["0809_parent_tool_path_gate"]["suppressed_calls"]
    ]
    suppressed_parent_reasons = Counter(
        reason for call in suppressed_parent_calls for reason in call["reasons"]
    )
    post_closure_repairs = [
        row
        for row in base_core_rows
        if row["metadata"].get("0809_post_closure_gate", {}).get("repaired") is True
    ]
    post_closure_removed_calls = sum(
        row["metadata"]["0809_post_closure_gate"]["removed_positive_tool_call_count"]
        for row in post_closure_repairs
    )
    action_grounding_repairs = [
        row
        for row in base_core_rows
        if row["metadata"]
        .get("0809_action_visible_grounding_gate", {})
        .get("repaired")
        is True
    ]
    action_grounding_issue_counts = Counter(
        issue
        for row in action_grounding_repairs
        for issue in row["metadata"]["0809_action_visible_grounding_gate"][
            "original_issue_codes"
        ]
    )
    action_rows = [
        row
        for row in counterfactual_rows
        if row["metadata"].get("target_type")
        == "counterfactual_action_selection"
    ]
    structural_action_distribution = Counter(
        (
            int(
                row["metadata"]["0809_action_visible_grounding_gate"][
                    "best_binding_matched_fact_count"
                ]
            ),
            int(
                row["metadata"]["0809_action_visible_grounding_gate"][
                    "best_binding_expected_fact_count"
                ]
            ),
        )
        for row in action_rows
    )
    parent_manifest = load_json(PARENT_MANIFEST)
    formal_config = load_json(ROOT / "config" / "qwen36_0809_formal_training.json")
    preflight = None
    if paths["preflight"].is_file():
        candidate = load_json(paths["preflight"])
        candidate_datasets = candidate.get("datasets", {})
        preflight_dataset_names = [
            "train_semantic_pool",
            "validation",
            *[
                f"{kind}_epoch_{epoch:02d}"
                for epoch in range(1, EPOCHS + 1)
                for kind in ("core", "endpoint")
            ],
        ]
        preflight_matches_outputs = all(
            candidate_datasets.get(name, {}).get("sha256_lf_normalized")
            == outputs[name]["sha256_lf_normalized"]
            and candidate_datasets.get(name, {}).get("rows") == outputs[name]["rows"]
            for name in preflight_dataset_names
        )
        totals = candidate.get("totals", {})
        if (
            candidate.get("schema_version")
            == "0809-target-tokenizer-loss-mask-preflight.v1"
            and candidate.get("status") == "passed"
            and preflight_matches_outputs
            and totals.get("over_max_length_rows") == 0
            and totals.get("loss_mask_failures") == 0
            and int(totals.get("max_tokens", 16385)) <= 16384
        ):
            preflight = {
                "path": paths["preflight"].relative_to(ROOT).as_posix(),
                "status": candidate.get("status"),
                "sha256_lf_normalized": digest_file(paths["preflight"]),
                "datasets_match_current_outputs": True,
                "max_tokens": totals["max_tokens"],
                "over_max_length_rows": totals["over_max_length_rows"],
                "loss_mask_failures": totals["loss_mask_failures"],
            }

    manifest = {
        "schema_version": "qwen36-0809-agent-error-aware-sft.v7",
        "status": (
            "static_validation_ready_tokenizer_preflight_archived"
            if preflight and preflight["status"] == "passed"
            else "static_validation_ready_tokenizer_preflight_required"
        ),
        "scope": "data/2026-08-09 only",
        "data_root": "data/2026-08-09",
        "design_input": {
            "path": DESIGN_AUDIT_REPORT.relative_to(ROOT).as_posix(),
            "sha256_lf_normalized": digest_file(DESIGN_AUDIT_REPORT),
            "scope_note": "repository-contained v7 design input: message-derived structural action closure, candidate-scope-calibrated endpoints, portable release paths, reachable marker-free targets, one bundle per real evidence path, runtime identity binding and complete dependency closure",
        },
        "parent_release": {
            "path": PARENT_ROOT.relative_to(ROOT).as_posix(),
            "manifest": PARENT_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256_lf_normalized": digest_file(PARENT_MANIFEST),
            "schema_version": parent_manifest["schema_version"],
            "raw_tree_sha256": tree_digest(parent_raw_tree),
        },
        "raw_archive": {
            "files": len(release_raw_tree),
            "bytes": sum(size for size, _ in release_raw_tree.values()),
            "tree_sha256": tree_digest(release_raw_tree),
            "byte_identical_to_parent_0805": True,
        },
        "split": {
            "source": "frozen 0804/0805 case split",
            "train_case_count": 72,
            "validation_case_count": 12,
            "validation_case_ids": sorted(FROZEN_VALIDATION_CASES),
            "validation_role": "error-mining/dev; not a clean proof after its aggregate failures informed design",
            "counterfactual_train_validation_intersection": [],
            "agent_evaluation_trajectories_used_for_training": False,
            "clean_topology_heldout_required_for_final_claim": True,
        },
        "conversion": {
            "parent_semantic_rows_considered": len(train_rows),
            "parent_core_rows_retained": len(base_core_rows),
            "parent_endpoint_groups_considered": endpoint_gate_audit["inherited_groups"],
            "strict_endpoint_groups_rebuilt": endpoint_gate_audit["strict_groups"],
            "failed_endpoint_groups_removed": endpoint_gate_audit["removed_groups"],
            "strict_endpoint_case_count": endpoint_gate_audit["strict_case_count"],
            "same_query_recovery_case_count": endpoint_gate_audit[
                "same_query_recovery_case_count"
            ],
            "same_query_recovery_case_ids": endpoint_gate_audit[
                "same_query_recovery_case_ids"
            ],
            "endpoint_gate_audit": "data/2026-08-09/curation/endpoint_gate_audit.json",
            "counterfactual_rows": len(counterfactual_rows),
            "counterfactual_types": dict(sorted(counterfactual_counts.items())),
            "counterfactual_source": "308 distinct high-information action contexts from successful training-case evidence only",
            "unsafe_parent_tool_call_policy": "remove cross-snapshot, unresolved-snapshot and saved_configs path-glob call/response pairs; zero loss is not sufficient because history still conditions targets",
            "unsafe_parent_tool_calls_suppressed": len(suppressed_parent_calls),
            "unsafe_parent_tool_calls_suppressed_by_reason": dict(
                sorted(suppressed_parent_reasons.items())
            ),
            "post_closure_parent_rows_repaired": len(post_closure_repairs),
            "post_closure_positive_tool_calls_removed": post_closure_removed_calls,
            "post_closure_repair_policy": "recompute every strict label-family fact binding from tool responses visible before the first positive loss; if any binding is complete, remove all positive continuation calls and append an evidence-specific stop",
            "action_visible_grounding_parent_rows_repaired": len(action_grounding_repairs),
            "action_visible_grounding_issue_counts": dict(
                sorted(action_grounding_issue_counts.items())
            ),
            "action_visible_grounding_policy": "for all selected actions, reconstruct every strict endpoint binding for the same query from tool responses visible before the first positive loss; an incomplete binding may supervise only literal matched facts, explicit missing requirements, a pending hypothesis, and one or two read-only actions",
            "action_policy": "all 308 label-relevant current-snapshot action contexts are structurally incomplete under message-derived strict bindings; complete contexts are converted to stop, while incomplete contexts must state matched/expected fact counts and cannot assert a closed loop, unique root cause, or final label",
            "stop_policy": "complete structural evidence stops further calls; endpoint bundles combine evidence summary, positive label boundary, candidate-scope calibration, current-evidence minimal set, stop, and final result once per real evidence path",
            "label_boundary_policy": "bind direct raw output lines to one of six fact families; positive evidence confirms the mapped label but does not prove unchecked neighboring candidates absent",
            "minimal_set_policy": "supervise only the result set confirmed by the current complete positive binding; do not reinterpret it as global negative proof about untested candidates",
            "elimination_policy": "retain only 103 inherited visible-source elimination nodes across 58 queries plus eight declared same-target sampling replays; endpoint bundles add no unsupported candidate exclusion",
            "inherited_endpoint_policy": "rerun v5 literal gates over all 237 inherited groups; recover missing cases only from same-query/snapshot frozen 0805 successful events; emit exactly one reachable bundle per distinct raw evidence prefix and never count an objective rewrite as a new path",
            "metadata_negative_policy": "malformed protocol, rejected action, minimal-set and rejected-continuation negatives are audit evidence only and are not counted as training signal; use them for grammar/preference evaluation, never as positive SFT targets",
            "synthetic_endpoint_eligible_case_count": len(direct_evidence_cases),
            "synthetic_endpoint_eligibility_by_label": direct_evidence_labels,
            "synthetic_endpoint_omitted_case_ids": sorted(
                train_case_ids - direct_evidence_cases
            ),
            "validation_trajectory_policy": "the 12 validation cases and their Agent error trajectories are never augmentation inputs",
            "infrastructure_failures_and_interruptions": "not recorded, sampled, or included in denominators",
        },
        "sampling": {
            "epochs": EPOCHS,
            "base_core_pool_rows": len(base_core_rows),
            "counterfactual_pool_rows": len(counterfactual_rows),
            "endpoint_pool_rows": len(endpoint_pool_rows),
            "base_core_rows_per_epoch": len(base_core_rows),
            "counterfactual_rows_per_epoch": len(counterfactual_rows),
            "endpoint_rows_per_epoch": len(endpoint_schedules[1]),
            "effective_rows_per_epoch": len(core_schedules[1]) + len(endpoint_schedules[1]),
            "effective_batch_size": 8,
            "optimizer_steps_per_epoch": math.ceil(
                (len(core_schedules[1]) + len(endpoint_schedules[1])) / 8
            ),
            "cumulative_optimizer_steps_by_epoch": [144, 288, 432, 576, 720],
            "fixed_agent_checkpoint_epoch": 3,
            "fixed_agent_checkpoint_global_step": 432,
            "budget_match": "exactly 1151 rows and 144 optimizer steps per epoch: 719 base core + 308 reachable actions + 8 explicit same-target elimination replays + 116 real-path endpoint bundles",
            "counterfactual_query_balance": "all 308 distinct reachable action rows are exposed once per epoch; every new target is therefore visible before fixed checkpoint-432",
            "epoch_3_unique_counterfactual_coverage": dict(sorted(epoch3_counterfactual_counts.items())),
            "endpoint_query_balance": "all 116 distinct raw evidence paths are exposed once per epoch with 1/path_count query loss normalization; no objective rewrite or replay counts as a path",
        },
        "counts": {
            "train_semantic_pool_rows": len(semantic_rows),
            "parent_train_rows": len(train_rows),
            "base_core_pool_rows": len(base_core_rows),
            "counterfactual_pool_rows": len(counterfactual_rows),
            "endpoint_pool_rows": len(endpoint_pool_rows),
            "action_visible_grounding_parent_rows_repaired": len(
                action_grounding_repairs
            ),
            "action_visible_grounding_action_rows_repaired": sum(
                row["metadata"]
                .get("0809_action_visible_grounding_gate", {})
                .get("repaired")
                is True
                for row in counterfactual_rows
                if row["metadata"].get("target_type")
                == "counterfactual_action_selection"
            ),
            "structurally_incomplete_action_rows": sum(
                row["metadata"]["0809_action_visible_grounding_gate"].get(
                    "structural_binding_complete"
                )
                is False
                for row in action_rows
            ),
            "structural_action_matched_expected_distribution": {
                f"{matched}/{expected}": count
                for (matched, expected), count in sorted(
                    structural_action_distribution.items()
                )
            },
            "post_closure_parent_rows_repaired": len(post_closure_repairs),
            "post_closure_positive_tool_calls_removed": post_closure_removed_calls,
            "strict_endpoint_groups": endpoint_gate_audit["strict_groups"],
            "unique_raw_evidence_paths": endpoint_gate_audit["unique_raw_evidence_path_count"],
            "elimination_replay_rows": sum(
                row["metadata"]["target_type"] == "hypothesis_elimination_replay"
                for row in counterfactual_rows
            ),
            "removed_inherited_endpoint_groups": endpoint_gate_audit["removed_groups"],
            "validation_rows": len(validation_rows),
            "real_hypothesis_elimination_rows": sum(
                row["metadata"]["target_type"] == "hypothesis_elimination"
                for row in base_core_rows
            ),
            "real_hypothesis_elimination_case_count": 72 - len(missing_elimination_cases),
            "evidence_bound_elimination_rows": 0,
            "endpoint_bundles_with_neighbor_elimination": 0,
            "endpoint_bundles_with_candidate_scope_calibration": len(
                endpoint_pool_rows
            ),
            "endpoint_rejected_label_declarations": 0,
            "combined_evidence_backed_elimination_case_count": 72
            - len(missing_elimination_cases),
            "queries_without_real_elimination_node": sorted(
                missing_elimination_cases
            ),
        },
        "heuristic_training_signal_by_epoch": per_epoch_signal,
        "training_profile": {
            "formal_config": "config/qwen36_0809_formal_training.json",
            "formal_entry": "scripts/train_qwen36_0809_agent_error_aware_5epoch.sh",
            "distributed_strategy": "ddp",
            "world_size": 2,
            "cuda_visible_devices": "0,1",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "max_length": 16384,
            "fixed_validation_epoch": 3,
            "agent_checkpoint_selection": False,
            "runtime_identity_gate_required": True,
            "canonical_config_and_plugin_required": True,
            "dataset_components_by_epoch": {
                f"epoch_{epoch:02d}": [
                    f"data/2026-08-09/sft/qwen3_6_27b_0809_core_epoch_{epoch:02d}.jsonl",
                    f"data/2026-08-09/sft/qwen3_6_27b_0809_endpoint_epoch_{epoch:02d}.jsonl",
                ]
                for epoch in range(1, EPOCHS + 1)
            },
        },
        "evaluation_protocol": formal_config["evaluation_policy"],
        "reproducibility": {
            "single_data_root_contract": True,
            "document": "data/2026-08-09/REPRODUCIBILITY.md",
            "tracked_files": {
                name: {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256_lf_normalized": digest_file(path),
                }
                for name, path in tracked_paths.items()
            },
            "release_record": "data/2026-08-09/sft/RELEASE_RECORD.json",
        },
        "target_tokenizer_preflight": preflight or {
            "path": paths["preflight"].relative_to(ROOT).as_posix(),
            "status": "required_before_training",
        },
        "outputs": outputs,
    }
    write_json(paths["manifest"], manifest)

    output_tree = {
        record["path"]: (record["bytes"], record["sha256_lf_normalized"])
        for record in outputs.values()
    }
    release_record = {
        "schema_version": "0809-immutable-release-record.v1",
        "release_id": "0809",
        "data_root": "data/2026-08-09",
        "diagnostics": {"generated_at_resolved_data_root": data_root.as_posix()},
        "repository": git_state(),
        "manifest": {
            "path": paths["manifest"].relative_to(ROOT).as_posix(),
            "sha256_lf_normalized": digest_file(paths["manifest"]),
        },
        "parent_manifest_sha256_lf_normalized": digest_file(PARENT_MANIFEST),
        "raw_tree_sha256": tree_digest(release_raw_tree),
        "output_tree_sha256": tree_digest(output_tree),
        "outputs": outputs,
        "note": "A Git commit is still required for an immutable published release; this record freezes the generated worktree artifacts before commit.",
    }
    write_json(paths["release_record"], release_record)

    allowed = {path.resolve() for path in paths.values()}
    allowed.update(
        (sft / f"qwen3_6_27b_0809_core_epoch_{epoch:02d}.jsonl").resolve()
        for epoch in range(1, EPOCHS + 1)
    )
    allowed.update(
        (sft / f"qwen3_6_27b_0809_endpoint_epoch_{epoch:02d}.jsonl").resolve()
        for epoch in range(1, EPOCHS + 1)
    )
    for stale in sft.iterdir():
        if stale.is_file() and stale.resolve() not in allowed:
            stale.unlink()

    print(f"resolved_data_root={data_root}")
    print(
        f"semantic={len(semantic_rows)} base_core={len(base_core_rows)} "
        f"counterfactual={len(counterfactual_rows)} endpoint={len(endpoint_pool_rows)} "
        f"validation={len(validation_rows)}"
    )
    print(
        "per_epoch="
        f"{len(core_schedules[1]) + len(endpoint_schedules[1])} rows / "
        "144 optimizer steps; fixed epoch-3 checkpoint=432"
    )
    print(f"manifest={paths['manifest'].relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
