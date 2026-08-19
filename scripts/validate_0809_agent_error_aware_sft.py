#!/usr/bin/env python3
"""Official validator for the 0809 error-aware SFT release."""

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
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-08-09"
VALIDATION_CASES = {2, 12, 19, 20, 29, 38, 65, 71, 85, 86, 99, 100}
EXPECTED_CF = {
    "counterfactual_action_selection": 308,
    "hypothesis_elimination_replay": 8,
}
EXPECTED_AUDIT_ACTION_REPAIRS = {
    "0809_cf_action_q0011_a01": "unsupported_global_stp_disabled",
    "0809_cf_action_q0014_a04": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0015_a01": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0018_a02": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0021_a01": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0022_a01": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0024_a03": "unsupported_bpdu_filter_enabled",
}
EXPECTED_V7_STRUCTURAL_ACTION_REPAIRS = {
    "0809_cf_action_q0015_a02",
    "0809_cf_action_q0025_a02",
    "0809_cf_action_q0026_a01",
    "0809_cf_action_q0026_a02",
    "0809_cf_action_q0027_a02",
    "0809_cf_action_q0027_a03",
    "0809_cf_action_q0028_a01",
    "0809_cf_action_q0028_a02",
    "0809_cf_action_q0028_a03",
    "0809_cf_action_q0031_a01",
    "0809_cf_action_q0034_a01",
    "0809_cf_action_q0035_a03",
    "0809_cf_action_q0037_a03",
    "0809_cf_action_q0039_a01",
    "0809_cf_action_q0060_a02",
    "0809_cf_action_q0061_a02",
    "0809_cf_action_q0066_a03",
}
AUDITED_COMPLETE_CONTINUATION_SOURCE = "q0023_path_02_success_09_step_04"
CANONICAL_SCORING_POLICY_VERSION = (
    "agent-final-answer.v4.2026-08-12-incomplete-result-exact-recovery"
)
PREMATURE_CONVERGENCE_PHRASES = (
    "证据已经形成闭环", "证据已形成闭环", "关键区分证据已出现",
    "决定性证据已出现", "唯一异常", "唯一的异常", "决定性异常", "根因应归",
)
EVENT_ITEM_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def resolve_data_root(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    path = path.resolve()
    expected = (ROOT / "data" / "2026-08-09").resolve()
    if path != expected:
        raise ValueError(f"validator expected {expected}, received {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def result_block(items: list[str]) -> str:
    return "<result>\n[\n" + ",\n".join(
        json.dumps(item, ensure_ascii=False) for item in items
    ) + "\n]\n</result>"


def validate_recovery_provenance(facts: list[dict[str, Any]], row_id: str) -> None:
    records = [fact.get("source_action_provenance") for fact in facts]
    if not any(records):
        return
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{row_id}: recovery provenance is partial")
    for fact, record in zip(facts, records):
        event_file = str(record["source_event_file"])
        event_path = (ROOT / event_file).resolve()
        if ROOT.resolve() not in event_path.parents or not event_path.is_file():
            raise ValueError(f"{row_id}: recovery event path is missing/outside repository")
        if digest_file(event_path) != record["source_event_sha256_lf_normalized"]:
            raise ValueError(f"{row_id}: recovery event hash mismatch")
        if event_file not in EVENT_ITEM_CACHE:
            items: dict[str, dict[str, Any]] = {}
            for line in event_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                item = event.get("item", {})
                if event.get("type") == "item.completed" and item.get("type") == "command_execution":
                    items[str(item["id"])] = item
            EVENT_ITEM_CACHE[event_file] = items
        item = EVENT_ITEM_CACHE[event_file].get(str(record["item_id"]))
        if item is None or int(item.get("exit_code", 1)) != 0 or item.get("status") != "completed":
            raise ValueError(f"{row_id}: recovery item is absent or unsuccessful")
        command = str(item.get("command") or "")
        output = str(item.get("aggregated_output") or "")
        normalized_output = output.replace("\r\n", "\n").replace("\r", "\n")
        normalized_fact = re.sub(r"\s+", " ", str(fact["output_line"])).strip()
        raw_span = str(record["raw_output_line_or_span"])
        if (
            record.get("action_id") is None
            or command != record.get("source_command")
            or digest_text(command.replace("\r\n", "\n").replace("\r", "\n"))
            != record.get("source_command_sha256_lf_normalized")
            or digest_text(normalized_output)
            != record.get("raw_output_sha256_lf_normalized")
            or raw_span not in output.splitlines()
            or re.sub(r"\s+", " ", raw_span).strip() != normalized_fact
            or record.get("normalized_observation") != normalized_fact
            or record.get("normalization_policy")
            != "collapse_whitespace_to_single_ascii_space_and_strip"
        ):
            raise ValueError(f"{row_id}: recovery action/command/output provenance mismatch")


def expected_snapshot(row: dict[str, Any]) -> str:
    user_text = "\n".join(
        message.get("content", "")
        for message in row["messages"]
        if message.get("role") == "user"
    )
    match = re.search(r"CampusNetwork(?:-for-perf|_\d+)", user_text)
    if not match:
        raise ValueError(f"{row['id']}: no snapshot in user prompt")
    return match.group(0)


def validate_binding_semantics(
    binding: dict[str, Any], metadata: dict[str, Any], row_id: str
) -> None:
    label = binding.get("label")
    if metadata.get("correct_labels") != [label]:
        raise ValueError(f"{row_id}: binding label differs from the supervised label")
    items = metadata.get("actual_result_items") or []
    items = [items] if isinstance(items, str) else items
    answer_devices = {item.split(";", 1)[0] for item in items}
    facts = binding["facts"]
    kinds = Counter(fact["fact_kind"] for fact in facts)
    if label == "全局STP未使能":
        valid = {fact["device"] for fact in facts} == answer_devices and kinds == Counter({"protocol_status_disabled": len(answer_devices)}) and all(
            re.search(r"Protocol Status\s*:\s*Disabled", fact["output_line"], re.I)
            for fact in facts
        )
    elif label == "STP BPDU被过滤":
        valid = {fact["device"] for fact in facts} == answer_devices and kinds == Counter({"bpdu_filter_current_configuration": len(answer_devices)}) and all(
            "display_current-configuration" in fact["command"].lower()
            and re.search(r"(?:^|:\d+:)stp port bpdu-filter enable\s*$", fact["output_line"], re.I)
            for fact in facts
        )
    elif label == "VRRP工作在非抢占模式":
        valid = {fact["device"] for fact in facts} == answer_devices and kinds == Counter({"vrrp_preempt_no": len(answer_devices)}) and all(
            re.search(r"Preempt\s*:\s*NO", fact["output_line"], re.I)
            for fact in facts
        )
    elif label == "存在IP路由环路":
        routes = [fact for fact in facts if fact["fact_kind"] == "static_route_directed_next_hop"]
        owners = [fact for fact in facts if fact["fact_kind"] == "next_hop_interface_owner"]
        successor = {fact["device"]: fact.get("next_hop_owner") for fact in routes}
        start = sorted(answer_devices)[0] if answer_devices else ""
        valid = (
            kinds == Counter({"static_route_directed_next_hop": 3, "next_hop_interface_owner": 3})
            and {fact["device"] for fact in routes} == answer_devices
            and set(successor.values()) == answer_devices
            and successor.get(successor.get(successor.get(start))) == start
            and len({tuple(fact["parsed_transition"])[0] for fact in routes}) == 1
            and {fact["device"] for fact in owners} == answer_devices
        )
    elif label == "存在MPLS标签环路":
        routes = [fact for fact in facts if fact["fact_kind"] == "static_lsp_directed_label_hop"]
        owners = [fact for fact in facts if fact["fact_kind"] == "next_hop_interface_owner"]
        by_device = {fact["device"]: fact for fact in routes}
        successor = {fact["device"]: fact.get("next_hop_owner") for fact in routes}
        start = sorted(answer_devices)[0] if answer_devices else ""
        valid = (
            kinds == Counter({"static_lsp_directed_label_hop": 3, "next_hop_interface_owner": 3})
            and set(by_device) == answer_devices
            and set(successor.values()) == answer_devices
            and successor.get(successor.get(successor.get(start))) == start
            and all(
                tuple(fact["parsed_transition"])[2]
                == tuple(by_device[successor[fact["device"]]]["parsed_transition"])[0]
                for fact in routes
            )
            and {fact["device"] for fact in owners} == answer_devices
        )
    elif label == "VRRP Master角色规划不合理":
        valid = kinds == Counter(
            {
                "source_host_ipv4_vlan": 1,
                "vrrp_master": len(answer_devices),
                "mst_vlan_instance_mapping": len(answer_devices),
                "mst_alternate_discarding": len(answer_devices),
            }
        ) and all(
            {fact["device"] for fact in facts if fact["fact_kind"] == kind} == answer_devices
            for kind in ("vrrp_master", "mst_vlan_instance_mapping", "mst_alternate_discarding")
        )
    else:
        valid = False
    if not valid:
        raise ValueError(f"{row_id}: label-specific direct-evidence closure failed")


def validate_output(path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    if path.relative_to(ROOT).as_posix() != record["path"]:
        raise ValueError(f"manifest path mismatch for {path}")
    rows = load_jsonl(path)
    if len(rows) != record["rows"]:
        raise ValueError(f"{path}: row count differs from manifest")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"{path}: byte count differs from manifest")
    if digest_file(path) != record["sha256_lf_normalized"]:
        raise ValueError(f"{path}: digest differs from manifest")
    return rows


def case_id(row: dict[str, Any]) -> int:
    return int(row["metadata"]["case_id"])


def positive_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in row["messages"]
        if float(message.get("loss_scale", 0) or 0) > 0
    ]


def positive_prefix_text(row: dict[str, Any]) -> str:
    prefix: list[str] = []
    for message in row["messages"]:
        if float(message.get("loss_scale", 0) or 0) > 0:
            break
        prefix.append(message.get("content", ""))
    return "\n".join(prefix)


def positive_target_text(row: dict[str, Any]) -> str:
    return json.dumps(
        [
            (message.get("role"), message.get("content", ""), message.get("loss_scale"))
            for message in row["messages"]
            if float(message.get("loss_scale", 0) or 0) > 0
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def normalize_visible_literal(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def recompute_structural_binding_status(
    row: dict[str, Any], endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    boundary = next(
        (
            index
            for index, message in enumerate(row["messages"])
            if float(message.get("loss_scale", 0) or 0) > 0
        ),
        len(row["messages"]),
    )
    visible_responses = [
        message.get("content", "")
        for message in row["messages"][:boundary]
        if message.get("role") == "tool_response"
    ]
    visible = normalize_visible_literal("\n".join(visible_responses))
    candidates = []
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
                "matched": matched,
                "complete": len(matched) == len(facts),
            }
        )
    if not candidates:
        raise ValueError(f"{row['id']}: structural validator lacks endpoint binding")
    return max(
        candidates,
        key=lambda item: (
            int(item["complete"]),
            len(item["matched"]),
            -len(item["binding"]["facts"]),
            item["binding"]["sha256"],
        ),
    )


def affirmative_root_claim(text: str) -> bool:
    if any(token in text for token in ("待验证", "尚未", "没有形成", "未形成", "若", "如果")):
        return False
    patterns = (
        r"首尾相接.{0,24}(?:构成|形成).{0,24}(?:闭环|环)",
        r"(?:三节点|三个节点).{0,30}(?:静态路由环|路由环|转发环|闭环)",
        r"静态标签交换.{0,30}(?:构成|形成).{0,24}(?:闭环|环)",
        r"启用(?:了)?\s*`?BPDU\s*Filter`?",
        r"关键路径已闭合",
        r"(?:应按|直接按).{0,16}存在.{0,16}环路.{0,16}归因",
    )
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def validate_structural_action_contract(
    row: dict[str, Any], endpoint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    status = recompute_structural_binding_status(row, endpoint_rows)
    positives = positive_messages(row)
    assistant_text = "\n".join(
        message.get("content", "")
        for message in positives
        if message.get("role") == "assistant"
    )
    positive_tools = [
        message for message in positives if message.get("role") == "tool_call"
    ]
    if status["complete"] and positive_tools:
        raise ValueError(f"{row['id']}: complete structural evidence still calls tools")
    if not status["complete"]:
        expected = len(status["binding"]["facts"])
        matched = len(status["matched"])
        if (
            "待验证假设" not in assistant_text
            or f"{matched}/{expected}" not in assistant_text
            or "结果返回前不输出最终答案" not in assistant_text
            or affirmative_root_claim(assistant_text)
        ):
            raise ValueError(
                f"{row['id']}: incomplete action is not a calibrated pending hypothesis"
            )
    return status


def action_visible_grounding_issues(row: dict[str, Any]) -> list[str]:
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
        if (
            re.search(r"Protocol\s+Status\s*:\s*Disabled", positive, re.I)
            or re.search(r"全局.{0,24}STP.{0,24}(?:未使能|未启用|Disabled)", positive, re.I)
            or re.search(r"(?:缺少|没有).{0,20}`?stp\s+enable`?", positive, re.I)
        ) and not re.search(r"Protocol\s+Status\s*:\s*Disabled", visible, re.I):
            issues.append("unsupported_global_stp_disabled")
        if (
            re.search(r"stp\s+port\s+bpdu-filter\s+enable", positive, re.I)
            or (re.search(r"BPDU", positive, re.I) and "过滤" in positive)
        ) and not re.search(r"stp\s+port\s+bpdu-filter\s+enable", visible, re.I):
            issues.append("unsupported_bpdu_filter_enabled")
        if (
            re.search(r"Preempt\s*:\s*NO", positive, re.I)
            or re.search(r"preempt\s+disable", positive, re.I)
            or "非抢占" in positive
        ) and not re.search(r"Preempt\s*:\s*NO|preempt\s+disable", visible, re.I):
            issues.append("unsupported_vrrp_nonpreempt")
        if re.search(
            r"(?:配置|启用|存在|包含).{0,20}`?stp\s+(?:global\s+)?enable`?",
            positive,
            re.I,
        ) and not re.search(
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


def validate_action_grounding_negative_fixtures() -> None:
    claims = (
        ("unsupported_global_stp_disabled", "已确认 Protocol Status: Disabled。"),
        ("unsupported_bpdu_filter_enabled", "上联配置了 stp port bpdu-filter enable。"),
        ("unsupported_vrrp_nonpreempt", "运行态为 Preempt : NO。"),
        ("premature_convergence_assertion", "证据已经形成闭环。"),
        ("premature_convergence_assertion", "当前已经形成 IP 路由环路。"),
    )
    for expected, claim in claims:
        fixture = {
            "messages": [
                {"role": "tool_response", "content": "unrelated output"},
                {"role": "assistant", "content": claim, "loss_scale": 1.0},
            ]
        }
        if expected not in action_visible_grounding_issues(fixture):
            raise ValueError(f"action grounding negative fixture missed {expected}")
    exclusion_fixture = {
        "messages": [
            {"role": "tool_response", "content": "route checks are visible"},
            {
                "role": "assistant",
                "content": "没有形成 IP/MPLS 环路，继续核对其他候选。",
                "loss_scale": 1.0,
            },
        ]
    }
    if "premature_convergence_assertion" in action_visible_grounding_issues(
        exclusion_fixture
    ):
        raise ValueError("action grounding gate erased a negated loop exclusion")
    for claim in (
        "三台PE的静态下一跳首尾相接，构成闭环。",
        "已发现两组相反方向的三节点静态路由环。",
        "三台PE上的静态标签交换关系均构成三节点闭环。",
        "上联端口启用了 BPDU Filter。",
        "关键路径已闭合，应直接归因。",
    ):
        if not affirmative_root_claim(claim):
            raise ValueError(f"structural synonym fixture was missed: {claim}")
    for safe in (
        "待验证假设：若三台下一跳最终首尾相接，则可能形成环路。",
        "没有形成 IP/MPLS 环路，继续核对其他候选。",
    ):
        if affirmative_root_claim(safe):
            raise ValueError(f"structural synonym fixture false-positive: {safe}")


def validate_tool_call(message: dict[str, Any], row_id: str) -> None:
    try:
        payload = json.loads(message["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row_id}: malformed supervised tool JSON") from exc
    if set(payload) != {"name", "arguments"}:
        raise ValueError(f"{row_id}: tool payload must contain name and arguments only")
    if payload["name"] != "exec_command":
        raise ValueError(f"{row_id}: unexpected tool name {payload['name']!r}")
    if not isinstance(payload["arguments"], dict) or set(payload["arguments"]) != {"cmd"}:
        raise ValueError(f"{row_id}: exec_command arguments must be cmd-only")
    cmd = payload["arguments"]["cmd"]
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError(f"{row_id}: empty cmd")
    if not cmd.startswith(("cat ", "grep ", "find ", "head ", "tail ", "test ")):
        raise ValueError(f"{row_id}: command is outside the read-only allowlist: {cmd}")
    if "saved_configs/" not in cmd:
        raise ValueError(f"{row_id}: command does not use a repository-relative path")


def validate_training_tool_path(
    message: dict[str, Any], row: dict[str, Any]
) -> None:
    command = json.loads(message["content"])["arguments"]["cmd"]
    snapshot = expected_snapshot(row)
    folders = set(re.findall(r"saved_configs/([^/\s]+)", command))
    if folders != {snapshot}:
        raise ValueError(f"{row['id']}: supervised tool call crosses/unresolves snapshot")
    if re.search(r"saved_configs/[^\s]*(?:\*|\?|\[)", command):
        raise ValueError(f"{row['id']}: supervised tool call contains a path glob")


def validate_row(row: dict[str, Any], *, counterfactual: bool) -> None:
    row_id = row.get("id")
    if not row_id or not isinstance(row.get("messages"), list):
        raise ValueError("row lacks id/messages")
    if row["messages"][0].get("role") != "system":
        raise ValueError(f"{row_id}: first message is not system")
    metadata = row.get("metadata", {})
    if metadata.get("release_id") != "0809":
        raise ValueError(f"{row_id}: release_id is not 0809")
    if metadata.get("release_data_root") != "data/2026-08-09":
        raise ValueError(f"{row_id}: release data root is stale")
    if metadata.get("error_mining_agent_trajectory_used") is not False:
        raise ValueError(f"{row_id}: Agent evaluation trajectory use is not explicitly false")
    if metadata.get("validation_case_used_for_augmentation") is not False:
        raise ValueError(f"{row_id}: validation-case augmentation flag is not false")
    if metadata.get("split") == "train" and case_id(row) in VALIDATION_CASES:
        raise ValueError(f"{row_id}: validation case leaked into training")
    if counterfactual and case_id(row) in VALIDATION_CASES:
        raise ValueError(f"{row_id}: counterfactual row uses a validation case")
    positives = positive_messages(row)
    if not positives:
        raise ValueError(f"{row_id}: no supervised target")
    for message in positives:
        scale = float(message["loss_scale"])
        if not 0 < scale <= 1:
            raise ValueError(f"{row_id}: invalid loss scale {scale}")
        if message.get("role") == "tool_call":
            validate_tool_call(message, row_id)
    if metadata.get("split") == "train":
        for context_message in row["messages"]:
            if context_message.get("role") == "tool_call":
                validate_training_tool_path(context_message, row)
    if metadata.get("target_type") == "counterfactual_stop":
        if any(message.get("role") == "tool_call" for message in positives):
            raise ValueError(f"{row_id}: stop target still supervises a tool call")
        if metadata.get("must_emit_no_tool_call") is not True:
            raise ValueError(f"{row_id}: stop target lacks no-tool contract")
    if metadata.get("dataset_type") == "0809_strict_evidence_endpoint":
        binding = metadata.get("evidence_binding") or {}
        facts = binding.get("facts") or []
        canonical = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        if (
            metadata.get("strict_endpoint_gate_passed") is not True
            or metadata.get("strict_endpoint_gate_version") != "direct_evidence_binding.v5"
            or not facts
            or binding.get("fact_count") != len(facts)
            or binding.get("sha256") != digest_text(canonical)
            or binding.get("snapshot") != expected_snapshot(row)
        ):
            raise ValueError(f"{row_id}: strict endpoint binding is empty or stale")
        tool_outputs = "\n".join(
            message.get("content", "")
            for message in row["messages"]
            if message.get("role") == "tool_response"
        )
        if any(
            fact.get("output_line") not in tool_outputs
            or fact.get("snapshot") != binding["snapshot"]
            for fact in facts
        ):
            raise ValueError(f"{row_id}: strict endpoint fact is not literal visible output")
        validate_binding_semantics(binding, metadata, row_id)
        validate_recovery_provenance(facts, row_id)
        recovery_source = metadata.get("0809_recovery_source")
        if recovery_source:
            fact_files = sorted(
                {
                    fact["source_action_provenance"]["source_event_file"]
                    for fact in facts
                }
            )
            if (
                recovery_source.get("source_event_files") != fact_files
                or metadata.get("source_event_file") is not None
                or metadata.get("source_event_sha256_lf_normalized") is not None
            ):
                raise ValueError(f"{row_id}: recovery source-file set is stale/misleading")
        if metadata.get("target_type") != "endpoint_bundle":
            raise ValueError(f"{row_id}: strict endpoint is not a reachable bundle")
        if any(message.get("role") == "tool_call" for message in positives):
            raise ValueError(f"{row_id}: strict endpoint continues tool use after closure")
        if metadata.get("must_emit_no_tool_call") is not True:
            raise ValueError(f"{row_id}: strict endpoint lacks no-tool contract")
        positive_text = "\n".join(
            message.get("content", "") for message in positives
        )
        neighbors = metadata.get("unverified_neighbor_labels") or []
        if (
            metadata.get("rejected_labels") != []
            or not neighbors
            or metadata.get("candidate_negative_evidence_bindings") != {}
            or metadata.get("candidate_negative_evidence_complete") is not False
            or metadata.get("hypothesis_elimination_supervised") is not False
            or "保持未验证" not in positive_text
            or "不能推出" not in positive_text
            or any(
                phrase in positive_text
                for phrase in ("排除“", "应予排除", "已经证伪", "候选排除：")
            )
        ):
            raise ValueError(
                f"{row_id}: endpoint treats unsupported neighbor absence as elimination"
            )
    if metadata.get("0809_post_closure_gate", {}).get("repaired") is True:
        binding = metadata.get("evidence_binding") or {}
        facts = binding.get("facts") or []
        canonical = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        visible = "\n".join(
            message.get("content", "")
            for message in row["messages"]
            if message.get("role") == "tool_response"
        )
        if (
            not facts
            or binding.get("sha256") != digest_text(canonical)
            or binding.get("snapshot") != expected_snapshot(row)
            or any(fact.get("output_line") not in visible for fact in facts)
        ):
            raise ValueError(f"{row_id}: post-closure repair binding is invalid")
        validate_binding_semantics(binding, metadata, row_id)
        if any(message.get("role") == "tool_call" for message in positives):
            raise ValueError(f"{row_id}: repaired post-closure row still supervises a tool call")
        if metadata.get("must_emit_no_tool_call") is not True:
            raise ValueError(f"{row_id}: repaired post-closure row lacks stop contract")
    if counterfactual:
        if metadata.get("training_sampling_role") != "counterfactual_pool":
            raise ValueError(f"{row_id}: wrong counterfactual sampling role")
        if metadata.get("review_status") != "auto_generated_evidence_bound_requires_domain_review":
            raise ValueError(f"{row_id}: review status is missing")
        if metadata.get("target_type") in {
            "counterfactual_label_boundary",
            "counterfactual_minimal_set",
            "evidence_bound_elimination",
            "counterfactual_stop",
        }:
            binding = metadata.get("evidence_binding") or {}
            facts = binding.get("facts")
            canonical = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
            if (
                not isinstance(facts, list)
                or not facts
                or digest_text(canonical) != binding.get("sha256")
                or binding.get("fact_count") != len(facts)
                or binding.get("snapshot") != expected_snapshot(row)
            ):
                raise ValueError(f"{row_id}: evidence binding is empty or stale")
            tool_outputs = "\n".join(
                message.get("content", "")
                for message in row["messages"]
                if message.get("role") == "tool_response"
            )
            for fact in facts:
                if (
                    set(fact)
                    != {"command", "snapshot", "device", "output_line", "fact_kind"}
                    or fact["snapshot"] != binding["snapshot"]
                    or not fact["device"]
                    or fact["output_line"] not in tool_outputs
                ):
                    raise ValueError(f"{row_id}: evidence fact is not literal visible output")
            validate_binding_semantics(binding, metadata, row_id)
        if metadata.get("target_type") == "counterfactual_action_selection":
            commands = metadata.get("selected_action_commands") or []
            if not 1 <= len(commands) <= 2:
                raise ValueError(f"{row_id}: action target is not one or two calls")
            snapshot = expected_snapshot(row)
            for command in commands:
                folders = set(re.findall(r"saved_configs/([^/]+)/", command))
                if folders != {snapshot} or any(character in command for character in "*?["):
                    raise ValueError(f"{row_id}: action is not a current-snapshot direct path")
            positive_commands = [
                json.loads(message["content"])["arguments"]["cmd"]
                for message in positives
                if message.get("role") == "tool_call"
            ]
            if commands != positive_commands:
                raise ValueError(f"{row_id}: action metadata differs from supervised calls")
            if metadata.get("complete_binding_before_positive_tool_call") is not False:
                raise ValueError(f"{row_id}: action does not certify pre-action evidence is incomplete")
            if (
                metadata.get("action_visible_grounding_checked") is not True
                or metadata.get("action_visible_grounding_remaining_issues") != []
                or action_visible_grounding_issues(row)
            ):
                raise ValueError(
                    f"{row_id}: positive action reasoning is not grounded at its visible boundary"
                )


def main() -> None:
    validate_action_grounding_negative_fixtures()
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    sft = data_root / "sft"
    manifest_path = sft / "0809_agent_error_aware_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "qwen36-0809-agent-error-aware-sft.v7":
        raise ValueError("unexpected manifest schema")
    if manifest.get("scope") != "data/2026-08-09 only":
        raise ValueError("manifest scope is not 0809-only")
    if manifest.get("data_root") != "data/2026-08-09" or "resolved_data_root_at_generation" in manifest:
        raise ValueError("manifest data root is not portable/logical")
    if set(manifest["split"]["validation_case_ids"]) != VALIDATION_CASES:
        raise ValueError("validation split differs from the frozen split")
    if manifest["split"]["counterfactual_train_validation_intersection"]:
        raise ValueError("manifest reports counterfactual split leakage")
    if manifest["split"]["agent_evaluation_trajectories_used_for_training"] is not False:
        raise ValueError("Agent evaluation trajectories must not be training inputs")
    design_record = manifest.get("design_input", {})
    design_audit = (ROOT / str(design_record.get("path", ""))).resolve()
    if (
        ROOT.resolve() not in design_audit.parents
        or
        not design_audit.is_file()
        or digest_file(design_audit)
        != design_record.get("sha256_lf_normalized")
    ):
        raise ValueError("design audit report is missing or differs from the manifest")

    output_paths = {
        name: ROOT / record["path"] for name, record in manifest["outputs"].items()
    }
    outputs = {
        name: validate_output(path, manifest["outputs"][name])
        for name, path in output_paths.items()
    }
    semantic = outputs["train_semantic_pool"]
    base_core = outputs["base_core_pool"]
    counterfactual = outputs["counterfactual_pool"]
    endpoint_pool = outputs["endpoint_pool"]
    validation = outputs["validation"]
    if (len(semantic), len(base_core), len(counterfactual), len(endpoint_pool), len(validation)) != (
        1151,
        719,
        316,
        116,
        245,
    ):
        raise ValueError("unexpected primary pool counts")
    if len({row["id"] for row in semantic}) != len(semantic):
        raise ValueError("semantic pool contains duplicate row IDs")
    semantic_ids = {row["id"] for row in semantic}
    if not {row["id"] for row in base_core + endpoint_pool + counterfactual} <= semantic_ids:
        raise ValueError("component pool contains a row absent from the semantic pool")

    for row in semantic:
        validate_row(row, counterfactual=row["metadata"]["dataset_type"] == "0809_agent_error_aware_counterfactual")
    for row in validation:
        validate_row(row, counterfactual=False)
    if {case_id(row) for row in validation} != VALIDATION_CASES:
        raise ValueError("validation file has wrong case inventory")
    if {case_id(row) for row in semantic} & VALIDATION_CASES:
        raise ValueError("semantic training pool intersects validation")
    banned_markers = (
        "当前可见状态：下一节点",
        "当前证据尚未闭环；下一节点",
        "下一节点只核对",
        "下一节点只保留",
    )
    for row in semantic:
        if row["metadata"].get("synthetic_assistant_control_marker_used") is True:
            raise ValueError(f"{row['id']}: synthetic assistant control marker is enabled")
        if any(
            marker in message.get("content", "")
            for marker in banned_markers
            for message in row["messages"]
        ):
            raise ValueError(f"{row['id']}: unreachable assistant control marker remains")

    if any(
        row["metadata"].get("dataset_type") != "0809_strict_evidence_endpoint"
        or row["metadata"].get("target_type") != "endpoint_bundle"
        or row["metadata"].get("counts_as_real_evidence_path") is not True
        for row in endpoint_pool
    ):
        raise ValueError("endpoint pool must contain only real-path reachable bundles")
    raw_prefixes = [positive_prefix_text(row) for row in endpoint_pool]
    if len(set(raw_prefixes)) != 116:
        raise ValueError("116 endpoint bundles do not have 116 real raw prefixes")
    for row, prefix in zip(endpoint_pool, raw_prefixes):
        metadata = row["metadata"]
        if metadata.get("raw_first_positive_prefix_sha256") != digest_text(prefix):
            raise ValueError(f"{row['id']}: raw-prefix digest is stale")
        if metadata.get("endpoint_chain_history_policy") != "single_reachable_completion_from_raw_context":
            raise ValueError(f"{row['id']}: endpoint still depends on a synthetic history node")
        required_objectives = {
            "evidence_summary", "label_boundary", "candidate_scope_calibration",
            "minimal_set", "decision_ready", "decision",
        }
        if set(metadata.get("endpoint_objectives", [])) != required_objectives:
            raise ValueError(f"{row['id']}: endpoint bundle objectives are incomplete")
        if result_block(metadata["actual_result_items"]) not in [
            message.get("content", "") for message in positive_messages(row)
        ]:
            raise ValueError(f"{row['id']}: verified result is absent from bundle")
    path_count_by_case = Counter(case_id(row) for row in endpoint_pool)
    if Counter(path_count_by_case.values()) != Counter({1: 47, 2: 9, 3: 13, 4: 3}):
        raise ValueError(f"real endpoint path distribution changed: {path_count_by_case}")
    for current_case, path_count in path_count_by_case.items():
        rows = [row for row in endpoint_pool if case_id(row) == current_case]
        if any(
            row["metadata"].get("query_raw_evidence_path_count") != path_count
            or not math.isclose(
                float(row["metadata"].get("query_normalized_path_weight", 0)),
                1 / path_count,
                rel_tol=0,
                abs_tol=1e-8,
            )
            for row in rows
        ):
            raise ValueError(f"q{current_case}: endpoint query normalization is wrong")

    cf_counts = Counter(row["metadata"]["target_type"] for row in counterfactual)
    if dict(cf_counts) != EXPECTED_CF:
        raise ValueError(f"counterfactual type counts differ: {dict(cf_counts)}")
    cf_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in counterfactual:
        cf_by_case[case_id(row)].append(row)
    if len(cf_by_case) != 72:
        raise ValueError("reachable action pool does not cover all 72 train cases")
    action_rows = [
        row for row in counterfactual
        if row["metadata"]["target_type"] == "counterfactual_action_selection"
    ]
    action_by_id = {row["id"]: row for row in action_rows}
    base_by_id = {row["id"]: row for row in base_core}
    endpoint_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint_pool:
        endpoint_by_case[case_id(row)].append(row)

    # This is deliberately recomputed from the serialized messages and strict
    # endpoint facts.  Metadata written by the converter is audit evidence, not
    # authority for deciding whether an action was still necessary.
    for source in base_core:
        status = recompute_structural_binding_status(
            source, endpoint_by_case[case_id(source)]
        )
        positive_tools = [
            message
            for message in positive_messages(source)
            if message.get("role") == "tool_call"
        ]
        if status["complete"] and positive_tools:
            raise ValueError(
                f"{source['id']}: complete structural evidence still supervises continuation"
            )

    structural_distributions: Counter[tuple[int, int]] = Counter()
    for action in action_rows:
        status = validate_structural_action_contract(
            action, endpoint_by_case[case_id(action)]
        )
        expected = len(status["binding"]["facts"])
        matched = len(status["matched"])
        structural_distributions[(matched, expected)] += 1
        gate = action["metadata"].get("0809_action_visible_grounding_gate", {})
        if (
            gate.get("structural_binding_complete") is not False
            or gate.get("pending_hypothesis_supervised") is not True
            or gate.get("best_binding_sha256") != status["binding"]["sha256"]
            or gate.get("best_binding_expected_fact_count") != expected
            or gate.get("best_binding_matched_fact_count") != matched
        ):
            raise ValueError(
                f"{action['id']}: structural gate metadata differs from message-derived status"
            )
    if len(action_rows) != 308 or sum(structural_distributions.values()) != 308:
        raise ValueError("not all 308 action rows passed the structural pending gate")
    if set(EXPECTED_V7_STRUCTURAL_ACTION_REPAIRS) - set(action_by_id):
        raise ValueError("a latest structural-audit action fixture disappeared")
    for action_id in EXPECTED_V7_STRUCTURAL_ACTION_REPAIRS:
        action = action_by_id[action_id]
        status = recompute_structural_binding_status(
            action, endpoint_by_case[case_id(action)]
        )
        if status["complete"] or affirmative_root_claim(
            "\n".join(
                message.get("content", "")
                for message in positive_messages(action)
                if message.get("role") == "assistant"
            )
        ):
            raise ValueError(
                f"{action_id}: latest structural-audit claim is not safely pending"
            )

    audited_source = base_by_id.get(AUDITED_COMPLETE_CONTINUATION_SOURCE)
    if audited_source is None:
        raise ValueError("audited complete-continuation source disappeared")
    audited_status = recompute_structural_binding_status(
        audited_source, endpoint_by_case[case_id(audited_source)]
    )
    audited_gate = audited_source["metadata"].get("0809_post_closure_gate", {})
    if (
        not audited_status["complete"]
        or any(
            message.get("role") == "tool_call"
            for message in positive_messages(audited_source)
        )
        or audited_gate.get("repaired") is not True
        or audited_gate.get("expected_fact_count") != 1
        or audited_gate.get("matched_fact_count") != 1
        or audited_gate.get("removed_positive_tool_call_count") != 2
        or any(
            row["metadata"].get("counterfactual_source_row_id")
            == AUDITED_COMPLETE_CONTINUATION_SOURCE
            for row in action_rows
        )
    ):
        raise ValueError(
            "audited q0023 complete-continuation source was not converted to stop"
        )
    for action_id, expected_issue in EXPECTED_AUDIT_ACTION_REPAIRS.items():
        action = action_by_id.get(action_id)
        if action is None:
            raise ValueError(f"latest-audit action fixture disappeared: {action_id}")
        gate = action["metadata"].get("0809_action_visible_grounding_gate", {})
        source = base_by_id.get(action["metadata"].get("counterfactual_source_row_id"))
        source_gate = (
            source["metadata"].get("0809_action_visible_grounding_gate", {})
            if source
            else {}
        )
        if (
            gate.get("repaired") is not True
            or expected_issue not in gate.get("original_issue_codes", [])
            or source_gate.get("repaired") is not True
            or expected_issue not in source_gate.get("original_issue_codes", [])
            or action_visible_grounding_issues(action)
            or action_visible_grounding_issues(source)
        ):
            raise ValueError(
                f"{action_id}: latest-audit claim was not repaired in both source and action"
            )
    repaired_actions = [
        row
        for row in action_rows
        if row["metadata"].get("0809_action_visible_grounding_gate", {}).get("repaired")
        is True
    ]
    repaired_sources = [
        row
        for row in base_core
        if row["metadata"].get("0809_action_visible_grounding_gate", {}).get("repaired")
        is True
    ]
    if any(action_visible_grounding_issues(row) for row in action_rows):
        raise ValueError("an action row still has unsupported pre-loss facts/convergence")
    if (
        len(repaired_actions)
        != manifest["counts"]["action_visible_grounding_action_rows_repaired"]
        or len(repaired_sources)
        != manifest["counts"]["action_visible_grounding_parent_rows_repaired"]
    ):
        raise ValueError("action visible-grounding repair counts differ from manifest")
    for current_case in cf_by_case:
        signatures = [
            (
                positive_prefix_text(row),
                tuple(row["metadata"]["selected_action_commands"]),
            )
            for row in cf_by_case[current_case]
            if row["metadata"]["target_type"] == "counterfactual_action_selection"
        ]
        if not signatures:
            raise ValueError(f"q{current_case}: reachable action contexts are empty")
    replay_rows = [
        row for row in counterfactual
        if row["metadata"]["target_type"] == "hypothesis_elimination_replay"
    ]
    for row in replay_rows:
        source = base_by_id.get(row["metadata"].get("sampling_source_row_id"))
        if (
            source is None
            or source["metadata"]["target_type"] != "hypothesis_elimination"
            or source["messages"] != row["messages"]
            or row["metadata"].get("counts_as_new_semantic_target") is not False
        ):
            raise ValueError(f"{row['id']}: elimination replay is not an exact same-target source")

    real_elimination_cases = {
        case_id(row)
        for row in base_core
        if row["metadata"]["target_type"] == "hypothesis_elimination"
    }
    train_cases = {case_id(row) for row in base_core}
    if len(real_elimination_cases) != 58:
        raise ValueError("real evidence-backed elimination coverage changed")
    if {case_id(row) for row in endpoint_pool} != train_cases:
        raise ValueError("candidate-scope endpoint coverage is incomplete")

    base_core_ids = {row["id"] for row in base_core}
    cf_ids = {row["id"] for row in counterfactual}
    endpoint_ids = {row["id"] for row in endpoint_pool}
    base_exposures: Counter[str] = Counter()
    cf_exposures: Counter[str] = Counter()
    for epoch in range(1, 6):
        core = outputs[f"core_epoch_{epoch:02d}"]
        endpoint = outputs[f"endpoint_epoch_{epoch:02d}"]
        if (len(core), len(endpoint), len(core) + len(endpoint)) != (1035, 116, 1151):
            raise ValueError(f"epoch {epoch}: schedule row budget differs from 0805")
        if math.ceil((len(core) + len(endpoint)) / 8) != 144:
            raise ValueError(f"epoch {epoch}: optimizer step budget differs")
        epoch_cf = [row for row in core if row["id"] in cf_ids]
        epoch_base = [row for row in core if row["id"] in base_core_ids]
        if len(epoch_cf) != 316 or len(epoch_base) != 719:
            raise ValueError(f"epoch {epoch}: full core/action/replay coverage is wrong")
        if {row["id"] for row in endpoint} != endpoint_ids:
            raise ValueError(f"epoch {epoch}: not all real endpoint paths are exposed")
        if {row["id"] for row in core} & {row["id"] for row in endpoint}:
            raise ValueError(f"epoch {epoch}: core and endpoint row IDs overlap")
        message_signatures = {
            json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":"))
            for row in [*core, *endpoint]
        }
        if len(message_signatures) != 1143:
            raise ValueError(f"epoch {epoch}: only eight declared same-target replays may duplicate messages")
        if any(
            row["metadata"].get("endpoint_schedule_role") != "real_evidence_path_bundle"
            for row in endpoint
        ):
            raise ValueError(f"epoch {epoch}: endpoint schedule contains a non-path row")
        base_exposures.update(row["id"] for row in epoch_base)
        cf_exposures.update(row["id"] for row in epoch_cf)
    if set(base_exposures) != base_core_ids:
        raise ValueError("some base-core rows never appear in five schedules")
    if set(cf_exposures) != cf_ids:
        raise ValueError("some counterfactual rows never appear in five schedules")
    base_case_by_id = {row["id"]: case_id(row) for row in base_core}
    cf_case_by_id = {row["id"]: case_id(row) for row in counterfactual}
    for current_case in train_cases:
        base_values = [
            base_exposures[row_id]
            for row_id, row_case in base_case_by_id.items()
            if row_case == current_case
        ]
        cf_values = [
            cf_exposures[row_id]
            for row_id, row_case in cf_case_by_id.items()
            if row_case == current_case
        ]
        if max(base_values) - min(base_values) > 1:
            raise ValueError(f"q{current_case}: base-core exposure is imbalanced")
        if max(cf_values) - min(cf_values) > 1:
            raise ValueError(f"q{current_case}: counterfactual exposure is imbalanced")

    first_three = [
        row
        for epoch in range(1, 4)
        for row in outputs[f"core_epoch_{epoch:02d}"]
        if row["id"] in cf_ids
    ]
    if len({row["id"] for row in first_three}) != 316:
        raise ValueError("fixed checkpoint has not seen every action/replay target")
    action_labels = {
        label
        for row in first_three
        if row["metadata"]["target_type"] == "counterfactual_action_selection"
        for label in row["metadata"]["correct_labels"]
    }
    expected_labels = {
        item.split(";", 1)[1]
        for row in endpoint_pool
        for item in row["metadata"]["actual_result_items"]
    }
    if action_labels != expected_labels:
        raise ValueError(
            f"epoch-3 action label coverage is incomplete: {sorted(action_labels)}"
        )

    training_entry = (ROOT / "scripts" / "train_qwen36_0809_agent_error_aware_5epoch.sh").read_text(encoding="utf-8")
    if '--external_plugins "${LR_PLUGIN_PATH}"' not in training_entry:
        raise ValueError("training entry does not import the fixed-LR external plugin")
    if "plugin_registration_smoke_check" not in training_entry:
        raise ValueError("training entry lacks callback registration smoke check")
    if 'manifest["outputs"].values()' not in training_entry or 'manifest["reproducibility"]["tracked_files"].values()' not in training_entry:
        raise ValueError("training entry does not require all manifest outputs/dependencies to be Git tracked")
    for contract in (
        "runtime model/tokenizer identity mismatch",
        "runtime ms-swift version differs",
        "runtime transformers version differs",
        "model.safetensors.index.json",
        "RUNTIME_GATE_ONLY",
    ):
        if contract not in training_entry:
            raise ValueError(f"training entry lacks runtime identity contract: {contract}")
    if '"${completed_stage}" -ge 3 && ! -d "${OUTPUT_DIR}/checkpoint-432"' not in training_entry:
        raise ValueError("training entry checks checkpoint-432 too late for stage-5 early exit")

    formal = load_json(ROOT / "config" / "qwen36_0809_formal_training.json")
    policy = formal.get("evaluation_policy", {})
    if formal.get("schema_version") != "qwen36-0809-formal-training.v3":
        raise ValueError("formal config does not use the canonical evaluation schema")
    if (
        policy.get("case_ids") != sorted(VALIDATION_CASES)
        or policy.get("repeats_per_case") != 5
        or policy.get("fixed_checkpoint") != "epoch_03/checkpoint-432"
        or policy.get("scoring_entry") != "scripts/final_answer_scoring.py"
        or policy.get("scoring_policy_version")
        != CANONICAL_SCORING_POLICY_VERSION
        or policy.get("correctness_basis") != "final_answer_only"
        or policy.get("incomplete_result_recovery")
        != "one complete fenced <result> JSON string list with no closing </result> is accepted only on exact match"
        or policy.get("launcher") != "scripts/run_agent_validation_resilient.sh"
        or policy.get("reasoning_effort") != "high"
        or policy.get("timeout_seconds_per_case_repeat") != 3600
        or policy.get("infrastructure_retry_limit") != 3
        or "inclusive-OR" not in policy.get("q73_q86_policy", "")
        or "diagnostic_only" not in policy.get(
            "process_tool_or_reasoning_errors", ""
        )
        or policy.get("terminal_without_valid_final_answer") != "model_error"
        or policy.get("clean_heldout_required_for_final_generalization_claim")
        is not True
    ):
        raise ValueError("formal Agent evaluation policy is not the frozen v4 contract")
    if manifest.get("evaluation_protocol") != policy:
        raise ValueError("manifest evaluation protocol differs from formal config")
    scorer = (ROOT / policy["scoring_entry"]).read_text(encoding="utf-8")
    if (
        re.search(
            rf'SCORING_POLICY_VERSION\s*=\s*(?:\(\s*)?["\']{re.escape(CANONICAL_SCORING_POLICY_VERSION)}["\']',
            scorer,
        )
        is None
        or "q73-q86 VRRP inclusive OR" not in scorer
    ):
        raise ValueError("canonical final-answer scorer identity/OR policy differs")
    launcher = (ROOT / policy["launcher"]).read_text(encoding="utf-8")
    for contract in (
        'REPEATS="${REPEATS:-5}"',
        'TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"',
        'REASONING_EFFORT="${REASONING_EFFORT:-high}"',
        'INFRA_MAX_RETRIES="${INFRA_MAX_RETRIES:-3}"',
        "topology=tp2x1/concurrency2",
    ):
        if contract not in launcher:
            raise ValueError(f"Agent launcher lacks frozen evaluation contract: {contract}")

    sampling = manifest["sampling"]
    if sampling["effective_rows_per_epoch"] != 1151:
        raise ValueError("manifest per-epoch row budget is stale")
    if sampling["optimizer_steps_per_epoch"] != 144:
        raise ValueError("manifest optimizer step budget is stale")
    if sampling["fixed_agent_checkpoint_epoch"] != 3 or sampling["fixed_agent_checkpoint_global_step"] != 432:
        raise ValueError("fixed checkpoint contract differs from 0805")

    preflight_record = manifest["target_tokenizer_preflight"]
    if preflight_record.get("status") == "passed":
        preflight_path = ROOT / preflight_record["path"]
        if (
            not preflight_path.is_file()
            or digest_file(preflight_path) != preflight_record["sha256_lf_normalized"]
        ):
            raise ValueError("archived tokenizer preflight is missing or stale")
        preflight = load_json(preflight_path)
        if preflight.get("schema_version") != "0809-target-tokenizer-loss-mask-preflight.v1":
            raise ValueError("unexpected tokenizer preflight schema")
        if preflight.get("status") != "passed":
            raise ValueError("manifest points to a failed tokenizer preflight")
        totals = preflight.get("totals", {})
        if (
            totals.get("over_max_length_rows") != 0
            or totals.get("loss_mask_failures") != 0
            or int(totals.get("max_tokens", 16385)) > 16384
        ):
            raise ValueError("tokenizer preflight release limits failed")
        preflight_names = [
            "train_semantic_pool",
            "validation",
            *[
                f"{kind}_epoch_{epoch:02d}"
                for epoch in range(1, 6)
                for kind in ("core", "endpoint")
            ],
        ]
        for dataset_name in preflight_names:
            dataset = preflight.get("datasets", {}).get(dataset_name, {})
            if (
                dataset.get("sha256_lf_normalized")
                != manifest["outputs"][dataset_name]["sha256_lf_normalized"]
                or dataset.get("rows") != manifest["outputs"][dataset_name]["rows"]
            ):
                raise ValueError(f"tokenizer preflight is stale for {dataset_name}")
        for epoch in range(1, 6):
            for name in (f"core_epoch_{epoch:02d}", f"endpoint_epoch_{epoch:02d}"):
                record = preflight["datasets"][name]
                if not record.get("weighted_supervised_token_sum_by_target_label"):
                    raise ValueError(f"tokenizer preflight lacks target/label signal for {name}")
        required_identity = {
            "tokenizer.json", "tokenizer_config.json", "config.json",
            "model.safetensors.index.json",
        }
        if not required_identity <= set(preflight.get("model_identity_files", {})):
            raise ValueError("tokenizer preflight lacks the model/tokenizer runtime identity set")
    elif preflight_record.get("status") != "required_before_training":
        raise ValueError("unknown tokenizer preflight release state")

    for name, record in manifest["reproducibility"]["tracked_files"].items():
        path = ROOT / record["path"]
        if not path.is_file() or digest_file(path) != record["sha256_lf_normalized"]:
            raise ValueError(f"tracked reproducibility file is stale: {name}")
    release_record = load_json(sft / "RELEASE_RECORD.json")
    if release_record["manifest"]["sha256_lf_normalized"] != digest_file(manifest_path):
        raise ValueError("release record manifest hash is stale")

    accepted = load_json(data_root / "curation" / "accepted_trajectory_selection.json")
    clusters = load_json(data_root / "curation" / "causal_path_clusters_per_case.json")
    if accepted["schema_version"] != "0809-accepted-trajectory-selection-parented-from-0805.v1":
        raise ValueError("0809 accepted-selection identity is stale")
    if clusters["schema_version"] != "0809-agent-error-aware-parent-cluster-selection.v1":
        raise ValueError("0809 cluster identity is stale")

    print(f"resolved_data_root={data_root}")
    print("0809 official validation passed")
    print(
        "semantic=1151; base_core=719; counterfactual=316; endpoint=116; "
        "validation=245"
    )
    print(
        "per_epoch=719 base_core + 308 action + 8 elimination replay + 116 endpoint = 1151 rows; "
        "144 steps; epoch-3 checkpoint=432"
    )
    print(
        "counterfactual=308 reachable action + 8 exact same-target elimination replay; "
        "strict endpoint=116 real-path reachable bundles/72 cases; no synthetic assistant marker"
    )


if __name__ == "__main__":
    main()
