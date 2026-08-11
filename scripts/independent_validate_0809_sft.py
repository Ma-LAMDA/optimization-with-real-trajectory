#!/usr/bin/env python3
"""Independent, converter-free validation of the 0809 v7 SFT release."""

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
DEV_CASES = {2, 12, 19, 20, 29, 38, 65, 71, 85, 86, 99, 100}
EVENT_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
AUDIT_ACTIONS = {
    "0809_cf_action_q0011_a01": "unsupported_global_stp_disabled",
    "0809_cf_action_q0014_a04": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0015_a01": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0018_a02": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0021_a01": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0022_a01": "unsupported_bpdu_filter_enabled",
    "0809_cf_action_q0024_a03": "unsupported_bpdu_filter_enabled",
}
STRUCTURAL_AUDIT_ACTIONS = {
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
AUDITED_COMPLETE_SOURCE = "q0023_path_02_success_09_step_04"
EXPECTED_FINAL_ANSWER_SCORER = "agent-final-answer.v3.2026-08-10-final-answer-only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def resolve_data_root(value: Path) -> Path:
    path = (value if value.is_absolute() else ROOT / value).resolve()
    expected = (ROOT / "data" / "2026-08-09").resolve()
    if path != expected:
        raise ValueError(f"expected checkout-local {expected}, received {path}")
    return path


def lf_digest(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def case_id(row: dict[str, Any]) -> int:
    return int(row["metadata"]["case_id"])


def positives(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [message for message in row["messages"] if float(message.get("loss_scale", 0) or 0) > 0]


def first_positive_prefix(row: dict[str, Any]) -> str:
    prefix: list[str] = []
    for message in row["messages"]:
        if float(message.get("loss_scale", 0) or 0) > 0:
            break
        prefix.append(message.get("content", ""))
    return "\n".join(prefix)


def independently_find_action_leaks(row: dict[str, Any]) -> set[str]:
    boundary = next(
        (i for i, message in enumerate(row["messages"]) if float(message.get("loss_scale", 0) or 0) > 0),
        len(row["messages"]),
    )
    observed = "\n".join(
        message.get("content", "")
        for message in row["messages"][:boundary]
        if message.get("role") == "tool_response"
    )
    target = "\n".join(
        message.get("content", "")
        for message in positives(row)
        if message.get("role") == "assistant"
    )
    pending = "待验证" in target or "尚不足以确认" in target
    found: set[str] = set()
    if not pending:
        disabled_claim = bool(
            re.search(r"Protocol\s+Status\s*:\s*Disabled", target, re.I)
            or re.search(r"全局.{0,24}STP.{0,24}(?:未使能|未启用|Disabled)", target, re.I)
            or re.search(r"(?:缺少|没有).{0,20}`?stp\s+enable`?", target, re.I)
        )
        if disabled_claim and not re.search(r"Protocol\s+Status\s*:\s*Disabled", observed, re.I):
            found.add("unsupported_global_stp_disabled")
        bpdu_claim = bool(
            re.search(r"stp\s+port\s+bpdu-filter\s+enable", target, re.I)
            or ("过滤" in target and re.search(r"BPDU", target, re.I))
        )
        if bpdu_claim and not re.search(r"stp\s+port\s+bpdu-filter\s+enable", observed, re.I):
            found.add("unsupported_bpdu_filter_enabled")
        nonpreempt_claim = bool(
            re.search(r"Preempt\s*:\s*NO|preempt\s+disable", target, re.I)
            or "非抢占" in target
        )
        if nonpreempt_claim and not re.search(r"Preempt\s*:\s*NO|preempt\s+disable", observed, re.I):
            found.add("unsupported_vrrp_nonpreempt")
        enabled_claim = re.search(
            r"(?:配置|启用|存在|包含).{0,20}`?stp\s+(?:global\s+)?enable`?",
            target,
            re.I,
        )
        if enabled_claim and not re.search(r"(?m)^\s*stp\s+(?:global\s+)?enable\s*$", observed, re.I):
            found.add("unsupported_stp_enabled")
    convergence_terms = (
        "证据已经形成闭环", "证据已形成闭环", "关键区分证据已出现",
        "决定性证据已出现", "唯一异常", "唯一的异常", "决定性异常", "根因应归",
    )
    formed_loop_claim = re.search(
        r"(?:已经|已)\s*(?:确认|证明)?\s*(?:形成|构成)\s*(?:IP|MPLS)?\s*(?:路由|标签|转发)?\s*环路",
        target,
        re.I,
    )
    if any(term in target for term in convergence_terms) or formed_loop_claim:
        found.add("premature_convergence_assertion")
    return found


def validate_independent_negative_fixtures() -> None:
    for expected, target in (
        ("unsupported_global_stp_disabled", "Protocol Status: Disabled"),
        ("unsupported_bpdu_filter_enabled", "配置 stp port bpdu-filter enable"),
        ("unsupported_vrrp_nonpreempt", "Preempt : NO"),
        ("premature_convergence_assertion", "关键区分证据已出现"),
        ("premature_convergence_assertion", "当前已经形成 MPLS 标签环路"),
    ):
        fixture = {
            "messages": [
                {"role": "tool_response", "content": "no decisive literal"},
                {"role": "assistant", "content": target, "loss_scale": 1.0},
            ]
        }
        if expected not in independently_find_action_leaks(fixture):
            raise ValueError(f"independent negative fixture missed {expected}")
    exclusion_fixture = {
        "messages": [
            {"role": "tool_response", "content": "route checks are visible"},
            {"role": "assistant", "content": "没有形成 IP/MPLS 环路。", "loss_scale": 1.0},
        ]
    }
    if "premature_convergence_assertion" in independently_find_action_leaks(
        exclusion_fixture
    ):
        raise ValueError("independent gate erased a negated loop exclusion")


def normalized_literal(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def independently_reconstruct_structural_closure(
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
    visible = normalized_literal(
        "\n".join(
            message.get("content", "")
            for message in row["messages"][:boundary]
            if message.get("role") == "tool_response"
        )
    )
    candidates: list[dict[str, Any]] = []
    for endpoint in endpoint_rows:
        binding = endpoint["metadata"]["evidence_binding"]
        facts = binding["facts"]
        hits = [
            fact
            for fact in facts
            if normalized_literal(str(fact["output_line"])) in visible
        ]
        candidates.append(
            {
                "binding": binding,
                "matched": hits,
                "complete": len(hits) == len(facts),
            }
        )
    if not candidates:
        raise ValueError(f"{row['id']}: independent structural gate has no endpoint")
    return max(
        candidates,
        key=lambda item: (
            int(item["complete"]),
            len(item["matched"]),
            -len(item["binding"]["facts"]),
            item["binding"]["sha256"],
        ),
    )


def independently_detect_closed_root_claim(value: str) -> bool:
    if any(word in value for word in ("待验证", "尚未", "没有形成", "未形成", "若", "如果")):
        return False
    patterns = (
        r"首尾相接.{0,24}(?:构成|形成).{0,24}(?:闭环|环路)",
        r"(?:三节点|三个节点).{0,30}(?:静态路由环|路由环|转发环|闭环)",
        r"静态标签交换.{0,30}(?:构成|形成).{0,24}(?:闭环|环路)",
        r"启用(?:了)?\s*`?BPDU\s*Filter`?",
        r"关键路径已闭合",
        r"(?:应按|直接按).{0,16}存在.{0,16}环路.{0,16}归因",
    )
    return any(re.search(pattern, value, re.I) for pattern in patterns)


def validate_structural_synonym_fixtures() -> None:
    for claim in (
        "三台PE的静态下一跳首尾相接，构成闭环。",
        "已发现两组相反方向的三节点静态路由环。",
        "三台PE上的静态标签交换关系均构成三节点闭环。",
        "上联端口启用了 BPDU Filter。",
        "关键路径已闭合，应直接归因。",
    ):
        if not independently_detect_closed_root_claim(claim):
            raise ValueError(f"independent structural fixture missed: {claim}")
    for safe in (
        "待验证假设：若三台下一跳最终首尾相接，则可能形成环路。",
        "没有形成 IP/MPLS 环路，继续核对其他候选。",
    ):
        if independently_detect_closed_root_claim(safe):
            raise ValueError(f"independent structural fixture false-positive: {safe}")


def validate_recovery_provenance(row: dict[str, Any], facts: list[dict[str, Any]]) -> None:
    records = [fact.get("source_action_provenance") for fact in facts]
    if not any(records):
        return
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{row['id']}: partial recovery provenance")
    for fact, record in zip(facts, records):
        relative = str(record["source_event_file"])
        event_path = (ROOT / relative).resolve()
        if ROOT.resolve() not in event_path.parents or not event_path.is_file():
            raise ValueError(f"{row['id']}: recovery event path escapes release")
        if lf_digest(event_path) != record.get("source_event_sha256_lf_normalized"):
            raise ValueError(f"{row['id']}: recovery event digest differs")
        if relative not in EVENT_CACHE:
            completed: dict[str, dict[str, Any]] = {}
            for line in event_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                item = event.get("item", {})
                if event.get("type") == "item.completed" and item.get("type") == "command_execution":
                    completed[str(item.get("id"))] = item
            EVENT_CACHE[relative] = completed
        item = EVENT_CACHE[relative].get(str(record.get("item_id")))
        if item is None or item.get("status") != "completed" or int(item.get("exit_code", 1)) != 0:
            raise ValueError(f"{row['id']}: recovery event item is not successful")
        command = str(item.get("command", ""))
        output = str(item.get("aggregated_output", ""))
        normalized_output = output.replace("\r\n", "\n").replace("\r", "\n")
        raw_span = str(record.get("raw_output_line_or_span", ""))
        normalized_fact = re.sub(r"\s+", " ", str(fact["output_line"])).strip()
        if (
            command != record.get("source_command")
            or text_digest(command.replace("\r\n", "\n").replace("\r", "\n"))
            != record.get("source_command_sha256_lf_normalized")
            or text_digest(normalized_output) != record.get("raw_output_sha256_lf_normalized")
            or raw_span not in output.splitlines()
            or re.sub(r"\s+", " ", raw_span).strip() != normalized_fact
            or record.get("normalized_observation") != normalized_fact
        ):
            raise ValueError(f"{row['id']}: recovery command/output/span does not round-trip")


def validate_binding(row: dict[str, Any]) -> None:
    metadata = row["metadata"]
    binding = metadata.get("evidence_binding") or {}
    facts = binding.get("facts") or []
    payload = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    visible = "\n".join(message.get("content", "") for message in row["messages"] if message.get("role") == "tool_response")
    user = "\n".join(message.get("content", "") for message in row["messages"] if message.get("role") == "user")
    snapshot = re.search(r"CampusNetwork(?:-for-perf|_\d+)", user)
    if (
        not facts
        or text_digest(payload) != binding.get("sha256")
        or not snapshot
        or binding.get("snapshot") != snapshot.group(0)
        or any(fact.get("output_line") not in visible or fact.get("snapshot") != snapshot.group(0) for fact in facts)
    ):
        raise ValueError(f"{row['id']}: literal evidence binding failed")
    label = binding.get("label")
    if metadata.get("correct_labels") != [label]:
        raise ValueError(f"{row['id']}: binding label differs")
    items = metadata.get("actual_result_items") or []
    items = [items] if isinstance(items, str) else items
    devices = {item.split(";", 1)[0] for item in items}
    kinds = Counter(fact["fact_kind"] for fact in facts)
    simple = {
        "全局STP未使能": "protocol_status_disabled",
        "STP BPDU被过滤": "bpdu_filter_current_configuration",
        "VRRP工作在非抢占模式": "vrrp_preempt_no",
    }
    if label in simple and ({fact["device"] for fact in facts} != devices or kinds != Counter({simple[label]: len(devices)})):
        raise ValueError(f"{row['id']}: simple-label closure failed")
    if label == "STP BPDU被过滤" and any(
        "display_current-configuration" not in fact["command"].lower()
        or not re.search(r"(?:^|:\d+:)stp port bpdu-filter enable\s*$", fact["output_line"], re.I)
        for fact in facts
    ):
        raise ValueError(f"{row['id']}: BPDU fact is not current configuration")
    if label in {"存在IP路由环路", "存在MPLS标签环路"}:
        route_kind = "static_route_directed_next_hop" if label == "存在IP路由环路" else "static_lsp_directed_label_hop"
        routes = [fact for fact in facts if fact["fact_kind"] == route_kind]
        owners = [fact for fact in facts if fact["fact_kind"] == "next_hop_interface_owner"]
        successor = {fact["device"]: fact.get("next_hop_owner") for fact in routes}
        start = sorted(devices)[0]
        if (
            len(routes) != 3 or len(owners) != 3 or set(successor) != devices
            or set(successor.values()) != devices or successor.get(successor.get(successor.get(start))) != start
            or {fact["device"] for fact in owners} != devices
        ):
            raise ValueError(f"{row['id']}: directed loop closure failed")
        if label == "存在IP路由环路" and len({tuple(fact["parsed_transition"])[0] for fact in routes}) != 1:
            raise ValueError(f"{row['id']}: IP loop prefixes differ")
        if label == "存在MPLS标签环路":
            by_device = {fact["device"]: fact for fact in routes}
            if any(tuple(fact["parsed_transition"])[2] != tuple(by_device[successor[fact["device"]]]["parsed_transition"])[0] for fact in routes):
                raise ValueError(f"{row['id']}: MPLS labels do not chain")
    if label == "VRRP Master角色规划不合理":
        expected = Counter({"source_host_ipv4_vlan": 1, "vrrp_master": len(devices), "mst_vlan_instance_mapping": len(devices), "mst_alternate_discarding": len(devices)})
        if kinds != expected or any({fact["device"] for fact in facts if fact["fact_kind"] == kind} != devices for kind in ("vrrp_master", "mst_vlan_instance_mapping", "mst_alternate_discarding")):
            raise ValueError(f"{row['id']}: VRRP role closure failed")
    validate_recovery_provenance(row, facts)


def main() -> None:
    validate_independent_negative_fixtures()
    validate_structural_synonym_fixtures()
    data_root = resolve_data_root(parse_args().data_root)
    sft = data_root / "sft"
    manifest_path = sft / "0809_agent_error_aware_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qwen36-0809-agent-error-aware-sft.v7":
        raise ValueError("wrong manifest schema")
    if manifest.get("data_root") != "data/2026-08-09" or "resolved_data_root_at_generation" in manifest:
        raise ValueError("manifest contains a host-specific data root")
    design = manifest.get("design_input", {})
    design_path = (ROOT / str(design.get("path", ""))).resolve()
    if ROOT.resolve() not in design_path.parents or not design_path.is_file() or lf_digest(design_path) != design.get("sha256_lf_normalized"):
        raise ValueError("design input is not closed inside the release")

    output: dict[str, list[dict[str, Any]]] = {}
    for name, record in manifest["outputs"].items():
        path = ROOT / record["path"]
        if not path.is_file() or lf_digest(path) != record["sha256_lf_normalized"]:
            raise ValueError(f"{name}: missing/stale release output")
        output[name] = load_rows(path)
        if len(output[name]) != record["rows"]:
            raise ValueError(f"{name}: row count differs")

    train = output["train_semantic_pool"]
    base = output["base_core_pool"]
    counterfactual = output["counterfactual_pool"]
    endpoint = output["endpoint_pool"]
    validation = output["validation"]
    if tuple(map(len, (train, base, counterfactual, endpoint, validation))) != (1151, 719, 316, 116, 245):
        raise ValueError("primary pool counts changed")
    train_cases = {case_id(row) for row in train}
    if {case_id(row) for row in validation} != DEV_CASES or train_cases & DEV_CASES or len(train_cases) != 72:
        raise ValueError("frozen train/dev split failed")
    banned = ("当前可见状态：下一节点", "当前证据尚未闭环；下一节点", "下一节点只核对", "下一节点只保留")
    for row in train:
        if row["metadata"].get("error_mining_agent_trajectory_used") is not False:
            raise ValueError(f"{row['id']}: evaluation trajectory provenance entered training")
        if row["metadata"].get("synthetic_assistant_control_marker_used") is True or any(token in message.get("content", "") for token in banned for message in row["messages"]):
            raise ValueError(f"{row['id']}: unreachable assistant control marker remains")
        user = "\n".join(message.get("content", "") for message in row["messages"] if message.get("role") == "user")
        snapshot = re.search(r"CampusNetwork(?:-for-perf|_\d+)", user)
        if not snapshot:
            raise ValueError(f"{row['id']}: snapshot absent")
        for message in row["messages"]:
            if message.get("role") != "tool_call":
                continue
            command = json.loads(message["content"])["arguments"]["cmd"]
            if set(re.findall(r"saved_configs/([^/\s]+)", command)) != {snapshot.group(0)} or re.search(r"saved_configs/[^\s]*(?:\*|\?|\[)", command):
                raise ValueError(f"{row['id']}: cross-snapshot/glob command remains")

    repaired = [row for row in base if row["metadata"].get("0809_post_closure_gate", {}).get("repaired") is True]
    if len(repaired) != 50:
        raise ValueError("post-closure repair count changed")
    for row in repaired:
        validate_binding(row)
        if any(message.get("role") == "tool_call" and float(message.get("loss_scale", 0) or 0) > 0 for message in row["messages"]):
            raise ValueError(f"{row['id']}: positive tool remains after complete evidence")

    objectives = {"evidence_summary", "label_boundary", "candidate_scope_calibration", "minimal_set", "decision_ready", "decision"}
    paths_per_case = Counter(case_id(row) for row in endpoint)
    if Counter(paths_per_case.values()) != Counter({1: 47, 2: 9, 3: 13, 4: 3}):
        raise ValueError(f"real-path distribution differs: {paths_per_case}")
    endpoint_prefixes: set[str] = set()
    for row in endpoint:
        metadata = row["metadata"]
        if metadata.get("dataset_type") != "0809_strict_evidence_endpoint" or metadata.get("target_type") != "endpoint_bundle" or metadata.get("counts_as_real_evidence_path") is not True:
            raise ValueError(f"{row['id']}: endpoint is not a declared real path")
        validate_binding(row)
        prefix = first_positive_prefix(row)
        endpoint_prefixes.add(prefix)
        if text_digest(prefix) != metadata.get("raw_first_positive_prefix_sha256"):
            raise ValueError(f"{row['id']}: raw-prefix digest differs")
        if set(metadata.get("endpoint_objectives", [])) != objectives or metadata.get("endpoint_chain_history_policy") != "single_reachable_completion_from_raw_context":
            raise ValueError(f"{row['id']}: endpoint bundle contract differs")
        current_positive = positives(row)
        positive_text = "\n".join(message.get("content", "") for message in current_positive)
        if (
            metadata.get("rejected_labels") != []
            or not metadata.get("unverified_neighbor_labels")
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
            raise ValueError(f"{row['id']}: endpoint asserts unsupported closed-world exclusion")
        if len(current_positive) != 3 or any(message["role"] == "tool_call" for message in current_positive) or "<result>" not in current_positive[-1].get("content", ""):
            raise ValueError(f"{row['id']}: endpoint completion is malformed")
        expected_weight = 1 / paths_per_case[case_id(row)]
        if metadata.get("query_raw_evidence_path_count") != paths_per_case[case_id(row)] or not math.isclose(float(metadata.get("query_normalized_path_weight", 0)), expected_weight, abs_tol=1e-8):
            raise ValueError(f"{row['id']}: query path normalization differs")
    if len(endpoint_prefixes) != 116:
        raise ValueError("116 endpoint objectives are not 116 distinct raw paths")

    cf_types = Counter(row["metadata"]["target_type"] for row in counterfactual)
    if cf_types != Counter({"counterfactual_action_selection": 308, "hypothesis_elimination_replay": 8}):
        raise ValueError(f"counterfactual inventory changed: {cf_types}")
    actions = [row for row in counterfactual if row["metadata"]["target_type"] == "counterfactual_action_selection"]
    actions_by_id = {row["id"]: row for row in actions}
    base_by_id = {row["id"]: row for row in base}
    endpoint_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint:
        endpoint_by_case[case_id(row)].append(row)

    for row in base:
        status = independently_reconstruct_structural_closure(
            row, endpoint_by_case[case_id(row)]
        )
        positive_tools = [
            message
            for message in positives(row)
            if message.get("role") == "tool_call"
        ]
        if status["complete"] and positive_tools:
            raise ValueError(f"{row['id']}: complete evidence still continues investigation")
    action_signatures: dict[int, set[tuple[str, tuple[str, ...]]]] = defaultdict(set)
    for row in actions:
        commands = tuple(row["metadata"].get("selected_action_commands", []))
        calls = [json.loads(message["content"])["arguments"]["cmd"] for message in positives(row) if message.get("role") == "tool_call"]
        if not 1 <= len(commands) <= 2 or list(commands) != calls or row["metadata"].get("complete_binding_before_positive_tool_call") is not False:
            raise ValueError(f"{row['id']}: reachable action target is malformed")
        if (
            row["metadata"].get("action_visible_grounding_checked") is not True
            or row["metadata"].get("action_visible_grounding_remaining_issues") != []
            or independently_find_action_leaks(row)
        ):
            raise ValueError(f"{row['id']}: unsupported fact/convergence remains before action")
        status = independently_reconstruct_structural_closure(
            row, endpoint_by_case[case_id(row)]
        )
        expected = len(status["binding"]["facts"])
        matched = len(status["matched"])
        assistant_target = "\n".join(
            message.get("content", "")
            for message in positives(row)
            if message.get("role") == "assistant"
        )
        gate = row["metadata"].get("0809_action_visible_grounding_gate", {})
        if (
            status["complete"]
            or "待验证假设" not in assistant_target
            or f"{matched}/{expected}" not in assistant_target
            or "结果返回前不输出最终答案" not in assistant_target
            or independently_detect_closed_root_claim(assistant_target)
            or gate.get("structural_binding_complete") is not False
            or gate.get("pending_hypothesis_supervised") is not True
            or gate.get("best_binding_sha256") != status["binding"]["sha256"]
            or gate.get("best_binding_expected_fact_count") != expected
            or gate.get("best_binding_matched_fact_count") != matched
        ):
            raise ValueError(f"{row['id']}: action is not an independently verified pending state")
        signature = (first_positive_prefix(row), commands)
        action_signatures[case_id(row)].add(signature)
    if set(action_signatures) != train_cases:
        raise ValueError("actions do not cover all 72 training queries")
    if STRUCTURAL_AUDIT_ACTIONS - set(actions_by_id):
        raise ValueError("latest structural-audit fixture disappeared")
    for action_id in STRUCTURAL_AUDIT_ACTIONS:
        row = actions_by_id[action_id]
        if independently_reconstruct_structural_closure(
            row, endpoint_by_case[case_id(row)]
        )["complete"]:
            raise ValueError(f"{action_id}: structural-audit action remains complete")

    audited_source = base_by_id.get(AUDITED_COMPLETE_SOURCE)
    if audited_source is None:
        raise ValueError("audited q0023 continuation source disappeared")
    audited_status = independently_reconstruct_structural_closure(
        audited_source, endpoint_by_case[case_id(audited_source)]
    )
    audited_gate = audited_source["metadata"].get("0809_post_closure_gate", {})
    if (
        not audited_status["complete"]
        or any(message.get("role") == "tool_call" for message in positives(audited_source))
        or audited_gate.get("repaired") is not True
        or audited_gate.get("expected_fact_count") != 1
        or audited_gate.get("matched_fact_count") != 1
        or audited_gate.get("removed_positive_tool_call_count") != 2
        or any(
            row["metadata"].get("counterfactual_source_row_id") == AUDITED_COMPLETE_SOURCE
            for row in actions
        )
    ):
        raise ValueError("audited q0023 continuation was not replaced by stop")

    for action_id, expected_issue in AUDIT_ACTIONS.items():
        action = actions_by_id.get(action_id)
        if action is None:
            raise ValueError(f"latest-audit fixture is absent: {action_id}")
        source = base_by_id.get(action["metadata"].get("counterfactual_source_row_id"))
        action_gate = action["metadata"].get("0809_action_visible_grounding_gate", {})
        source_gate = source["metadata"].get("0809_action_visible_grounding_gate", {}) if source else {}
        if (
            expected_issue not in action_gate.get("original_issue_codes", [])
            or expected_issue not in source_gate.get("original_issue_codes", [])
            or independently_find_action_leaks(action)
            or independently_find_action_leaks(source)
        ):
            raise ValueError(f"{action_id}: audited leakage was not fixed at both levels")
    repaired_action_count = sum(
        row["metadata"].get("0809_action_visible_grounding_gate", {}).get("repaired") is True
        for row in actions
    )
    repaired_parent_count = sum(
        row["metadata"].get("0809_action_visible_grounding_gate", {}).get("repaired") is True
        for row in base
    )
    if (
        repaired_action_count != manifest["counts"]["action_visible_grounding_action_rows_repaired"]
        or repaired_parent_count != manifest["counts"]["action_visible_grounding_parent_rows_repaired"]
    ):
        raise ValueError("independent action-grounding repair counts differ from manifest")

    replays = [row for row in counterfactual if row["metadata"]["target_type"] == "hypothesis_elimination_replay"]
    for row in replays:
        source = base_by_id.get(row["metadata"].get("sampling_source_row_id"))
        if source is None or source["metadata"].get("target_type") != "hypothesis_elimination" or source["messages"] != row["messages"] or row["metadata"].get("counts_as_new_semantic_target") is not False:
            raise ValueError(f"{row['id']}: replay is not an exact same-target source")
    real_elimination_cases = {case_id(row) for row in base if row["metadata"].get("target_type") == "hypothesis_elimination"}
    if len(real_elimination_cases) != 58:
        raise ValueError("real evidence-backed elimination coverage differs")
    if {case_id(row) for row in endpoint} != train_cases:
        raise ValueError("candidate-scope endpoint coverage differs")

    base_ids = set(base_by_id)
    cf_ids = {row["id"] for row in counterfactual}
    endpoint_ids = {row["id"] for row in endpoint}
    for epoch in range(1, 6):
        core = output[f"core_epoch_{epoch:02d}"]
        endpoint_epoch = output[f"endpoint_epoch_{epoch:02d}"]
        if (len(core), len(endpoint_epoch), len(core) + len(endpoint_epoch)) != (1035, 116, 1151):
            raise ValueError(f"epoch {epoch}: allocation differs")
        if math.ceil((len(core) + len(endpoint_epoch)) / 8) != 144:
            raise ValueError(f"epoch {epoch}: optimizer-step budget differs")
        if {row["id"] for row in core} != base_ids | cf_ids or {row["id"] for row in endpoint_epoch} != endpoint_ids:
            raise ValueError(f"epoch {epoch}: full target exposure differs")
        signatures = {json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")) for row in [*core, *endpoint_epoch]}
        if len(signatures) != 1143:
            raise ValueError(f"epoch {epoch}: only eight declared replays may duplicate payloads")
        if any(row["metadata"].get("endpoint_schedule_role") != "real_evidence_path_bundle" for row in endpoint_epoch):
            raise ValueError(f"epoch {epoch}: endpoint schedule includes a non-path row")
    exposed_before_fixed = {row["id"] for epoch in range(1, 4) for row in output[f"core_epoch_{epoch:02d}"] if row["id"] in cf_ids}
    if exposed_before_fixed != cf_ids:
        raise ValueError("fixed epoch-3 checkpoint misses a new target")

    training = (ROOT / "scripts" / "train_qwen36_0809_agent_error_aware_5epoch.sh").read_text(encoding="utf-8")
    contracts = (
        '--external_plugins "${LR_PLUGIN_PATH}"', "plugin_registration_smoke_check",
        'manifest["outputs"].values()', 'manifest["reproducibility"]["tracked_files"].values()',
        "runtime model/tokenizer identity mismatch", "runtime ms-swift version differs",
        "runtime transformers version differs", "model.safetensors.index.json", "RUNTIME_GATE_ONLY",
        '"${completed_stage}" -ge 3 && ! -d "${OUTPUT_DIR}/checkpoint-432"',
    )
    if any(contract not in training for contract in contracts):
        raise ValueError("formal training entry lacks a release/runtime contract")
    formal = json.loads((ROOT / "config" / "qwen36_0809_formal_training.json").read_text(encoding="utf-8"))
    if formal.get("schema_version") != "qwen36-0809-formal-training.v3" or formal.get("check_model") is not True:
        raise ValueError("formal config does not bind model identity")
    evaluation = formal.get("evaluation_policy", {})
    independent_contract = {
        "case_ids": sorted(DEV_CASES),
        "repeats_per_case": 5,
        "fixed_checkpoint": "epoch_03/checkpoint-432",
        "scoring_entry": "scripts/final_answer_scoring.py",
        "scoring_policy_version": EXPECTED_FINAL_ANSWER_SCORER,
        "correctness_basis": "final_answer_only",
        "launcher": "scripts/run_agent_validation_resilient.sh",
        "reasoning_effort": "high",
        "timeout_seconds_per_case_repeat": 3600,
        "infrastructure_retry_limit": 3,
    }
    if any(evaluation.get(key) != value for key, value in independent_contract.items()):
        raise ValueError("formal config does not bind the shared v3 Agent protocol")
    if (
        "inclusive-OR" not in evaluation.get("q73_q86_policy", "")
        or "diagnostic_only" not in evaluation.get(
            "process_tool_or_reasoning_errors", ""
        )
        or evaluation.get("terminal_without_valid_final_answer") != "model_error"
        or manifest.get("evaluation_protocol") != evaluation
    ):
        raise ValueError("Agent scoring semantics differ from the final-answer-only contract")
    scorer_path = ROOT / evaluation["scoring_entry"]
    scorer_source = scorer_path.read_text(encoding="utf-8")
    if (
        f'SCORING_POLICY_VERSION = "{EXPECTED_FINAL_ANSWER_SCORER}"'
        not in scorer_source
        or "q73-q86 VRRP inclusive OR" not in scorer_source
    ):
        raise ValueError("shared final-answer scorer implementation differs")
    launcher_source = (ROOT / evaluation["launcher"]).read_text(encoding="utf-8")
    for required in (
        'REPEATS="${REPEATS:-5}"',
        'TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"',
        'REASONING_EFFORT="${REASONING_EFFORT:-high}"',
        'INFRA_MAX_RETRIES="${INFRA_MAX_RETRIES:-3}"',
        "topology=tp2x1/concurrency2",
    ):
        if required not in launcher_source:
            raise ValueError(f"shared Agent launcher contract is absent: {required}")

    preflight_record = manifest["target_tokenizer_preflight"]
    if preflight_record.get("status") == "passed":
        preflight_path = ROOT / preflight_record["path"]
        if not preflight_path.is_file() or lf_digest(preflight_path) != preflight_record["sha256_lf_normalized"]:
            raise ValueError("target tokenizer preflight is missing/stale")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        totals = preflight.get("totals", {})
        if preflight.get("status") != "passed" or totals.get("over_max_length_rows") != 0 or totals.get("loss_mask_failures") != 0 or int(totals.get("max_tokens", 16385)) > 16384:
            raise ValueError("target tokenizer/loss-mask preflight failed")
        for name in ("train_semantic_pool", "validation", *[f"{kind}_epoch_{epoch:02d}" for epoch in range(1, 6) for kind in ("core", "endpoint")]):
            if preflight["datasets"][name]["sha256_lf_normalized"] != manifest["outputs"][name]["sha256_lf_normalized"]:
                raise ValueError(f"{name}: tokenizer preflight differs from release")
        if not {"tokenizer.json", "tokenizer_config.json", "config.json", "model.safetensors.index.json"} <= set(preflight.get("model_identity_files", {})):
            raise ValueError("preflight lacks runtime identity files")
    elif preflight_record.get("status") != "required_before_training":
        raise ValueError("unexpected tokenizer preflight state")

    for record in manifest["reproducibility"]["tracked_files"].values():
        path = ROOT / record["path"]
        if not path.is_file() or lf_digest(path) != record["sha256_lf_normalized"]:
            raise ValueError(f"tracked dependency missing/stale: {record['path']}")
    release = json.loads((sft / "RELEASE_RECORD.json").read_text(encoding="utf-8"))
    if release["manifest"]["sha256_lf_normalized"] != lf_digest(manifest_path):
        raise ValueError("release record manifest digest differs")

    print(f"resolved_data_root={data_root}")
    print("0809 independent validation passed")
    print("semantic=1151; base=719; action=308; elimination_replay=8; endpoint_real_paths=116; validation=245")
    print("per_epoch=1151 rows; 144 steps; all new targets precede fixed epoch-3 checkpoint=432")


if __name__ == "__main__":
    main()
