#!/usr/bin/env python3
"""Strict 0807 release validation without importing either 0807 generator."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2026-08-07"
MANIFEST = DATA / "sft" / "0807_evidence_gated_manifest.json"
SELECTION = DATA / "curation" / "causal_path_clusters_per_case.json"
CURATION = DATA / "curation" / "accepted_trajectory_selection.json"
SOURCE_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
VALID_CASES = set(range(1, 101)) - set(range(41, 57))
INCLUSIVE_OR_CASES = set(range(73, 87))
ROLE_REASON = "VRRP Master角色规划不合理"
KNOWN_BAD_ELIMINATIONS = {
    "q0004_path_05_success_08_step_03",
    "q0008_path_04_success_06_step_03",
    "q0032_path_01_success_06_step_03",
}
KNOWN_UNSUPPORTED_REASONING = {
    "q0005_path_04_success_01_step_05",
    "q0006_path_08_success_04_step_03",
    "q0007_path_06_success_05_step_03",
    "q0011_path_05_success_05_step_04",
    "q0022_path_04_success_10_step_02",
    "q0062_path_01_success_06_step_04",
    "q0072_path_01_success_01_step_03",
    "q0065_path_01_success_01_step_03",
    "q0008_path_01_success_10_step_02",
    "q0024_path_03_success_03_step_02",
    "q0039_path_01_success_10_step_02",
    "q0073_path_01_success_10_step_04",
    "q0075_path_05_success_09_step_02",
    "q0083_path_01_success_05_step_02",
    "q0087_path_01_success_06_step_03",
    "q0089_path_02_success_05_step_03",
    "q0090_path_04_success_04_step_03",
    "q0091_path_01_success_09_step_03",
    "q0092_path_02_success_07_step_03",
    "q0099_path_04_success_02_step_03",
    "q0100_path_04_success_07_step_03",
}
PATH_RE = re.compile(
    r"saved_configs/([^/\s'\"]+)/([^/\s'\"]+)/([^\s'\"]+)", re.IGNORECASE
)
GLOB_MARKERS = "*?["
INDEPENDENT_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lldp", ("lldp",)), ("bpdu", ("bpdu",)), ("stp", ("mstp", "stp")),
    ("vrrp", ("vrrp", "master", "backup", "preempt")),
    ("mpls", ("mpls", "lsp", "ldp", "label")),
    ("srv6", ("srv6", "segment-routing", "policy")), ("isis", ("isis",)),
    ("ospf", ("ospf",)), ("bgp", ("bgp", "vpnv4")),
    ("routing", ("routing-table", "route", "routing", "traceroute")),
    ("interface", ("interface", "eth-trunk", "port")), ("arp", ("arp",)),
    ("mac", ("mac-address", "mac address")),
    ("config", ("current-configuration", "configuration", "config")),
    ("log", ("logbuffer", "log ", "log'", 'log"')), ("alarm", ("alarm",)),
    ("cpu", ("cpu",)), ("memory", ("memory",)),
)
PURE_PROCEDURAL_PREFIX = re.compile(
    r"^(?:我会|我将|我先|下一步|接下来|需要|继续|准备|计划|先检查|先查看|"
    r"验证是否|确认是否|排查|读取|搜索|对比|优先|"
    r"(?:再|随后|然后)(?:我会|检查|查看|核对|验证|确认|排查|读取|搜索|对比))"
)
PROCEDURAL_FACT_MARKERS = (
    "已经", "已确认", "已发现", "我已", "发现", "显示", "表明", "可见", "说明",
    "均有", "保存了", "启用了", "开启了", "关闭了", "收敛到", "因此", "所以",
    "根因是", "可以确定", "直接支持", "证明",
)
KNOWN_LLDP_REGRESSION_ROWS = {
    "q0021_path_07_success_06_step_03", "q0023_path_01_success_10_step_04",
    "q0025_path_01_success_01_step_02", "q0027_path_01_success_05_step_03",
    "q0033_path_01_success_07_step_02", "q0033_path_01_success_07_step_03",
    "q0033_path_01_success_07_step_06", "q0063_path_01_success_09_step_03",
    "q0087_path_03_success_03_step_02", "q0089_path_02_success_05_step_02",
    "q0097_path_01_success_07_step_02", "q0019_path_01_success_08_step_02",
    "q0065_path_01_success_01_step_02",
}
UNSAFE_PROCEDURAL_TARGET_FRAGMENTS = (
    "路径核对发现关键局部异常", "我已把候选范围收敛到路径接口",
    "沿途邻接均有对应运行态文件",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def infer_families_independently(text: str) -> set[str]:
    lowered = re.sub(r"\bsaved_configs\b", "", text.lower())
    families: set[str] = set()
    for family, patterns in INDEPENDENT_FAMILY_PATTERNS:
        for pattern in patterns:
            if re.fullmatch(r"[a-z0-9]+", pattern):
                matched = re.search(
                    rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", lowered
                ) is not None
            else:
                matched = pattern in lowered
            if matched:
                families.add(family)
                break
    if any(marker in lowered for marker in (
        "get-childitem", "get-location", "test-path", "test -e"
    )):
        families.add("discovery")
    return families or {"other"}


def is_pure_procedural_independently(sentence: str) -> bool:
    stripped = sentence.strip().lstrip("而并且且但、，,；;：: ")
    return (
        PURE_PROCEDURAL_PREFIX.match(stripped) is not None
        and not any(marker in stripped for marker in PROCEDURAL_FACT_MARKERS)
    )


def option_key(items: list[str]) -> frozenset[str]:
    return frozenset(str(item) for item in items)


def inclusive_or_options_by_case() -> dict[int, list[list[str]]]:
    result: dict[int, list[list[str]]] = {}
    for row in load_jsonl(SOURCE_DATASET):
        case_id = int(row["id"])
        if case_id not in INCLUSIVE_OR_CASES:
            continue
        value = json.loads(row["answer"])
        if not isinstance(value, list):
            raise ValueError(f"q{case_id}: inclusive-OR answer is not nested")
        options = [[str(item) for item in option] for option in value]
        if [len(option) for option in options] != [1, 1, 2]:
            raise ValueError(f"q{case_id}: inclusive-OR answer must be A, B, A+B")
        answer_a, answer_b = options[0][0], options[1][0]
        if (
            answer_a != f"Core_SW_01;{ROLE_REASON}"
            or answer_b != f"Core_SW_02;{ROLE_REASON}"
            or option_key(options[2]) != frozenset({answer_a, answer_b})
            or len({option_key(option) for option in options}) != 3
        ):
            raise ValueError(f"q{case_id}: malformed inclusive-OR options")
        result[case_id] = options
    if set(result) != INCLUSIVE_OR_CASES:
        raise ValueError("source dataset is missing q73-q86")
    return result


def validate_reference_sync(curation: dict[str, Any]) -> None:
    if curation.get("schema_version") != "codex-ip-accepted-trajectory-curation.v5":
        raise ValueError("curation is not inclusive-OR schema v5")
    source_hash = digest_file(SOURCE_DATASET)
    if (
        curation.get("source_dataset") != SOURCE_DATASET.relative_to(ROOT).as_posix()
        or curation.get("source_dataset_sha256_lf_normalized") != source_hash
    ):
        raise ValueError("curation source dataset provenance is stale")
    options_by_case = inclusive_or_options_by_case()
    affected = dual = 0
    for item in curation["trajectories"]:
        if not item.get("selected") or int(item["case_id"]) not in INCLUSIVE_OR_CASES:
            continue
        affected += 1
        case_id = int(item["case_id"])
        raw_path = ROOT / str(item["raw_file"])
        if digest_file(raw_path) != item["raw_sha256_lf_normalized"]:
            raise ValueError(f"{item['id']}: synchronized raw hash is stale")
        raw = load_json(raw_path)
        options = options_by_case[case_id]
        if (
            item.get("reference_answer_options") != options
            or raw.get("reference_answer_options") != options
            or not item.get("reference_answer_revision")
            or not raw.get("reference_answer_revision")
            or raw.get("answer_matches_reference") is not True
        ):
            raise ValueError(f"{item['id']}: inclusive-OR archive fields disagree")
        prediction = [str(value) for value in raw.get("actual_result_items") or []]
        if option_key(prediction) not in {option_key(option) for option in options}:
            raise ValueError(f"{item['id']}: accepted result misses all OR options")
        dual += len(prediction) == 2
    policy = curation.get("reference_answer_policy") or {}
    if (
        affected != 140
        or dual != 0
        or policy.get("accepted_options_per_case") != 3
        or policy.get("updated_trajectory_count") != 140
        or policy.get("existing_dual_device_trajectory_count") != 0
        or policy.get("split_or_success_count_changed") is not False
    ):
        raise ValueError("inclusive-OR reference accounting is inconsistent")


def path_semantics(command: str) -> list[dict[str, Any]]:
    results = []
    for snapshot, device, filename in PATH_RE.findall(command):
        results.append({
            "snapshot": snapshot.lower(),
            "device": device.lower(),
            "filename": filename.lower(),
            "snapshot_glob": any(marker in snapshot for marker in GLOB_MARKERS),
            "device_glob": any(marker in device for marker in GLOB_MARKERS),
            "filename_glob": any(marker in filename for marker in GLOB_MARKERS),
        })
    return results


def visible_before_supervision(row: dict[str, Any]) -> str:
    visible: list[str] = []
    for message in row["messages"]:
        if (
            message["role"] == "assistant"
            and float(message.get("loss_scale", 0) or 0) > 0
        ):
            break
        if message["role"] in {"tool_call", "tool_response"}:
            visible.append(str(message["content"]))
    return normalized("\n".join(visible))


def positive_assistant(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message for message in row["messages"]
        if message["role"] == "assistant"
        and float(message.get("loss_scale", 0) or 0) > 0
    ]


def is_substantive_span(span: str) -> bool:
    lowered = normalized(span)
    if (
        not lowered
        or "unselected lines omitted" in lowered
        or lowered in {
            "(empty output)", "output:",
            "port vrf status ip address speed mtu",
            "mstid port role stp state protection cost edged",
            "interface physical protocol ip address description",
        }
        or lowered.startswith((
            "flag ", "flags:", "legend", "route flags:", "peer information",
            "route information", "lsp information", "display ", "chunk id:",
            "wall time:", "process exited", "original token count:",
        ))
    ):
        return False
    return bool(re.search(
        r"(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?|\d+%|"
        r"(?:ethernet|eth-trunk|vlanif|ge)\d+(?:[/_.-]\d+)*|"
        r"\b(?:up|down|enabled?|disabled?|master|backup|active|inactive|"
        r"error|drop|crc|mtu|cost|preempt|direct|ibgp)\b",
        lowered,
    ))


def contains_anchor(text: str, anchor: str) -> bool:
    anchor = anchor.lower()
    if re.fullmatch(r"[a-z][a-z0-9_./<>:-]*", anchor):
        return bool(re.search(
            rf"(?<![a-z0-9_]){re.escape(anchor)}(?![a-z0-9_])", text
        ))
    return anchor in text


def validate_regression_fixtures() -> None:
    bad = [
        (
            ["0%", "ethernet1/0/1"],
            ["Flag after LDP FRR: (L) - Logic FRR LSP"],
        ),
        (
            ["10.10.0.1", "ethernet1/0/1"],
            ["Route information for ISIS(1)", "LSP Information"],
        ),
        (
            ["mtu 1500", "crc", "up"],
            ["Port VRF Status IP Address Speed MTU", "Flags: FHRP router"],
        ),
        (
            ["isis", "cost"],
            ["... unselected lines omitted ..."],
        ),
    ]
    for anchors, spans in bad:
        substantive = [normalized(span) for span in spans if is_substantive_span(span)]
        if substantive and all(
            any(contains_anchor(span, anchor) for span in substantive)
            for anchor in anchors
        ):
            raise ValueError("unsupported elimination regression fixture was accepted")
    glob_fixture = path_semantics(
        "grep -n -- saved_configs/CampusNetwork_07/PE*/*route*.txt"
    )
    if not glob_fixture or not (
        glob_fixture[0]["device_glob"] and glob_fixture[0]["filename_glob"]
    ):
        raise ValueError("device/filename glob regression fixture was not detected")
    old_q86_wrong_vlan = [
        {"kind": "source_host_ipv4", "device": "guest_wifi_client03", "vlan": 120},
        {"kind": "vrrp_master", "device": "core_sw_01", "vlan": 30, "action_id": "A"},
        {"kind": "mst_vlan_instance_mapping", "device": "core_sw_01", "instance": 2, "vlans": [30, 40]},
        {"kind": "stp_alternate_discarding", "device": "core_sw_01", "instance": 2},
    ]
    wrong_instance_zero = [
        {"kind": "source_host_ipv4", "device": "guest_wifi_client03", "vlan": 120},
        {"kind": "vrrp_master", "device": "core_sw_02", "vlan": 120, "action_id": "A"},
        {"kind": "mst_vlan_instance_mapping", "device": "core_sw_02", "instance": 3, "vlans": [100, 110, 120]},
        {"kind": "stp_alternate_discarding", "device": "core_sw_02", "instance": 0},
    ]
    if role_closure_devices(old_q86_wrong_vlan, 120) or role_closure_devices(
        wrong_instance_zero, 120
    ):
        raise ValueError("legacy arbitrary-VLAN/instance role gate regression was accepted")


def validate_eliminations(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        metadata = row["metadata"]
        if metadata["target_type"] != "hypothesis_elimination":
            continue
        count += 1
        if row["id"] in KNOWN_BAD_ELIMINATIONS:
            raise ValueError(f"{row['id']}: known unsupported elimination survived")
        bindings = metadata.get("elimination_claim_bindings")
        facts = metadata.get("supervised_elimination_claims")
        if not bindings or len(bindings) != len(facts or []):
            raise ValueError(f"{row['id']}: claim bindings are incomplete")
        target = normalized("\n".join(message["content"] for message in positive_assistant(row)))
        visible = visible_before_supervision(row)
        bound_ids: set[str] = set()
        for binding, fact in zip(bindings, facts):
            if binding.get("supervised_fact") != fact or normalized(fact) not in target:
                raise ValueError(f"{row['id']}: supervised fact mismatch")
            evidence = binding.get("evidence") or []
            anchors = [str(value).lower() for value in binding.get("anchors") or []]
            if not evidence or not anchors:
                raise ValueError(f"{row['id']}: empty evidence or anchors")
            combined = normalized(" ".join(item["observation_span"] for item in evidence))
            if not all(contains_anchor(combined, anchor) for anchor in anchors):
                raise ValueError(f"{row['id']}: an atomic claim anchor is ungrounded")
            for item in evidence:
                span = str(item.get("observation_span") or "")
                bound_ids.add(str(item.get("action_id") or ""))
                if not is_substantive_span(span) or normalized(span) not in visible:
                    raise ValueError(f"{row['id']}: span is a header/empty/future result")
        if bound_ids != set(metadata.get("elimination_evidence_action_ids") or []):
            raise ValueError(f"{row['id']}: action binding set mismatch")
    return count


def validate_reasoning_bindings(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Independently require exact earlier output spans for ordinary facts."""
    bound = procedural = 0
    for row in rows:
        if row["metadata"]["target_type"] == "hypothesis_elimination":
            continue
        target = normalized("\n".join(
            message["content"] for message in positive_assistant(row)
        ))
        visible = visible_before_supervision(row)
        records = row["metadata"].get("text_optimization", {}).get(
            "retained_thinking_sentence_records", []
        )
        for record in records:
            kind = str(record.get("kind") or "")
            sentence = normalized(str(record.get("sentence") or ""))
            if not sentence or sentence not in target:
                raise ValueError(f"{row['id']}: reasoning record is absent from target")
            if kind == "procedural_plan":
                procedural += 1
                if record.get("action_ids") or record.get("claim_bindings"):
                    raise ValueError(f"{row['id']}: procedural record has factual evidence")
                if not is_pure_procedural_independently(
                    str(record.get("source_sentence") or "")
                ):
                    raise ValueError(f"{row['id']}: mixed factual procedural record survived")
                continue
            if kind != "observation_bound_reasoning":
                raise ValueError(f"{row['id']}: legacy keyword-grounded reasoning survived")
            bound += 1
            source = normalized(str(record.get("source_sentence") or ""))
            if not source or source == sentence or source in target:
                raise ValueError(f"{row['id']}: source inference still participates in loss")
            bindings = record.get("claim_bindings") or []
            if not bindings or not record.get("action_ids"):
                raise ValueError(f"{row['id']}: ordinary reasoning lacks atomic bindings")
            bound_ids: set[str] = set()
            for binding in bindings:
                claim = normalized(str(binding.get("claim") or ""))
                evidence = binding.get("evidence") or []
                if not claim or claim not in sentence or not evidence:
                    raise ValueError(f"{row['id']}: ordinary factual claim is incomplete")
                for item in evidence:
                    action_id = str(item.get("action_id") or "")
                    span = str(item.get("observation_span") or "")
                    bound_ids.add(action_id)
                    if not action_id or not is_substantive_span(span):
                        raise ValueError(f"{row['id']}: empty/header reasoning span survived")
                    if normalized(span) not in visible:
                        raise ValueError(f"{row['id']}: reasoning span is not earlier visible output")
            if bound_ids != set(record.get("action_ids") or []):
                raise ValueError(f"{row['id']}: reasoning action binding set mismatch")
        if row["id"] in KNOWN_UNSUPPORTED_REASONING:
            for record in records:
                source = normalized(str(record.get("source_sentence") or ""))
                if source and source in target:
                    raise ValueError(f"{row['id']}: known unsupported reasoning regressed")
        supervised = "\n".join(
            str(message["content"]) for message in positive_assistant(row)
        )
        if any(fragment in supervised for fragment in UNSAFE_PROCEDURAL_TARGET_FRAGMENTS):
            raise ValueError(f"{row['id']}: known procedural bypass leaked into loss")
    return bound, procedural


def validate_cycle(facts: list[dict[str, Any]], *, mpls: bool) -> None:
    owner = {
        str(fact["address"]): str(fact["device"])
        for fact in facts if fact["kind"] == "interface_ipv4_address"
    }
    kind = "mpls_static_lsp_hop" if mpls else "ip_static_route"
    forwarding = [fact for fact in facts if fact["kind"] == kind]
    if len(forwarding) != 3:
        raise ValueError("cycle endpoint does not contain exactly three forwarding facts")
    successor = {
        str(fact["device"]): owner.get(str(fact["next_hop"]))
        for fact in forwarding
    }
    if None in successor.values() or set(successor) != set(successor.values()):
        raise ValueError("next-hop ownership is not proven by visible address facts")
    start = sorted(successor)[0]
    if successor[successor[successor[start]]] != start:
        raise ValueError("forwarding facts do not form a three-device cycle")
    if mpls:
        by_device = {str(fact["device"]): fact for fact in forwarding}
        if any(
            int(fact["out_label"]) != int(by_device[successor[device]]["in_label"])
            for device, fact in by_device.items()
        ):
            raise ValueError("MPLS labels do not close over the visible cycle")


def role_closure_devices(facts: list[dict[str, Any]], vlan: int) -> set[str]:
    devices = {
        str(fact["device"])
        for fact in facts
        if fact.get("kind") == "vrrp_master" and fact.get("device")
    }
    closed: set[str] = set()
    for device in devices:
        matching_master = False
        for master in facts:
            if master.get("kind") != "vrrp_master" or master.get("device") != device:
                continue
            if master.get("vlan") is not None:
                matching_master = int(master["vlan"]) == vlan
            else:
                matching_master = any(
                    fact.get("kind") == "vrrp_interface_context"
                    and fact.get("device") == device
                    and fact.get("action_id") == master.get("action_id")
                    and int(fact.get("vlan", -1)) == vlan
                    for fact in facts
                )
            if matching_master:
                break
        mappings = [
            fact for fact in facts
            if fact.get("kind") == "mst_vlan_instance_mapping"
            and fact.get("device") == device
            and vlan in {int(value) for value in fact.get("vlans") or []}
        ]
        matching_alternate = any(
            fact.get("kind") == "stp_alternate_discarding"
            and fact.get("device") == device
            and int(fact.get("instance", -1)) == int(mapping["instance"])
            for mapping in mappings
            for fact in facts
        )
        if matching_master and mappings and matching_alternate:
            closed.add(device)
    return closed


def validate_endpoints(rows: list[dict[str, Any]]) -> tuple[int, int]:
    endpoints: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if row["metadata"]["target_type"] != "endpoint_bundle":
            continue
        key = (int(row["metadata"]["case_id"]), str(row["metadata"]["path_cluster_id"]))
        if key in endpoints:
            raise ValueError(f"{key}: duplicate endpoint bundle")
        endpoints[key] = row
    if {case_id for case_id, _ in endpoints} != VALID_CASES:
        raise ValueError("not every case has an endpoint bundle")
    inclusive_counts: Counter[int] = Counter()
    for key, row in endpoints.items():
        positive = positive_assistant(row)
        scales = Counter(float(message["loss_scale"]) for message in positive)
        if len(positive) != 5 or scales != Counter({0.05: 2, 0.1: 2, 1.0: 1}):
            raise ValueError(f"{row['id']}: endpoint must jointly supervise 2+2+1 messages")
        nondecision = normalized("\n".join(
            message["content"] for message in positive if "<result>" not in message["content"]
        ))
        gate = row["metadata"].get("endpoint_evidence_gate") or {}
        if not gate.get("passed") or str(gate.get("gate_rule")) in nondecision:
            raise ValueError(f"{row['id']}: endpoint gate failed or leaked internal ID")
        visible = normalized("\n".join(
            message["content"] for message in row["messages"]
            if message["role"] == "tool_response"
        ))
        facts = gate.get("selected_facts") or []
        for fact in facts:
            observation = normalized(str(fact.get("observation") or ""))
            if not observation or observation not in visible or observation not in nondecision:
                raise ValueError(f"{row['id']}: endpoint fact is not visible and supervised")
        if gate["gate_rule"] == "three_device_same_prefix_static_route_next_hop_cycle":
            validate_cycle(facts, mpls=False)
        if gate["gate_rule"] == "three_device_static_lsp_next_hop_and_label_cycle":
            validate_cycle(facts, mpls=True)
        if gate["gate_rule"] == "same_snapshot_source_vlan_vrrp_mst_instance_role_misalignment":
            host_facts = [fact for fact in facts if fact.get("kind") == "source_host_ipv4"]
            if len(host_facts) != 1:
                raise ValueError(f"{row['id']}: role closure needs one source-host fact")
            host = host_facts[0]
            user = normalized(next(
                message["content"]
                for message in row["messages"] if message["role"] == "user"
            ))
            if str(host.get("device") or "").lower() not in user:
                raise ValueError(f"{row['id']}: role closure uses the wrong source host")
            expected_devices = {
                str(value).split(";", 1)[0].lower()
                for value in row["metadata"].get("actual_result_items") or []
            }
            closed = role_closure_devices(facts, int(host["vlan"]))
            if closed != expected_devices:
                raise ValueError(
                    f"{row['id']}: strict role closure mismatch "
                    f"({sorted(closed)} != {sorted(expected_devices)})"
                )
        case_id = int(row["metadata"]["case_id"])
        if case_id in INCLUSIVE_OR_CASES:
            inclusive_counts[case_id] += 1
            if (
                gate.get("gate_rule")
                != "same_snapshot_source_vlan_vrrp_mst_instance_role_misalignment"
                or len(row["metadata"].get("actual_result_items") or []) != 1
                or row["metadata"].get(
                    "inclusive_or_singleton_selected_by_evidence"
                ) is not True
            ):
                raise ValueError(f"{row['id']}: q73-q86 endpoint policy mismatch")
    if set(inclusive_counts) != INCLUSIVE_OR_CASES or set(inclusive_counts.values()) != {1}:
        raise ValueError("q73-q86 must each have one strict singleton endpoint")
    return len(endpoints), len({case_id for case_id, _ in endpoints})


def validate_training_step_contract(
    manifest: dict[str, Any], rows_by_name: dict[str, list[dict[str, Any]]]
) -> None:
    plan = manifest.get("comparison_experiment_plan") or {}
    if (
        plan.get("train_rows_per_stage") != 216
        or plan.get("distributed_strategy") != "ddp"
        or plan.get("world_size") != 2
        or plan.get("cuda_visible_devices") != "0,1"
        or plan.get("gradient_accumulation_steps") != 4
        or plan.get("effective_batch_size") != 8
        or plan.get("optimizer_steps_per_stage") != 27
        or plan.get("expected_global_step_boundaries") != [0, 27, 54, 81, 108, 135]
        or plan.get("checkpoint_suffix_by_stage") != [27, 54, 81, 108, 135]
    ):
        raise ValueError("manifest stage-step contract is incomplete")
    if (
        plan.get("checkpoint_selection_strategy") != "fixed_epoch"
        or plan.get("fixed_validation_epoch") != 3
        or plan.get("fixed_validation_checkpoint_suffix") != 81
        or plan.get("agent_checkpoint_selection") is not False
        or "checkpoint_agent_selection" in plan
    ):
        raise ValueError("checkpoint selection is not fixed to epoch 3")
    for epoch in range(1, 6):
        rows = [
            *rows_by_name[f"train_core_epoch_{epoch:02d}"],
            *rows_by_name[f"train_endpoint_epoch_{epoch:02d}"],
        ]
        if len(rows) != 216 or (len(rows) + (2 * 1 * 4) - 1) // (2 * 1 * 4) != 27:
            raise ValueError(f"epoch {epoch}: row-to-optimizer-step arithmetic mismatch")
    script = (ROOT / "scripts" / "train_qwen36_0807_evidence_gated_5epoch.sh").read_text(
        encoding="utf-8"
    )
    required = (
        "EXPECTED_TRAIN_ROWS_PER_STAGE=216",
        "EXPECTED_OPTIMIZER_STEPS_PER_STAGE=27",
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"',
        'NPROC_PER_NODE="${NPROC_PER_NODE:-2}"',
        "GRADIENT_ACCUMULATION_STEPS=4",
        "FIXED_VALIDATION_EPOCH=3",
        "FIXED_VALIDATION_CHECKPOINT_SUFFIX=81",
        "step_globals != list(range(expected_start, expected_end))",
        'resume_checkpoint="${epoch_dir}/checkpoint-${expected_end_step}"',
    )
    if any(value not in script for value in required):
        raise ValueError("training entry lacks exact step/checkpoint postconditions")
    callback = (ROOT / "scripts" / "qwen36_0807_epoch_lr_callback.py").read_text(
        encoding="utf-8"
    )
    if (
        'PROCESS_RANK = int(os.environ.get("RANK", "0"))' not in callback
        or "if PROCESS_RANK != 0:" not in callback
    ):
        raise ValueError("DDP callback does not reserve audit writes for rank 0")


def main() -> None:
    if infer_families_independently(
        "cat saved_configs/CampusNetwork_01/PE1/display_lldp_neighbor_brief.txt"
    ) != {"lldp"}:
        raise ValueError("LLDP negative regression was classified as MPLS/LDP")
    if "mpls" not in infer_families_independently(
        "cat saved_configs/CampusNetwork_01/PE1/display_mpls_ldp_lsp.txt"
    ):
        raise ValueError("standalone MPLS/LDP positive regression was missed")
    validate_regression_fixtures()
    manifest = load_json(MANIFEST)
    selection = load_json(SELECTION)
    curation = load_json(CURATION)
    validate_reference_sync(curation)
    if manifest["schema_version"] != "qwen36-0807-evidence-gated-case-balanced-sft.v7":
        raise ValueError("unexpected manifest schema")
    if selection["schema_version"] != "0807-evidence-gated-cluster-selection.v6":
        raise ValueError("unexpected selection schema")
    selected = [item for item in curation["trajectories"] if item.get("selected")]
    if len(selected) != 840 or set(Counter(int(item["case_id"]) for item in selected).values()) != {10}:
        raise ValueError("source is not 84 cases x 10 successful trajectories")
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, record in manifest["outputs"].items():
        path = ROOT / record["path"]
        rows = load_jsonl(path)
        if (
            len(rows) != int(record["rows"])
            or path.stat().st_size != int(record["bytes"])
            or digest_file(path) != record["sha256_lf_normalized"]
        ):
            raise ValueError(f"{name}: manifest output record is stale")
        rows_by_name[name] = rows
    semantic = [*rows_by_name["train"], *rows_by_name["validation"]]
    glob_counts: Counter[str] = Counter()
    source_coverage: Counter[str] = Counter()
    supervised_coverage: Counter[str] = Counter()
    cross_snapshot = 0
    lldp_actions = 0
    lldp_regression_rows: set[str] = set()
    for row in semantic:
        user = next(message["content"] for message in row["messages"] if message["role"] == "user")
        snapshot = re.search(r"CampusNetwork(?:-for-perf)?_\d+", user, re.IGNORECASE).group(0).lower()
        actions = row["metadata"]["current_actions"]
        supervised_target = normalized("\n".join(
            message["content"] for message in positive_assistant(row)
        ))
        for action in actions:
            parsed = path_semantics(str(action["command"]))
            if not parsed:
                raise ValueError(f"{row['id']}: current action path is not independently parseable")
            for item in parsed:
                cross_snapshot += item["snapshot"] != snapshot
                for scope in ("snapshot", "device", "filename"):
                    glob_counts[scope] += bool(item[f"{scope}_glob"])
            semantics = action.get("causal_semantics") or {}
            inferred = infer_families_independently(
                f"{action['command']} {' '.join(semantics.get('filenames') or [])}"
            )
            if inferred != set(semantics.get("families") or []):
                raise ValueError(
                    f"{row['id']}: independently inferred family mismatch "
                    f"({sorted(inferred)} != {sorted(semantics.get('families') or [])})"
                )
            if "lldp" in inferred:
                lldp_actions += 1
                if row["id"] in KNOWN_LLDP_REGRESSION_ROWS:
                    lldp_regression_rows.add(row["id"])
                if (
                    "lldp" not in supervised_target
                    or re.search(
                        r"(?<![a-z0-9])(?:mpls|lsp|ldp)(?![a-z0-9])",
                        supervised_target,
                    )
                ):
                    raise ValueError(f"{row['id']}: LLDP command has MPLS/LSP supervision")
        if actions:
            selection_record = row["metadata"].get("action_selection") or {}
            source_status = str(selection_record.get("claim_coverage_status_after"))
            supervised_status = str(
                selection_record.get("supervised_intent_coverage_status")
            )
            if source_status not in {"full", "partial", "zero", "unscoped"}:
                raise ValueError(f"{row['id']}: invalid source intent coverage")
            if supervised_status not in {"full", "partial", "zero", "unscoped"}:
                raise ValueError(f"{row['id']}: invalid supervised intent coverage")
            if supervised_status not in {"full", "unscoped"}:
                raise ValueError(
                    f"{row['id']}: supervised intent is not fully covered by current actions"
                )
            source_coverage[source_status] += 1
            supervised_coverage[supervised_status] += 1
            if {
                str(binding.get("action_id"))
                for binding in selection_record.get("selected_action_bindings") or []
            } != {str(action.get("action_id")) for action in actions}:
                raise ValueError(f"{row['id']}: selected-action binding mismatch")
    if cross_snapshot or any(glob_counts.values()):
        raise ValueError(f"cross-snapshot/glob actions remain: {cross_snapshot}/{dict(glob_counts)}")
    bound_reasoning, procedural_reasoning = validate_reasoning_bindings(semantic)
    eliminations = validate_eliminations(semantic)
    paths, endpoint_cases = validate_endpoints(semantic)
    validate_training_step_contract(manifest, rows_by_name)
    for epoch in range(1, 6):
        core = rows_by_name[f"train_core_epoch_{epoch:02d}"]
        endpoint = rows_by_name[f"train_endpoint_epoch_{epoch:02d}"]
        if set(Counter(int(row["metadata"]["case_id"]) for row in core).values()) != {2}:
            raise ValueError(f"epoch {epoch}: core schedule is not 2/query")
        if set(Counter(int(row["metadata"]["case_id"]) for row in endpoint).values()) != {1}:
            raise ValueError(f"epoch {epoch}: endpoint schedule is not 1 bundle/query")
    print("0807 independent release validation passed")
    print(
        f"semantic_rows={len(semantic)}; endpoint_bundles={paths}/{endpoint_cases} cases; "
        f"observation_bound_reasoning={bound_reasoning}; "
        f"procedural_reasoning={procedural_reasoning}; "
        f"claim_bound_eliminations={eliminations}"
    )
    print(
        "cross_snapshot=0; snapshot_glob=0; device_glob=0; filename_glob=0; "
        "known_bad_elimination_regressions=0; legacy_role_gate_regressions=0; "
        f"lldp_actions={lldp_actions}; lldp_regression_rows={len(lldp_regression_rows)}"
    )
    print("q73-q86 inclusive-OR sync=140 trajectories; dual source targets=0; strict singleton endpoints=14")
    print(
        f"source_intent_coverage={dict(sorted(source_coverage.items()))}; "
        f"supervised_intent_coverage={dict(sorted(supervised_coverage.items()))}; "
        "known_bad_reasoning_regressions=0"
    )


if __name__ == "__main__":
    main()
