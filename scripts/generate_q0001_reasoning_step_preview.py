#!/usr/bin/env python3
"""Generate a manually curated reasoning-step SFT preview for q0001 success_01."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "2026-08-04" / "raw" / "q0001" / "run_01" / "conversation_trajectory.json"
OUTPUT_DIR = ROOT / "data" / "2026-08-04" / "sft_preview"
OUTPUT_PATH = OUTPUT_DIR / "q0001_success_01_reasoning_steps.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

THINKING_LOSS_SCALE = 0.4
TARGET_LOSS_SCALE = 1.0
HISTORY_LOSS_SCALE = 0.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "Run a read-only PowerShell command against the archived network dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact PowerShell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

SYSTEM = (
    "你是一名网络故障分析专家，当前处在一条逐步排障轨迹的特定时间点。"
    "只能使用题目、此前已经输出的阶段分析，以及当前时间点之前已经出现的执行证据；"
    "不得使用后续命令、后续观察或最终答案。先在 <think>...</think> 中给出可复核的推理，"
    "明确说明当前观察、候选假设、能够排除的方向和下一步取证目的。"
    "证据不足时不要提前输出 <result>，而应在阶段结论之后给出下一步实际工具调用；"
    "只有证据完成收敛时才输出最小根因集合。"
)


@dataclass(frozen=True)
class EvidenceSpec:
    item_id: str
    patterns: tuple[str, ...] | None = None
    context: int = 0


@dataclass(frozen=True)
class EvidenceGroup:
    title: str
    commands: tuple[EvidenceSpec, ...]


def evidence_group(title: str, *commands: EvidenceSpec) -> EvidenceGroup:
    return EvidenceGroup(title=title, commands=commands)


EVIDENCE_BY_MESSAGE_INDEX: dict[int, tuple[EvidenceGroup, ...]] = {
    0: (),
    1: (
        evidence_group(
            "定位 CampusNetwork_02 数据集",
            EvidenceSpec("item_3", ("CampusNetwork_02",)),
        ),
        evidence_group(
            "确认源端、边界、PE 和北京核心节点可取证",
            EvidenceSpec(
                "item_4",
                (
                    "BJHQ_CSR1000V_GW_01",
                    "Core_SW_01",
                    "Core_SW_02",
                    "PE1",
                    "PE2",
                    "PE3",
                    "SH_AR",
                    "SH_Core",
                    "SH_SAL_PC01",
                ),
            ),
        ),
    ),
    2: (
        evidence_group(
            "源主机地址正常",
            EvidenceSpec("item_22", ("eth0:", "inet 10.2.10.1/24"), 1),
        ),
        evidence_group("源主机默认路由正常", EvidenceSpec("item_25")),
        evidence_group("SH_AR 的最佳 BGP 路径指向 PE2", EvidenceSpec("item_24")),
        evidence_group("SH_AR 没有 SRv6 policy 输出", EvidenceSpec("item_21")),
        evidence_group(
            "PE2 实际运行 MPLS LDP",
            EvidenceSpec(
                "item_33",
                (
                    "mpls lsr-id",
                    "mpls ldp",
                    "description to_PE1",
                    "description to_PE3",
                    "isis enable 1",
                ),
                1,
            ),
        ),
        evidence_group(
            "PE2 的 SH VPN 路由经 PE1 到达目标网段",
            EvidenceSpec("item_36", ("10.1.10.0/24", "10.2.10.0/24"), 1),
        ),
    ),
    3: (
        evidence_group(
            "PE2 与 PE1 的 ISIS 路径双向直连",
            EvidenceSpec("item_46", ("1.1.1.1/32", "3.3.3.3/32"), 1),
            EvidenceSpec(
                "item_50",
                ("2.2.2.2/32", "3.3.3.3/32", "10.10.0.8/30"),
                1,
            ),
        ),
        evidence_group(
            "PE2 与 PE1 的 MPLS LSP 出接口和 ISIS 路径一致",
            EvidenceSpec("item_48"),
            EvidenceSpec("item_58"),
        ),
        evidence_group(
            "PE1-PE2 直连接口没有带宽挤占",
            EvidenceSpec(
                "item_60",
                (
                    "current state",
                    "Description:",
                    "Internet Address",
                    "Last 300 seconds input rate",
                    "Last 300 seconds output rate",
                    "utility rate",
                ),
            ),
        ),
        evidence_group(
            "Core_SW_01 承担目标 VIP 的 Master 角色",
            EvidenceSpec("item_82"),
        ),
        evidence_group(
            "两台核心到源网段的 OSPF 代价明显不对称",
            EvidenceSpec("item_83"),
            EvidenceSpec("item_87"),
        ),
    ),
    4: (
        evidence_group(
            "Core_SW_01 的全局 STP 状态为 Disabled",
            EvidenceSpec("item_129"),
        ),
        evidence_group(
            "Core_SW_02 存在正常 MSTP 端口角色",
            EvidenceSpec("item_131"),
        ),
    ),
}


RECONSTRUCTED_THINKING = (
    (
        "题目只确认 PE2 与 PE3 之间是高时延链路，并未证明业务实际经过该链路。"
        "因此不能直接把高延迟归因为 PE2-PE3；应先定位 CampusNetwork_02 的可用节点和回显，"
        "再从源主机默认网关开始还原去往 10.1.10.254 的三层与隧道路径，"
        "同时保留普通接口或路由异常作为竞争假设。"
    ),
    (
        "E01 已确认存在 CampusNetwork_02 数据集，E02 显示源主机、上海边界、PE1/PE2/PE3、"
        "北京核心和网关等关键节点均有独立数据目录。当前只有拓扑可观测性，尚不能判断根因；"
        "下一步应围绕源端下一跳、运营商承载路径和北京侧回程三个方向读取最小必要文件，"
        "避免直接扫描全部配置造成噪声。"
    ),
    (
        "E01、E02 表明 SH_SAL_PC01 地址为 10.2.10.1/24，默认路由指向 10.2.10.254，"
        "源端地址与默认网关没有明显异常。E03 显示 SH_AR 对 10.1.10.0/24 的最佳 BGP 路径"
        "下一跳为 PE2；E06 又显示 PE2 的 SH VPN 路由经 Ethernet1/0/1 到 PE1。"
        "E04 的 SRv6 policy 输出为空，而 E05 显示 PE2 实际启用了 MPLS LDP，"
        "所以 SRV6-Policy 隧道规划错误与现网承载方式不符。下一步需要用 ISIS 路由、"
        "MPLS LSP 和接口利用率确认 PE2 到 PE1 是否直连，以及是否存在标签环路或带宽挤占；"
        "同时读取北京两台核心的 VRRP 角色和到源网段路由，以便在排除运营商路径后立即转向"
        "园区侧二三层路径一致性。"
    ),
    (
        "E01 表明 PE2 与 PE1 双向均选择直连路径；E02 表明两端 MPLS LSP 的出接口"
        "与 ISIS 路径一致；E03 显示该直连接口为 UP，近 300 秒双向利用率为 0%。"
        "因此长链路绕行、MPLS 标签环路和端口带宽挤占均缺少证据。"
        "E04 显示 10.1.10.254 当前由 Core_SW_01 承担 Master；E05 显示两台核心到源网段的 "
        "OSPF 代价明显不对称。下一步应优先核验两台核心的 STP 全局状态和实例转发状态；"
        "如果 STP 正常，再继续核对 OSPF 度量。"
    ),
    (
        "此前证据已经排除了源端默认路由、SRv6、PE2-PE3 绕行、MPLS 标签环路和带宽挤占。"
        "最新 E01 直接显示 Core_SW_01 的 Protocol Status 为 Disabled，"
        "而 E02 显示 Core_SW_02 存在正常的 MSTP 端口角色与 forwarding 状态。"
        "这不是对 OSPF 代价的间接猜测，而是 Core_SW_01 全局 STP 未使能的直接状态证据；"
        "它也是候选故障类型中最具体、能够破坏二层路径规划的最小根因，因此只保留该项。"
    ),
)

STEP_TARGET_TYPES = ("planning", "reasoning", "reasoning", "reasoning", "decision")

OPTIMIZED_STAGE_OUTPUTS = {
    3: (
        "当前已经排除 PE2-PE3 长链路绕行、MPLS 标签环路和端口带宽挤占。"
        "10.1.10.254 当前由 Core_SW_01 承担 Master，但两台核心到源网段的 OSPF 代价"
        "明显不对称。下一步应优先核验两台核心的 STP 全局状态和实例转发状态；"
        "如果 STP 正常，再继续核对 OSPF 度量。"
    )
}

# Only facts that remain useful to later diagnosis are carried forward. Keys are
# (zero-based message index, zero-based evidence-group index).
PERSISTENT_LEDGER_FACTS: dict[tuple[int, int], dict[str, str]] = {
    (2, 0): {
        "fact": "SH_SAL_PC01 的业务接口地址为 10.2.10.1/24，接口处于 UP。",
        "key_values": "eth0: <BROADCAST,MULTICAST,UP,LOWER_UP>\ninet 10.2.10.1/24 scope global eth0",
    },
    (2, 1): {
        "fact": "SH_SAL_PC01 的默认路由指向本地网关 10.2.10.254。",
        "key_values": "default via 10.2.10.254 dev eth0",
    },
    (2, 2): {
        "fact": "SH_AR 到 10.1.10.0/24 的最佳 BGP 路径指向 PE2。",
        "key_values": "10.1.10.0/24\nFrom: 10.10.100.5 (2.2.2.2)\nDirect Out-interface: Ethernet1/0/1",
    },
    (2, 3): {
        "fact": "SH_AR 的 SRv6 policy 回显为空。",
        "key_values": "(empty output)",
    },
    (2, 4): {
        "fact": "PE2 实际启用了 MPLS LDP，PE1 与 PE3 方向接口均运行 ISIS/MPLS。",
        "key_values": "mpls lsr-id 2.2.2.2\nmpls ldp\nEthernet1/0/1: to_PE1, isis enable 1, mpls ldp\nEthernet1/0/2: to_PE3, isis enable 1, mpls ldp",
    },
    (2, 5): {
        "fact": "PE2 的 SH VPN 路由到 10.1.10.0/24 时经 Ethernet1/0/1 指向 PE1。",
        "key_values": "10.1.10.0/24 IBGP 255 3 RD 1.1.1.1 Ethernet1/0/1",
    },
    (3, 0): {
        "fact": "PE2 到 PE1、PE1 到 PE2 的 ISIS 路径均选择直连接口。",
        "key_values": "PE2: 1.1.1.1/32 -> Eth1/0/1 -> 10.10.0.1\nPE1: 2.2.2.2/32 -> Eth1/0/1 -> 10.10.0.2",
    },
    (3, 1): {
        "fact": "PE2 与 PE1 的 MPLS LSP 出接口与 ISIS 直连路径一致。",
        "key_values": "PE2: 1.1.1.1/32 NULL/3 -/Eth1/0/1\nPE1: 2.2.2.2/32 NULL/3 -/Eth1/0/1",
    },
    (3, 2): {
        "fact": "PE1-PE2 直连接口为 UP，近 300 秒双向利用率为 0%。",
        "key_values": "Ethernet1/0/1 current state : UP\nLast 300 seconds input utility rate: 0.00%\nLast 300 seconds output utility rate: 0.00%",
    },
    (3, 3): {
        "fact": "目标 VIP 10.1.10.254 当前由 Core_SW_01 的 Vlanif10 承担 Master。",
        "key_values": "1 Master Vlanif10 N 10.1.10.254",
    },
    (3, 4): {
        "fact": "两台核心到源网段 10.2.10.0/24 的 OSPF 代价明显不对称。",
        "key_values": "Core_SW_01: cost 102 -> 10.1.200.1 Vlanif201\nCore_SW_02: cost 3 -> 10.1.200.9 Vlanif203",
    },
    (4, 0): {
        "fact": "Core_SW_01 的全局 STP 状态为 Disabled。",
        "key_values": "Protocol Status :Disabled",
    },
    (4, 1): {
        "fact": "Core_SW_02 存在正常的 MSTP 端口角色与 forwarding 状态。",
        "key_values": "MSTID Port Role STP State\n0 Eth-Trunk1 DESI forwarding",
    },
}


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def digest_text(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected object")
    return value


def load_completed_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    messages: list[dict[str, Any]] = []
    commands: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for event_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            item = event.get("item") if isinstance(event, dict) else None
            if event.get("type") != "item.completed" or not isinstance(item, dict):
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
                item_id = str(item.get("id"))
                commands[item_id] = {
                    "event_line": event_line,
                    "item_id": item_id,
                    "command": normalize_text(str(item.get("command", ""))),
                    "output": normalize_text(str(item.get("aggregated_output", ""))),
                    "exit_code": item.get("exit_code"),
                    "status": item.get("status"),
                }
    return messages, commands


def excerpt_output(output: str, spec: EvidenceSpec) -> tuple[str, bool]:
    if not output:
        return "(empty output)", False
    if spec.patterns is None:
        return output, False

    lines = output.splitlines()
    lowered_patterns = tuple(pattern.lower() for pattern in spec.patterns)
    selected: set[int] = set()
    for index, line in enumerate(lines):
        if any(pattern in line.lower() for pattern in lowered_patterns):
            selected.update(
                range(max(0, index - spec.context), min(len(lines), index + spec.context + 1))
            )
    if not selected:
        raise ValueError(f"{spec.item_id}: no output line matched {spec.patterns}")

    chunks: list[str] = []
    previous: int | None = None
    for index in sorted(selected):
        if previous is not None and index != previous + 1:
            chunks.append("... [unselected lines omitted] ...")
        chunks.append(lines[index])
        previous = index
    return "\n".join(chunks), len(selected) != len(lines)


def render_evidence(
    groups: tuple[EvidenceGroup, ...],
    commands: dict[str, dict[str, Any]],
    *,
    lower_bound: int,
    upper_bound: int,
) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    metadata: list[dict[str, Any]] = []
    for evidence_index, group in enumerate(groups, start=1):
        evidence_id = f"E{evidence_index:02d}"
        command_blocks: list[str] = []
        command_metadata: list[dict[str, Any]] = []
        for command_index, spec in enumerate(group.commands, start=1):
            command = commands.get(spec.item_id)
            if command is None:
                raise ValueError(f"missing command event: {spec.item_id}")
            event_line = int(command["event_line"])
            if not lower_bound < event_line < upper_bound:
                raise ValueError(
                    f"{spec.item_id}: event line {event_line} is outside ({lower_bound}, {upper_bound})"
                )
            excerpt, excerpted = excerpt_output(str(command["output"]), spec)
            command_blocks.append(
                "\n".join(
                    [
                        f"#### 命令 {command_index}（源事件行 {event_line}，{spec.item_id}）",
                        "",
                        "```powershell",
                        str(command["command"]),
                        "```",
                        "",
                        f"退出码：{command['exit_code']}；状态：{command['status']}",
                        "",
                        "执行结果原文摘录：",
                        "```text",
                        excerpt,
                        "```",
                    ]
                )
            )
            command_metadata.append(
                {
                    "event_line": event_line,
                    "item_id": spec.item_id,
                    "exit_code": command["exit_code"],
                    "status": command["status"],
                    "command_sha256_lf_normalized": digest_text(
                        str(command["command"])
                    ),
                    "full_output_sha256_lf_normalized": digest_text(
                        str(command["output"])
                    ),
                    "excerpt_sha256_lf_normalized": digest_text(excerpt),
                    "output_is_excerpt": excerpted,
                    "output_excerpt": excerpt,
                }
            )
        blocks.append(
            f"### {evidence_id}：{group.title}\n\n" + "\n\n".join(command_blocks)
        )
        metadata.append(
            {
                "evidence_id": evidence_id,
                "title": group.title,
                "commands": command_metadata,
                "command_count": len(command_metadata),
                "selection": "manual_causal_evidence_review",
            }
        )
    return "\n\n".join(blocks) if blocks else "（本节点没有新增执行证据。）", metadata


def build_persistent_ledger_entries(
    message_index: int, evidence_metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for group_index, group in enumerate(evidence_metadata):
        fact = PERSISTENT_LEDGER_FACTS.get((message_index, group_index))
        if fact is None:
            continue
        entries.append(
            {
                "ledger_id": f"S{message_index + 1:02d}-E{group_index + 1:02d}",
                "source_step_index": message_index + 1,
                "source_evidence_id": group["evidence_id"],
                "title": group["title"],
                "fact": fact["fact"],
                "key_values": fact["key_values"],
                "source_commands": [
                    {
                        "event_line": command["event_line"],
                        "item_id": command["item_id"],
                        "command_sha256_lf_normalized": command[
                            "command_sha256_lf_normalized"
                        ],
                        "full_output_sha256_lf_normalized": command[
                            "full_output_sha256_lf_normalized"
                        ],
                    }
                    for command in group["commands"]
                ],
                "representation": "normalized_fact_and_key_values",
            }
        )
    return entries


def render_persistent_ledger(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "（暂无需要跨节点继承的诊断证据。）"
    blocks: list[str] = []
    for entry in entries:
        sources = "、".join(
            f"事件行 {command['event_line']} / {command['item_id']}"
            for command in entry["source_commands"]
        )
        blocks.append(
            "\n".join(
                [
                    f"### {entry['ledger_id']}：{entry['title']}",
                    "",
                    f"关键事实：{entry['fact']}",
                    "",
                    "关键值：",
                    "```text",
                    entry["key_values"],
                    "```",
                    "",
                    f"来源：{sources}。完整命令和输出可由对应哈希追溯。",
                ]
            )
        )
    return "\n\n".join(blocks)


def flatten_evidence_commands(groups: tuple[EvidenceGroup, ...]) -> list[EvidenceSpec]:
    return [spec for group in groups for spec in group.commands]


def render_next_actions(
    message_index: int,
    messages: list[dict[str, Any]],
    commands: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    if message_index == len(messages) - 1:
        return "", []
    specs = flatten_evidence_commands(EVIDENCE_BY_MESSAGE_INDEX[message_index + 1])
    lower_bound = int(messages[message_index]["event_line"])
    upper_bound = int(messages[message_index + 1]["event_line"])
    rendered: list[str] = []
    metadata: list[dict[str, Any]] = []
    for action_index, spec in enumerate(specs, start=1):
        command = commands.get(spec.item_id)
        if command is None:
            raise ValueError(f"missing next-action command: {spec.item_id}")
        event_line = int(command["event_line"])
        if not lower_bound < event_line < upper_bound:
            raise ValueError(
                f"{spec.item_id}: next action line {event_line} is outside "
                f"({lower_bound}, {upper_bound})"
            )
        action_id = f"A{action_index:02d}"
        rendered.append(
            "\n".join(
                [
                    f'<tool_call id="{action_id}">',
                    json.dumps(
                        {
                            "tool": "powershell",
                            "command": str(command["command"]),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "</tool_call>",
                ]
            )
        )
        metadata.append(
            {
                "action_id": action_id,
                "item_id": spec.item_id,
                "source_event_line": event_line,
                "tool": "powershell",
                "command": str(command["command"]),
                "command_sha256_lf_normalized": digest_text(str(command["command"])),
                "result_visible_in_current_sample": False,
                "result_supplied_as_next_step_evidence": True,
                "selection": "actual_causally_useful_command_after_detour_pruning",
            }
        )
    return "\n\n".join(rendered), metadata


def build_initial_user_prompt(raw: dict[str, Any]) -> str:
    source_record = raw["source_record"]
    return "\n\n".join(
        [
            str(source_record["question"]),
            "请在排障过程中只使用当前消息序列中已经返回的工具结果；证据不足时继续调用工具，不要提前输出最终答案。",
            "答案格式要求：",
            str(source_record["output_format"]),
        ]
    )


def build_assistant_messages(
    stage_outputs: list[str], message_index: int, *, is_target: bool
) -> list[dict[str, Any]]:
    thinking_scale = THINKING_LOSS_SCALE if is_target else HISTORY_LOSS_SCALE
    conclusion_scale = TARGET_LOSS_SCALE if is_target else HISTORY_LOSS_SCALE
    return [
        {
            "role": "assistant",
            "content": f"<think>\n{RECONSTRUCTED_THINKING[message_index]}\n</think>\n\n",
            "loss_scale": thinking_scale,
        },
        {
            "role": "assistant",
            "content": stage_outputs[message_index],
            "loss_scale": conclusion_scale,
        },
    ]


def build_tool_call_messages(
    actions: list[dict[str, Any]], *, loss_scale: float
) -> list[dict[str, Any]]:
    return [
        {
            "role": "tool_call",
            "content": json.dumps(
                {
                    "name": action["tool"],
                    "arguments": {"command": action["command"]},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "loss_scale": loss_scale,
        }
        for action in actions
    ]


def build_tool_response_messages(
    evidence_metadata: list[dict[str, Any]],
    *,
    evidence_stage_index: int,
    compact: bool,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for group_index, group in enumerate(evidence_metadata):
        ledger_fact = PERSISTENT_LEDGER_FACTS.get(
            (evidence_stage_index, group_index)
        )
        for command in group["commands"]:
            payload: dict[str, Any] = {
                "status": command["status"],
                "exit_code": command["exit_code"],
            }
            if compact:
                payload["compacted_from_prior_tool_response"] = True
                if ledger_fact is None:
                    payload["summary"] = group["title"]
                    payload["detail"] = (
                        "one-time discovery evidence; exact response was available at "
                        "its first reasoning checkpoint"
                    )
                else:
                    payload["fact"] = ledger_fact["fact"]
                    payload["key_values"] = ledger_fact["key_values"]
            else:
                payload["output"] = command["output_excerpt"]
            responses.append(
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    return responses


def build_native_messages(
    raw: dict[str, Any],
    stage_outputs: list[str],
    evidence_by_stage: list[list[dict[str, Any]]],
    actions_by_stage: list[list[dict[str, Any]]],
    target_index: int,
) -> list[dict[str, Any]]:
    native_messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_initial_user_prompt(raw)},
    ]
    for stage_index in range(target_index + 1):
        is_target = stage_index == target_index
        native_messages.extend(
            build_assistant_messages(
                stage_outputs, stage_index, is_target=is_target
            )
        )
        native_messages.extend(
            build_tool_call_messages(
                actions_by_stage[stage_index],
                loss_scale=(TARGET_LOSS_SCALE if is_target else HISTORY_LOSS_SCALE),
            )
        )
        if not is_target:
            native_messages.extend(
                build_tool_response_messages(
                    evidence_by_stage[stage_index + 1],
                    evidence_stage_index=stage_index + 1,
                    compact=stage_index + 1 < target_index,
                )
            )
    return native_messages


def validate_loss_policy(
    rows: list[dict[str, Any]],
    actions_by_stage: list[list[dict[str, Any]]],
    evidence_by_stage: list[list[dict[str, Any]]],
) -> None:
    for stage_index, actions in enumerate(actions_by_stage[:-1]):
        action_items = [action["item_id"] for action in actions]
        next_evidence_items = [
            command["item_id"]
            for group in evidence_by_stage[stage_index + 1]
            for command in group["commands"]
        ]
        if action_items != next_evidence_items:
            raise ValueError(
                f"step {stage_index + 1}: tool calls do not match next-step evidence"
            )

    for target_index, row in enumerate(rows):
        messages = row["messages"]
        assistant_messages = [
            message for message in messages if message["role"] == "assistant"
        ]
        expected_assistant_scales = [HISTORY_LOSS_SCALE] * (2 * target_index) + [
            THINKING_LOSS_SCALE,
            TARGET_LOSS_SCALE,
        ]
        actual_assistant_scales = [
            message.get("loss_scale") for message in assistant_messages
        ]
        if actual_assistant_scales != expected_assistant_scales:
            raise ValueError(
                f"{row['id']}: assistant loss scales {actual_assistant_scales} != "
                f"{expected_assistant_scales}"
            )
        tool_calls = [message for message in messages if message["role"] == "tool_call"]
        expected_tool_scales = [
            scale
            for stage_index in range(target_index + 1)
            for scale in [
                TARGET_LOSS_SCALE
                if stage_index == target_index
                else HISTORY_LOSS_SCALE
            ]
            * len(actions_by_stage[stage_index])
        ]
        actual_tool_scales = [message.get("loss_scale") for message in tool_calls]
        if actual_tool_scales != expected_tool_scales:
            raise ValueError(
                f"{row['id']}: tool-call loss scales {actual_tool_scales} != "
                f"{expected_tool_scales}"
            )
        expected_trainable_calls = len(actions_by_stage[target_index])
        actual_trainable_calls = sum(
            message.get("loss_scale") == TARGET_LOSS_SCALE for message in tool_calls
        )
        if actual_trainable_calls != expected_trainable_calls:
            raise ValueError(
                f"{row['id']}: trainable tool-call count {actual_trainable_calls} != "
                f"{expected_trainable_calls}"
            )
        if any(
            "loss" in message or "loss_scale" in message
            for message in messages
            if message["role"] == "tool_response"
        ):
            raise ValueError(f"{row['id']}: tool responses must remain context-only")
        if any(
            "loss" in message or "loss_scale" in message
            for message in messages
            if message["role"] in {"system", "user", "tool_response"}
        ):
            raise ValueError(f"{row['id']}: context-only role was marked trainable")
        if any("loss" in message for message in messages):
            raise ValueError(f"{row['id']}: binary loss field must not be mixed with loss_scale")
        expected_response_count = sum(
            sum(group["command_count"] for group in evidence_by_stage[stage_index])
            for stage_index in range(1, target_index + 1)
        )
        actual_response_count = sum(
            message["role"] == "tool_response" for message in messages
        )
        if actual_response_count != expected_response_count:
            raise ValueError(
                f"{row['id']}: tool-response count {actual_response_count} != "
                f"{expected_response_count}"
            )
        if messages[-1]["role"] == "tool_response":
            raise ValueError(f"{row['id']}: current tool results leaked into target")


def build_user_prompt(
    raw: dict[str, Any],
    messages: list[dict[str, Any]],
    stage_outputs: list[str],
    message_index: int,
    ledger_text: str,
    evidence_text: str,
) -> str:
    source_record = raw["source_record"]
    prior_messages = messages[:message_index]
    if prior_messages:
        history = "\n\n".join(
            f"### 历史节点 {index + 1}（源事件行 {message['event_line']}）\n\n{stage_outputs[index]}"
            for index, message in enumerate(prior_messages)
        )
    else:
        history = "（无；这是轨迹的第一个推理节点。）"

    is_final = message_index == len(messages) - 1
    task = (
        "结合历史分析和最新证据完成最终归因，输出最小根因集合。"
        if is_final
        else "根据当前时点已经出现的信息，更新候选假设、说明可排除项，并输出下一步最有区分度的实际工具调用；不要提前输出最终答案。"
    )
    return "\n\n".join(
        [
            "根据题目和截至当前时间点的轨迹信息继续分析。",
            f"## 题目\n\n{source_record['question']}",
            f"## 答案格式约束\n\n{source_record['output_format']}",
            f"## 当前推理节点\n\n第 {message_index + 1}/{len(messages)} 个节点。{task}",
            f"## 此前已输出的阶段分析\n\n{history}",
            f"## 累计关键证据账本\n\n{ledger_text}",
            f"## 本节点新增的执行证据\n\n{evidence_text}",
            "## 时间约束\n\n不得使用源事件中晚于当前节点的命令、结果、Agent 消息或最终答案。",
        ]
    )


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = load_json(RAW_PATH)
    events_path = ROOT / raw["source"]["events_file"]
    messages, commands = load_completed_events(events_path)
    archived_messages = raw.get("agent_messages")
    if not isinstance(archived_messages, list) or [
        (item.get("event_line"), item.get("item_id"), normalize_text(str(item.get("text", ""))))
        for item in archived_messages
    ] != [
        (item["event_line"], item["item_id"], item["text"])
        for item in messages
    ]:
        raise ValueError("raw agent messages do not match source events")
    if len(messages) != 5:
        raise ValueError(f"expected 5 agent messages, found {len(messages)}")
    if messages[-1]["text"] != normalize_text(str(raw["final_answer"])):
        raise ValueError("final agent message differs from archived final answer")

    trajectory_id = str(raw["id"])
    stage_outputs = [
        OPTIMIZED_STAGE_OUTPUTS.get(index, message["text"])
        for index, message in enumerate(messages)
    ]
    stage_outputs[-1] = normalize_text(str(raw["final_answer"]))

    evidence_by_stage: list[list[dict[str, Any]]] = []
    inherited_ledger_by_stage: list[list[dict[str, Any]]] = []
    new_ledger_by_stage: list[list[dict[str, Any]]] = []
    actions_by_stage: list[list[dict[str, Any]]] = []
    cumulative_ledger: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        previous_line = int(messages[message_index - 1]["event_line"]) if message_index else 0
        _, evidence_metadata = render_evidence(
            EVIDENCE_BY_MESSAGE_INDEX[message_index],
            commands,
            lower_bound=previous_line,
            upper_bound=int(message["event_line"]),
        )
        inherited_ledger = [dict(entry) for entry in cumulative_ledger]
        new_ledger_entries = build_persistent_ledger_entries(
            message_index, evidence_metadata
        )
        _, next_actions_metadata = render_next_actions(
            message_index, messages, commands
        )
        evidence_by_stage.append(evidence_metadata)
        inherited_ledger_by_stage.append(inherited_ledger)
        new_ledger_by_stage.append(new_ledger_entries)
        actions_by_stage.append(next_actions_metadata)
        cumulative_ledger.extend(new_ledger_entries)

    rows: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        evidence_metadata = evidence_by_stage[message_index]
        inherited_ledger = inherited_ledger_by_stage[message_index]
        new_ledger_entries = new_ledger_by_stage[message_index]
        next_actions_metadata = actions_by_stage[message_index]
        is_final = message_index == len(messages) - 1
        row = {
            "id": f"{trajectory_id}_step_{message_index + 1:02d}",
            "tools": json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":")),
            "messages": build_native_messages(
                raw,
                stage_outputs,
                evidence_by_stage,
                actions_by_stage,
                message_index,
            ),
            "metadata": {
                "dataset_type": "reasoning_trajectory_step",
                "target_type": STEP_TARGET_TYPES[message_index],
                "review_status": "draft_preview",
                "split": "train",
                "trajectory_id": trajectory_id,
                "case_id": int(raw["case_id"]),
                "row_index": int(raw["row_index"]),
                "repeat_index": int(raw["success_slot"]),
                "attempt_index": int(raw["attempt_index"]),
                "step_index": message_index + 1,
                "step_count": len(messages),
                "source_file": RAW_PATH.relative_to(ROOT).as_posix(),
                "source_event_file": raw["source"]["events_file"],
                "source_event_sha256_lf_normalized": raw["source"][
                    "events_sha256_lf_normalized"
                ],
                "source_thread_id": raw["source"]["thread_id"],
                "source_message_index": message_index,
                "source_message_event_line": int(message["event_line"]),
                "source_message_item_id": message["item_id"],
                "prior_message_indices": list(range(message_index)),
                "original_visible_output_source": (
                    "verified_final_answer"
                    if is_final
                    else (
                        "optimized_reconstruction_from_available_evidence"
                        if message_index in OPTIMIZED_STAGE_OUTPUTS
                        else "original_agent_message"
                    )
                ),
                "source_message_retained_in_sft": (
                    is_final or message_index not in OPTIMIZED_STAGE_OUTPUTS
                ),
                "source_message_omission_reason": (
                    "nonoptimal_detour_removed"
                    if message_index in OPTIMIZED_STAGE_OUTPUTS
                    else None
                ),
                "thinking_source": "reconstructed_from_available_evidence",
                "thinking_is_original_hidden_chain_of_thought": False,
                "evidence_selection": "manual_causal_evidence_review",
                "evidence": evidence_metadata,
                "evidence_count": len(evidence_metadata),
                "evidence_command_count": sum(
                    group["command_count"] for group in evidence_metadata
                ),
                "inherited_evidence_ledger": inherited_ledger,
                "inherited_evidence_count": len(inherited_ledger),
                "new_persistent_evidence": new_ledger_entries,
                "new_persistent_evidence_count": len(new_ledger_entries),
                "next_actions": next_actions_metadata,
                "next_action_count": len(next_actions_metadata),
                "next_action_target_step_index": (
                    message_index + 2 if next_actions_metadata else None
                ),
                "loss_policy": {
                    "framework": "ms-swift==4.4.2",
                    "cli_loss_scale": "default",
                    "cli_is_binary_loss_scale": False,
                    "current_thinking_loss_scale": THINKING_LOSS_SCALE,
                    "current_conclusion_loss_scale": TARGET_LOSS_SCALE,
                    "current_tool_call_loss_scale": TARGET_LOSS_SCALE,
                    "historical_assistant_and_tool_call_loss_scale": HISTORY_LOSS_SCALE,
                    "masked": (
                        "system, user, historical assistant/tool calls, tool responses, padding"
                    ),
                    "target_assistant_message_count": 2,
                    "target_tool_call_count": len(next_actions_metadata),
                },
                "future_event_leakage_checked": True,
                "final_answer_visible": is_final,
                "actual_result_items": raw["actual_result_items"] if is_final else None,
                "reference_answer_match": bool(raw["answer_matches_reference"])
                if is_final
                else None,
            },
        }
        rows.append(row)

    validate_loss_policy(rows, actions_by_stage, evidence_by_stage)

    manifest = {
        "schema_version": "qwen36-reasoning-trajectory-step-preview.v3",
        "status": "preview_requires_user_review",
        "scope": "q0001 success_01 only",
        "source_raw": RAW_PATH.relative_to(ROOT).as_posix(),
        "source_events": raw["source"]["events_file"],
        "source_events_sha256_lf_normalized": raw["source"][
            "events_sha256_lf_normalized"
        ],
        "trajectory_count": 1,
        "sft_sample_count": len(rows),
        "training_profile": {
            "scope": "data/2026-08-04 only",
            "max_length": 16384,
            "truncation_strategy": "delete",
            "loss_scale": "default",
            "is_binary_loss_scale": False,
            "add_non_thinking_prefix": False,
            "preserve_thinking": True,
            "tokenizer_preflight_required": True,
        },
        "split_rule": "one_sft_sample_per_valuable_reasoning_checkpoint",
        "message_format": "native ms-swift multi-turn agent messages",
        "temporal_rule": "each sample ends at its current target response/tool calls and contains no future results",
        "history_policy": {
            "stage_analysis": "all prior optimized assistant turns are native prefix messages",
            "evidence_ledger": (
                "persistent normalized facts remain in metadata for audit; the model input "
                "inherits prior assistant turns and selected tool responses directly"
            ),
            "current_evidence": "selected prior tool calls and exact output excerpts as native tool responses",
            "older_tool_responses": (
                "compacted to persistent facts/key values after their first reasoning checkpoint; "
                "exact excerpts remain in source provenance"
            ),
            "one_time_discovery_evidence": "not carried forward",
        },
        "action_policy": {
            "nonfinal_output": "reasoning, optimized stage conclusion, then actual tool calls",
            "command_source": "source events after pruning nonoptimal detours",
            "result_visibility": "not in current output; supplied as next-step input evidence",
            "chain_invariant": "step N output action IDs equal step N+1 evidence command IDs",
        },
        "loss_policy": {
            "framework": "ms-swift==4.4.2",
            "cli": "--loss_scale default",
            "required_cli": "--is_binary_loss_scale false",
            "message_override": "assistant/tool_call loss_scale fields override the basic default strategy",
            "current_thinking": THINKING_LOSS_SCALE,
            "current_conclusion_or_result": TARGET_LOSS_SCALE,
            "current_tool_calls": TARGET_LOSS_SCALE,
            "historical_assistant_and_tool_calls": HISTORY_LOSS_SCALE,
            "masked": "system, user, all historical assistant/tool_call messages, all tool responses, padding",
            "reason": (
                "soften wording-level supervision for reconstructed reasoning while keeping "
                "diagnostic conclusions, exact actions, and final answers strict"
            ),
            "future_sequence_objective": (
                "semantic/path/evidence consistency is reserved for a later DPO/GRPO reward stage; "
                "it does not replace token cross-entropy in this SFT preview"
            ),
        },
        "sequence_alignment_policy": {
            "stage": "after weighted SFT",
            "method": "DPO or GRPO; not part of this preview loss",
            "hard_gates": [
                "final root-cause set is exactly correct",
                "tool-call syntax and arguments are executable",
                "no future evidence is used",
            ],
            "reward_dimensions": [
                "semantic similarity to an accepted attribution path",
                "evidence-to-claim consistency",
                "earlier entry into the correct causal path",
                "utility of selected commands",
                "penalty for avoidable detours and redundant evidence",
            ],
            "status": "specified_not_executed",
        },
        "thinking_policy": {
            "original_visible_agent_message_policy": (
                "retain useful messages; replace nonoptimal detours with "
                "evidence-grounded optimized targets"
            ),
            "optimized_source_message_indices": sorted(OPTIMIZED_STAGE_OUTPUTS),
            "reconstructed_thinking_added": True,
            "reconstructed_thinking_claimed_as_original": False,
        },
        "command_policy": {
            "selection": "manual causal review",
            "command_text": "exact",
            "output": "exact relevant excerpts with full-output hashes",
            "irrelevant_or_duplicate_commands": "omitted",
        },
        "samples_by_target_type": {
            target_type: STEP_TARGET_TYPES.count(target_type)
            for target_type in sorted(set(STEP_TARGET_TYPES))
        },
    }
    return rows, manifest


def main() -> None:
    rows, manifest = build_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    OUTPUT_PATH.write_text(output_text, encoding="utf-8", newline="\n")
    manifest["output"] = {
        "path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "samples": len(rows),
        "bytes": len(output_text.encode("utf-8")),
        "sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Generated {len(rows)} reasoning-step samples")
    print(f"- output: {OUTPUT_PATH.relative_to(ROOT).as_posix()}")
    print(
        "- evidence-group counts: "
        + ", ".join(str(row["metadata"]["evidence_count"]) for row in rows)
    )
    print(
        "- evidence-command counts: "
        + ", ".join(
            str(row["metadata"]["evidence_command_count"]) for row in rows
        )
    )


if __name__ == "__main__":
    main()
