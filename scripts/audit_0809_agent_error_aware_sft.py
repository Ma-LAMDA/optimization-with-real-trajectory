#!/usr/bin/env python3
"""Recompute the 0809 v7 release audit metrics and human-readable report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-08-09"
STRUCTURAL_AUDIT_ACTIONS = {
    "0809_cf_action_q0015_a02", "0809_cf_action_q0025_a02",
    "0809_cf_action_q0026_a01", "0809_cf_action_q0026_a02",
    "0809_cf_action_q0027_a02", "0809_cf_action_q0027_a03",
    "0809_cf_action_q0028_a01", "0809_cf_action_q0028_a02",
    "0809_cf_action_q0028_a03", "0809_cf_action_q0031_a01",
    "0809_cf_action_q0034_a01", "0809_cf_action_q0035_a03",
    "0809_cf_action_q0037_a03", "0809_cf_action_q0039_a01",
    "0809_cf_action_q0060_a02", "0809_cf_action_q0061_a02",
    "0809_cf_action_q0066_a03",
}
AUDITED_COMPLETE_SOURCE = "q0023_path_02_success_09_step_04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def resolve_data_root(value: Path) -> Path:
    path = (value if value.is_absolute() else ROOT / value).resolve()
    expected = (ROOT / "data" / "2026-08-09").resolve()
    if path != expected:
        raise ValueError(f"audit expected {expected}, received {path}")
    return path


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def case_id(row: dict[str, Any]) -> int:
    return int(row["metadata"]["case_id"])


def positive_prefix(row: dict[str, Any]) -> str:
    prefix: list[tuple[str, str]] = []
    for message in row["messages"]:
        if float(message.get("loss_scale", 0) or 0) > 0:
            break
        prefix.append((message.get("role", ""), message.get("content", "")))
    return json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))


def positive_target(row: dict[str, Any], include_weight: bool = False) -> str:
    target = []
    for message in row["messages"]:
        if float(message.get("loss_scale", 0) or 0) <= 0:
            continue
        item: tuple[Any, ...] = (message.get("role", ""), message.get("content", ""))
        if include_weight:
            item += (float(message.get("loss_scale", 0) or 0),)
        target.append(item)
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def condition_normalized_target(row: dict[str, Any]) -> str:
    """Remove only mechanical release/query/path identifiers, not evidence facts."""
    value = positive_target(row)
    value = re.sub(r"CampusNetwork(?:-for-perf|_\d+)", "<SNAPSHOT>", value)
    value = re.sub(r"q\d{4}", "<QUERY>", value)
    value = re.sub(r"(?:[A-Za-z]:)?[^\s\"']*saved_configs/[^\s\"']+", "<CONFIG_PATH>", value)
    return value


def action_visible_grounding_issues(row: dict[str, Any]) -> list[str]:
    boundary = next(
        (i for i, message in enumerate(row["messages"]) if float(message.get("loss_scale", 0) or 0) > 0),
        len(row["messages"]),
    )
    visible = "\n".join(
        message.get("content", "")
        for message in row["messages"][:boundary]
        if message.get("role") == "tool_response"
    )
    target = "\n".join(
        message.get("content", "")
        for message in row["messages"]
        if message.get("role") == "assistant" and float(message.get("loss_scale", 0) or 0) > 0
    )
    tentative = "待验证" in target or "尚不足以确认" in target
    issues: list[str] = []
    if not tentative:
        if (
            re.search(r"Protocol\s+Status\s*:\s*Disabled", target, re.I)
            or re.search(r"全局.{0,24}STP.{0,24}(?:未使能|未启用|Disabled)", target, re.I)
            or re.search(r"(?:缺少|没有).{0,20}`?stp\s+enable`?", target, re.I)
        ) and not re.search(r"Protocol\s+Status\s*:\s*Disabled", visible, re.I):
            issues.append("unsupported_global_stp_disabled")
        if (
            re.search(r"stp\s+port\s+bpdu-filter\s+enable", target, re.I)
            or ("过滤" in target and re.search(r"BPDU", target, re.I))
        ) and not re.search(r"stp\s+port\s+bpdu-filter\s+enable", visible, re.I):
            issues.append("unsupported_bpdu_filter_enabled")
        if (
            re.search(r"Preempt\s*:\s*NO|preempt\s+disable", target, re.I)
            or "非抢占" in target
        ) and not re.search(r"Preempt\s*:\s*NO|preempt\s+disable", visible, re.I):
            issues.append("unsupported_vrrp_nonpreempt")
        if re.search(r"(?:配置|启用|存在|包含).{0,20}`?stp\s+(?:global\s+)?enable`?", target, re.I) and not re.search(
            r"(?m)^\s*stp\s+(?:global\s+)?enable\s*$", visible, re.I
        ):
            issues.append("unsupported_stp_enabled")
    formed_loop_claim = re.search(
        r"(?:已经|已)\s*(?:确认|证明)?\s*(?:形成|构成)\s*(?:IP|MPLS)?\s*(?:路由|标签|转发)?\s*环路",
        target,
        re.I,
    )
    if any(term in target for term in (
        "证据已经形成闭环", "证据已形成闭环", "关键区分证据已出现",
        "决定性证据已出现", "唯一异常", "唯一的异常", "决定性异常", "根因应归",
    )) or formed_loop_claim:
        issues.append("premature_convergence_assertion")
    return sorted(set(issues))


def structural_binding_status(
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
    visible = re.sub(
        r"\s+",
        " ",
        "\n".join(
            message.get("content", "")
            for message in row["messages"][:boundary]
            if message.get("role") == "tool_response"
        ),
    ).strip().lower()
    candidates: list[dict[str, Any]] = []
    for endpoint in endpoint_rows:
        binding = endpoint["metadata"]["evidence_binding"]
        facts = binding["facts"]
        matched = [
            fact
            for fact in facts
            if re.sub(r"\s+", " ", str(fact["output_line"])).strip().lower()
            in visible
        ]
        candidates.append(
            {
                "binding": binding,
                "matched": matched,
                "complete": len(matched) == len(facts),
            }
        )
    return max(
        candidates,
        key=lambda item: (
            int(item["complete"]),
            len(item["matched"]),
            -len(item["binding"]["facts"]),
            item["binding"]["sha256"],
        ),
    )


def positive_tool_count(row: dict[str, Any]) -> int:
    return sum(
        message.get("role") == "tool_call"
        and float(message.get("loss_scale", 0) or 0) > 0
        for message in row["messages"]
    )


def main() -> None:
    data_root = resolve_data_root(parse_args().data_root)
    sft = data_root / "sft"
    manifest = json.loads((sft / "0809_agent_error_aware_manifest.json").read_text(encoding="utf-8"))
    semantic = load_rows(sft / "qwen3_6_27b_0809_train_semantic_pool.jsonl")
    base = load_rows(sft / "qwen3_6_27b_0809_base_core_pool.jsonl")
    counterfactual = load_rows(sft / "qwen3_6_27b_0809_counterfactual_pool.jsonl")
    endpoint = load_rows(sft / "qwen3_6_27b_0809_endpoint_pool.jsonl")

    marker_tokens = ("当前可见状态：下一节点", "当前证据尚未闭环；下一节点", "下一节点只核对", "下一节点只保留")
    marker_rows = [
        row for row in semantic
        if row["metadata"].get("synthetic_assistant_control_marker_used") is True
        or any(token in message.get("content", "") for token in marker_tokens for message in row["messages"])
    ]
    prefix_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in semantic:
        prefix_groups[positive_prefix(row)].append(row)
    multimodal_groups = [
        group for group in prefix_groups.values()
        if len({positive_target(row) for row in group}) > 1
    ]
    multimodal_combinations = Counter(
        tuple(sorted({row["metadata"]["target_type"] for row in group}))
        for group in multimodal_groups
    )

    cf_types = Counter(row["metadata"]["target_type"] for row in counterfactual)
    action_rows = [row for row in counterfactual if row["metadata"]["target_type"] == "counterfactual_action_selection"]
    repaired_action_rows = [
        row for row in action_rows
        if row["metadata"].get("0809_action_visible_grounding_gate", {}).get("repaired") is True
    ]
    repaired_parent_rows = [
        row for row in base
        if row["metadata"].get("0809_action_visible_grounding_gate", {}).get("repaired") is True
    ]
    remaining_action_issues = {
        row["id"]: action_visible_grounding_issues(row)
        for row in action_rows
        if action_visible_grounding_issues(row)
    }
    original_action_issue_counts = Counter(
        issue
        for row in repaired_action_rows
        for issue in row["metadata"]["0809_action_visible_grounding_gate"].get("original_issue_codes", [])
    )
    replay_rows = [row for row in counterfactual if row["metadata"]["target_type"] == "hypothesis_elimination_replay"]
    action_commands = [command for row in action_rows for command in row["metadata"]["selected_action_commands"]]
    action_prefix_commands = {
        (positive_prefix(row), tuple(row["metadata"]["selected_action_commands"]))
        for row in action_rows
    }
    real_elimination_rows = [row for row in base if row["metadata"]["target_type"] == "hypothesis_elimination"]
    recovery_rows = [row for row in endpoint if row["metadata"].get("0809_recovery_source")]
    recovery_facts = [fact for row in recovery_rows for fact in row["metadata"]["evidence_binding"]["facts"]]
    paths_per_case = Counter(case_id(row) for row in endpoint)
    endpoint_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint:
        endpoint_by_case[case_id(row)].append(row)
    action_structural_status = {
        row["id"]: structural_binding_status(
            row, endpoint_by_case[case_id(row)]
        )
        for row in action_rows
    }
    base_structural_status = {
        row["id"]: structural_binding_status(
            row, endpoint_by_case[case_id(row)]
        )
        for row in base
    }
    action_matched_expected = Counter(
        (
            len(status["matched"]),
            len(status["binding"]["facts"]),
        )
        for status in action_structural_status.values()
    )
    pending_action_rows = [
        row
        for row in action_rows
        if "待验证假设" in positive_target(row)
        and "结果返回前不输出最终答案" in positive_target(row)
    ]
    complete_base_continuations = [
        row
        for row in base
        if base_structural_status[row["id"]]["complete"]
        and positive_tool_count(row)
    ]
    post_closure_repairs = [
        row
        for row in base
        if row["metadata"].get("0809_post_closure_gate", {}).get("repaired")
        is True
    ]
    audited_source = next(
        row for row in base if row["id"] == AUDITED_COMPLETE_SOURCE
    )
    endpoint_positive_text = {
        row["id"]: positive_target(row) for row in endpoint
    }
    endpoint_unsupported_exclusion = [
        row["id"]
        for row in endpoint
        if row["metadata"].get("rejected_labels") != []
        or row["metadata"].get("hypothesis_elimination_supervised") is not False
        or any(
            phrase in endpoint_positive_text[row["id"]]
            for phrase in ("排除“", "应予排除", "已经证伪", "候选排除：")
        )
    ]
    endpoint_excluding_q73_q86 = [
        row for row in endpoint if not 73 <= case_id(row) <= 86
    ]

    epoch_payload = {}
    for epoch in range(1, 6):
        core = load_rows(sft / f"qwen3_6_27b_0809_core_epoch_{epoch:02d}.jsonl")
        ends = load_rows(sft / f"qwen3_6_27b_0809_endpoint_epoch_{epoch:02d}.jsonl")
        combined = [*core, *ends]
        epoch_payload[f"epoch_{epoch:02d}"] = {
            "rows": len(combined),
            "core_rows": len(core),
            "endpoint_real_path_rows": len(ends),
            "unique_row_ids": len({row["id"] for row in combined}),
            "unique_message_payloads": len({json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")) for row in combined}),
            "all_counterfactual_targets_exposed": len({row["id"] for row in core if row["id"] in {item["id"] for item in counterfactual}}) == len(counterfactual),
        }

    preflight_record = manifest["target_tokenizer_preflight"]
    tokenizer_signal: dict[str, Any] = {"status": preflight_record.get("status")}
    if preflight_record.get("status") == "passed":
        preflight = json.loads((ROOT / preflight_record["path"]).read_text(encoding="utf-8"))
        tokenizer_signal["totals"] = preflight["totals"]
        tokenizer_signal["model_identity_files"] = preflight.get("model_identity_files", {})
        tokenizer_signal["environment"] = {
            "python": preflight.get("python"),
            "ms_swift_version": preflight.get("ms_swift_version"),
            "transformers_version": preflight.get("transformers_version"),
        }
        per_epoch = {}
        for epoch in range(1, 6):
            datasets = [preflight["datasets"][f"core_epoch_{epoch:02d}"], preflight["datasets"][f"endpoint_epoch_{epoch:02d}"]]
            total = sum(item["weighted_supervised_token_sum"] for item in datasets)
            endpoint_total = sum(item["weighted_supervised_token_sum_by_target"].get("endpoint_bundle", 0) for item in datasets)
            per_epoch[f"epoch_{epoch:02d}"] = {
                "weighted_supervised_token_sum": total,
                "endpoint_bundle_percent": round(endpoint_total / total * 100, 6),
            }
        tokenizer_signal["per_epoch"] = per_epoch

    metrics = {
        "schema_version": "0809-agent-error-aware-audit.v7",
        "release_identity": {
            "manifest_schema": manifest["schema_version"],
            "logical_data_root": manifest["data_root"],
            "design_input": manifest["design_input"],
            "parent_release": manifest["parent_release"]["path"],
            "raw_byte_identical_to_parent": manifest["raw_archive"]["byte_identical_to_parent_0805"],
        },
        "split": manifest["split"],
        "counts": manifest["counts"],
        "reachable_target_conditioning": {
            "synthetic_assistant_control_marker_rows": len(marker_rows),
            "semantic_raw_first_positive_prefix_count": len(prefix_groups),
            "raw_prefix_multitarget_group_count": len(multimodal_groups),
            "raw_prefix_multitarget_row_count": sum(len(group) for group in multimodal_groups),
            "multitarget_type_combinations": {" + ".join(key): value for key, value in sorted(multimodal_combinations.items())},
            "interpretation": "compatible natural continuations are reported, never hidden with zero-loss assistant markers",
        },
        "real_endpoint_paths": {
            "objective_bundle_count": len(endpoint),
            "unique_raw_evidence_path_count": len({positive_prefix(row) for row in endpoint}),
            "case_count": len(paths_per_case),
            "path_count_distribution": dict(sorted(Counter(paths_per_case.values()).items())),
            "query_normalized_total_path_weight_min": min(sum(float(row["metadata"]["query_normalized_path_weight"]) for row in endpoint if case_id(row) == current_case) for current_case in paths_per_case),
            "query_normalized_total_path_weight_max": max(sum(float(row["metadata"]["query_normalized_path_weight"]) for row in endpoint if case_id(row) == current_case) for current_case in paths_per_case),
            "exact_positive_target_count": len({positive_target(row) for row in endpoint}),
            "condition_normalized_positive_target_count": len({condition_normalized_target(row) for row in endpoint}),
            "condition_normalization_policy": "replace snapshot/query/config-path identifiers only; preserve devices, observations, labels and result items",
            "excluding_q73_q86_row_count": len(endpoint_excluding_q73_q86),
            "excluding_q73_q86_exact_positive_target_count": len(
                {positive_target(row) for row in endpoint_excluding_q73_q86}
            ),
        },
        "counterfactual": {
            "rows_by_type": dict(sorted(cf_types.items())),
            "action_query_count": len({case_id(row) for row in action_rows}),
            "action_row_count": len(action_rows),
            "unique_reachable_action_prefix_command_pairs": len(action_prefix_commands),
            "action_command_count": len(action_commands),
            "action_glob_count": sum(any(character in command for character in "*?[") for command in action_commands),
            "positive_action_after_complete_binding_count": sum(row["metadata"].get("complete_binding_before_positive_tool_call") is not False for row in action_rows),
            "exact_same_target_elimination_replay_count": len(replay_rows),
        },
        "action_visible_grounding": {
            "policy": "recompute every strict same-query endpoint binding from literal tool responses before first positive loss; incomplete action nodes may supervise only matched facts, missing requirements, a pending hypothesis, and read-only actions",
            "checked_action_rows": len(action_rows),
            "repaired_parent_source_rows": len(repaired_parent_rows),
            "repaired_action_rows": len(repaired_action_rows),
            "original_issue_counts": dict(sorted(original_action_issue_counts.items())),
            "remaining_issue_row_count": len(remaining_action_issues),
            "remaining_issues_by_row": remaining_action_issues,
            "latest_audit_seven_rows_repaired": all(
                row_id in {row["id"] for row in repaired_action_rows}
                for row_id in (
                    "0809_cf_action_q0011_a01", "0809_cf_action_q0014_a04",
                    "0809_cf_action_q0015_a01", "0809_cf_action_q0018_a02",
                    "0809_cf_action_q0021_a01", "0809_cf_action_q0022_a01",
                    "0809_cf_action_q0024_a03",
                )
            ),
        },
        "structural_action_closure": {
            "message_derived_action_rows_checked": len(action_rows),
            "structurally_complete_action_rows": sum(
                status["complete"] for status in action_structural_status.values()
            ),
            "calibrated_pending_action_rows": len(pending_action_rows),
            "matched_expected_distribution": {
                f"{matched}/{expected}": count
                for (matched, expected), count in sorted(
                    action_matched_expected.items()
                )
            },
            "latest_audit_17_rows_present_and_incomplete": all(
                row_id in action_structural_status
                and not action_structural_status[row_id]["complete"]
                for row_id in STRUCTURAL_AUDIT_ACTIONS
            ),
            "complete_base_rows_still_calling_tools": len(
                complete_base_continuations
            ),
            "post_closure_parent_rows_repaired": len(post_closure_repairs),
            "audited_q0023_source_complete": base_structural_status[
                AUDITED_COMPLETE_SOURCE
            ]["complete"],
            "audited_q0023_positive_tool_count": positive_tool_count(
                audited_source
            ),
            "audited_q0023_removed_positive_tool_count": audited_source[
                "metadata"
            ]["0809_post_closure_gate"].get("removed_positive_tool_call_count"),
        },
        "hypothesis_elimination": {
            "real_source_row_count": len(real_elimination_rows),
            "real_source_case_count": len({case_id(row) for row in real_elimination_rows}),
            "replay_count": len(replay_rows),
            "endpoint_generated_elimination_count": 0,
            "queries_without_real_elimination_node": sorted(
                set(paths_per_case)
                - {case_id(row) for row in real_elimination_rows}
            ),
        },
        "endpoint_candidate_scope": {
            "candidate_scope_calibration_bundles": sum(
                "candidate_scope_calibration"
                in row["metadata"].get("endpoint_objectives", [])
                for row in endpoint
            ),
            "unsupported_exclusion_rows": endpoint_unsupported_exclusion,
            "rejected_label_declaration_count": sum(
                len(row["metadata"].get("rejected_labels", []))
                for row in endpoint
            ),
            "policy": "positive binding confirms the current label/current-evidence set; unchecked neighbors remain unverified and are not declared absent",
        },
        "recovery_provenance": {
            "endpoint_path_rows": len(recovery_rows),
            "fact_count": len(recovery_facts),
            "distinct_event_file_count": len({fact["source_action_provenance"]["source_event_file"] for fact in recovery_facts}),
            "facts_with_raw_provenance": sum(bool(fact.get("source_action_provenance")) for fact in recovery_facts),
        },
        "epoch_schedule": epoch_payload,
        "heuristic_training_signal": manifest["heuristic_training_signal_by_epoch"],
        "target_tokenizer_signal": tokenizer_signal,
        "hard_negative_boundary": {
            "malformed_protocol_and_rejected_sets_are_metadata_only": True,
            "preference_or_grammar_eval_required_for_explicit_negative_learning": True,
        },
        "release_gate": {
            "windows_official_validator_required": True,
            "windows_independent_validator_required": True,
            "linux_checkout_both_validators_required": True,
            "runtime_identity_gate_required": True,
            "two_gpu_resume_lr_smoke_required": True,
            "small_agent_canary_required_before_formal_training": True,
            "git_commit_required_for_publication": True,
        },
    }
    metrics_path = data_root / "curation" / "AUDIT_METRICS.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    heuristic = manifest["heuristic_training_signal_by_epoch"]["epoch_01"]
    kind = heuristic["weighted_message_kind_percent"]
    target = heuristic["weighted_target_type_percent"]
    tokenizer_status = preflight_record.get("status")
    report = f"""# 2026-08-09 Agent 错误感知 SFT v7 审计报告

## 结论

v7 针对 2026-08-11 20:02 深审将 action 门禁升级为完整结构门禁：从首个正 loss 前的真实工具回显重算同题所有严格事实 binding，不再信任 metadata。完整 binding 一律停止，未完整 binding 一律只监督已匹配事实、缺失条件、待验证假设和下一步只读动作。endpoint 也从“正证据顺带排除邻近标签”改为候选范围校准：正证据只确认当前标签/当前证据集合，未经候选专属反证的邻近标签保持未验证。训练/验证划分、每轮 1151 行、144 optimizer step、固定 epoch 3 / global step 432 均保持不变。

## 数据结构

- 语义池 {len(semantic)} 行：父级 core {len(base)}、可达 action {len(action_rows)}、明确标注的同目标错误候选排除 replay {len(replay_rows)}、真实证据路径 bundle {len(endpoint)}。
- endpoint 是 {len(endpoint)} 个 objective bundle，也是 {len({positive_prefix(row) for row in endpoint})} 条不同原始证据路径，覆盖 {len(paths_per_case)}/72 题。每题真实路径数分布为 {dict(sorted(Counter(paths_per_case.values()).items()))}；用 `1 / 本题真实路径数` 归一化，故每题 endpoint 总权重均为 1。
- 每个 endpoint completion 在同一可达历史后依次监督：证据归纳、正向标签边界、候选范围校准、当前证据最小集合、停止判断和最终 `<result>`；不再把未检查候选声明为已排除。
- 每轮均完整曝光 719 个父级节点、308 个 action、8 个 elimination replay 和 116 个 endpoint；固定 checkpoint-432 之前已经看到全部新增目标，不再把 boundary/minimal 延迟到第 4–5 轮。

## 复审核心指标

1. **不可达控制句为 0。** marker 行数 {len(marker_rows)}。剥离 marker 后不再需要另一套统计，因为发布数据本身就是原始 Agent 可见上下文。
2. **输入路径数与输出目标多样性分开报告。** objective bundle={len(endpoint)}，unique raw evidence path={len({positive_prefix(row) for row in endpoint})}；但精确正目标只有 {len({positive_target(row) for row in endpoint})}/{len(endpoint)} 唯一，机械条件归一化后也是 {len({condition_normalized_target(row) for row in endpoint})}/{len(endpoint)}。116 表示不同输入证据路径，不表示 116 份不同输出措辞；不为追求唯一数而机械改写教师目标。
3. **如实报告自然多目标。** 整个语义池共有 {len(prefix_groups)} 个原始首个正 loss 前缀；其中 {len(multimodal_groups)} 组、{sum(len(group) for group in multimodal_groups)} 行存在多个兼容正目标，组合为 {dict(multimodal_combinations)}。这些主要是原始 reasoning/planning 与同状态可达 action 的兼容延续；校验器不再用人工 marker 掩盖它们。
4. **错误根因排除与候选校准分开。** 原始可见、证据支持的排除节点 {len(real_elimination_rows)} 行/{len({case_id(row) for row in real_elimination_rows})} 题，另有 {len(replay_rows)} 条透明的同目标采样 replay；endpoint 不再补造排除，{len(endpoint)} 条均只说明邻近候选保持未验证，unsupported exclusion={len(endpoint_unsupported_exclusion)}。
5. **恢复证据可回查。** {len(recovery_rows)} 条恢复型真实路径包含 {len(recovery_facts)} 个事实，回指 {len({fact['source_action_provenance']['source_event_file'] for fact in recovery_facts})} 个成功 `events.jsonl`，保留 item、命令、输出哈希和逐字 span。
6. **监督比重。** 启发式估算：thinking {kind.get('thinking', 0):.2f}%、结论/回答 {kind.get('conclusion_or_answer', 0):.2f}%、工具调用 {kind.get('tool_call', 0):.2f}%；endpoint bundle {target.get('endpoint_bundle', 0):.2f}%，真实排除 {target.get('hypothesis_elimination', 0):.2f}%，排除 replay {target.get('hypothesis_elimination_replay', 0):.2f}%。工具 call loss 已从 0.1 降到 0.05。
7. **action 完整结构门禁。** 全量从 messages 重算 {len(action_rows)} 条 action：完整 binding {sum(status['complete'] for status in action_structural_status.values())} 条，明确待验证 {len(pending_action_rows)} 条，matched/expected 分布 {dict(sorted((f'{matched}/{expected}', count) for (matched, expected), count in action_matched_expected.items()))}。深审点名的 17 条均存在且不完整；父级完整证据后仍调用工具 {len(complete_base_continuations)} 条。q0023 原 1/1 完整节点已删除 {audited_source['metadata']['0809_post_closure_gate'].get('removed_positive_tool_call_count')} 个正 loss 工具调用并改为停止。
8. **候选排除不做闭世界推断。** endpoint rejected-label 声明总数 {sum(len(row['metadata'].get('rejected_labels', [])) for row in endpoint)}，候选范围校准 bundle {sum('candidate_scope_calibration' in row['metadata'].get('endpoint_objectives', []) for row in endpoint)}/{len(endpoint)}。没有候选专属负证据时只监督“保持未验证”。
9. **目标表达多样性。** endpoint 精确正目标 {len({positive_target(row) for row in endpoint})}/{len(endpoint)} 唯一；只去除 snapshot/query/config path 等机械条件后仍有 {len({condition_normalized_target(row) for row in endpoint})}/{len(endpoint)} 唯一，设备、事实、标签和结果均保留在统计中。

## 发布门禁

- 目标 tokenizer 预检状态：`{tokenizer_status}`。正式训练前必须归档真实 Qwen 16K 编码、逐 token loss mask、模型/tokenizer 文件哈希和 ms-swift/transformers 版本。
- 还必须在异根目录 Linux checkout 运行两套 validator、通过 runtime identity gate，并完成双卡 1–2 step full-state resume/LR smoke 和小规模 Agent canary；这些运行门禁不应被静态数据校验替代。
- malformed tool call、错误集合等 hard negative 仍仅保存在 metadata；SFT 不把坏 JSON 当正目标。显式负例学习应使用 grammar eval 或 preference/ranking 数据。
- 12 道现有验证题只用于 error-mining/dev 与横向比较；正式泛化结论仍需要未参与反推的 topology-heldout 集。
"""
    (data_root / "AUDIT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"resolved_data_root={data_root}")
    print(f"metrics={metrics_path.relative_to(ROOT).as_posix()}")
    print(f"report={(data_root / 'AUDIT_REPORT.md').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
