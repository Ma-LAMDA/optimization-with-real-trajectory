#!/usr/bin/env python3
"""Independently validate the 0807 evidence-gated SFT artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import convert_0804_best_trajectory_reasoning_sft as base
import convert_0807_evidence_gated_reasoning_sft as converter


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-07"
FROZEN_0804 = ROOT / "data" / "2026-08-04" / "curation" / "accepted_trajectory_selection.json"
CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
SELECTION = DATA_ROOT / "curation" / "causal_path_clusters_per_case.json"
MANIFEST = DATA_ROOT / "sft" / "0807_evidence_gated_manifest.json"
SOURCE_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
VALID_CASES = set(range(1, 101)) - set(range(41, 57))
INCLUSIVE_OR_CASES = set(range(73, 87))
ROLE_REASON = "VRRP Master角色规划不合理"
WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\")
VALIDATOR_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
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
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def output_path(record: dict[str, Any]) -> Path:
    return ROOT / str(record["path"])


def assert_output_record(name: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    path = output_path(record)
    rows = load_jsonl(path)
    if len(rows) != int(record["rows"]):
        raise ValueError(f"{name}: row count differs from manifest")
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{name}: byte count differs from manifest")
    if base.digest_file(path) != record["sha256_lf_normalized"]:
        raise ValueError(f"{name}: SHA256 differs from manifest")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate row IDs")
    return rows


def current_supervised_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(message["content"])
        for message in row["messages"]
        if message["role"] == "assistant"
        and float(message.get("loss_scale", 0) or 0) > 0
    )


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def independently_infer_families(text: str) -> set[str]:
    lowered = re.sub(r"\bsaved_configs\b", "", text.lower())
    families: set[str] = set()
    for family, patterns in VALIDATOR_FAMILY_PATTERNS:
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


def independently_safe_procedural(sentence: str) -> bool:
    stripped = sentence.strip().lstrip("而并且且但、，,；;：: ")
    return (
        PURE_PROCEDURAL_PREFIX.match(stripped) is not None
        and not any(marker in stripped for marker in PROCEDURAL_FACT_MARKERS)
    )


def option_key(items: list[str]) -> frozenset[str]:
    return frozenset(str(item) for item in items)


def inclusive_or_options_by_case() -> dict[int, list[list[str]]]:
    rows = load_jsonl(SOURCE_DATASET)
    result: dict[int, list[list[str]]] = {}
    for row in rows:
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
        raise ValueError("source dataset does not contain all q73-q86 options")
    return result


def prior_visible_tool_text(row: dict[str, Any]) -> str:
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


def validate_source() -> tuple[set[int], set[int]]:
    source = load_json(CURATION)
    frozen = load_json(FROZEN_0804)
    if source.get("schema_version") != "codex-ip-accepted-trajectory-curation.v5":
        raise ValueError("0807 curation is not the inclusive-OR v5 schema")
    if source.get("source_dataset") != SOURCE_DATASET.relative_to(ROOT).as_posix():
        raise ValueError("0807 curation source dataset path is stale")
    if source.get("source_dataset_sha256_lf_normalized") != base.digest_file(
        SOURCE_DATASET
    ):
        raise ValueError("0807 curation source dataset hash is stale")
    options_by_case = inclusive_or_options_by_case()
    if source["source_attempt_status_counts"] != {
        "accepted": 840, "format_error": 0, "incorrect": 502
    }:
        raise ValueError("0807 source status accounting is not model-valid-only")
    if source["counts"]["source_attempt_count_scope"] != "model_valid_outcomes_only":
        raise ValueError("0807 source count scope includes infrastructure outcomes")
    selected = [item for item in source["trajectories"] if item.get("selected")]
    if len(selected) != 840:
        raise ValueError("0807 must contain 840 selected successful trajectories")
    per_case = Counter(int(item["case_id"]) for item in selected)
    if set(per_case) != VALID_CASES or set(per_case.values()) != {10}:
        raise ValueError("0807 must contain ten successful trajectories for each of 84 cases")
    affected = dual_targets = 0
    for item in selected:
        if not str(item["raw_file"]).startswith("data/2026-08-07/raw/"):
            raise ValueError(f"{item['id']}: raw source does not point into 0807")
        raw_path = ROOT / item["raw_file"]
        if base.digest_file(raw_path) != item["raw_sha256_lf_normalized"]:
            raise ValueError(f"{item['id']}: raw source hash mismatch")
        raw = load_json(raw_path)
        if not raw["answer_matches_reference"] or not raw["independent_judgment"]["correct"]:
            raise ValueError(f"{item['id']}: source is not strictly successful")
        case_id = int(item["case_id"])
        if case_id in INCLUSIVE_OR_CASES:
            affected += 1
            options = options_by_case[case_id]
            if (
                item.get("reference_answer_options") != options
                or raw.get("reference_answer_options") != options
                or not item.get("reference_answer_revision")
                or not raw.get("reference_answer_revision")
            ):
                raise ValueError(f"{item['id']}: inclusive-OR archive is not synchronized")
            prediction = [str(value) for value in raw.get("actual_result_items") or []]
            if option_key(prediction) not in {option_key(option) for option in options}:
                raise ValueError(f"{item['id']}: accepted answer misses all OR options")
            dual_targets += len(prediction) == 2
    policy = source.get("reference_answer_policy") or {}
    if (
        affected != 140
        or dual_targets != 0
        or policy.get("accepted_options_per_case") != 3
        or policy.get("updated_trajectory_count") != 140
        or policy.get("existing_dual_device_trajectory_count") != 0
        or policy.get("split_or_success_count_changed") is not False
    ):
        raise ValueError("q73-q86 reference revision accounting is inconsistent")
    train = set(source["split"]["train_case_ids"])
    validation = set(source["split"]["validation_case_ids"])
    if train & validation or train | validation != VALID_CASES:
        raise ValueError("0807 case split is invalid")
    if train != set(frozen["split"]["train_case_ids"]) or validation != set(
        frozen["split"]["validation_case_ids"]
    ):
        raise ValueError("0807 split differs from frozen 0804")
    return train, validation


def validate_manifest_and_files() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = load_json(MANIFEST)
    selection = load_json(SELECTION)
    if manifest["schema_version"] != "qwen36-0807-evidence-gated-case-balanced-sft.v7":
        raise ValueError("unexpected 0807 manifest schema")
    if selection["schema_version"] != "0807-evidence-gated-cluster-selection.v6":
        raise ValueError("unexpected 0807 selection schema")
    if manifest["status"] not in {
        "rule_validated_draft_requires_target_tokenizer_preflight",
        "rule_and_target_tokenizer_validated_release_candidate",
    }:
        raise ValueError("unexpected 0807 release status")
    if base.digest_file(CURATION) != manifest["source_curation_sha256_lf_normalized"]:
        raise ValueError("curation hash mismatch")
    if base.digest_file(SELECTION) != manifest["cluster_selection_sha256_lf_normalized"]:
        raise ValueError("selection hash mismatch")
    for case in selection["cases"]:
        for candidate in case["candidates"]:
            if candidate["quality"].get("label_features_used") is not False:
                raise ValueError("candidate ranking used verified-label features")
        for cluster in case["clusters"]:
            if cluster["representative_quality"].get("label_features_used") is not False:
                raise ValueError("representative selection used verified-label features")
    tracked = manifest["reproducibility"]["tracked_files"]
    required = {
        "root_readme", "date_readme", "reproducibility_document",
        "causal_path_converter", "causal_path_validator", "training_entry",
        "fixed_lr_callback", "tokenizer_preflight", "audit_report",
        "audit_metrics", "audit_generator", "independent_release_validator",
        "inclusive_or_source_updater", "inclusive_or_archive_sync",
        "source_dataset",
    }
    if not required <= set(tracked):
        raise ValueError("manifest does not track every reproducibility-critical file")
    for name, record in tracked.items():
        path = ROOT / record["path"]
        if base.digest_file(path) != record["sha256_lf_normalized"]:
            raise ValueError(f"tracked reproducibility file is stale: {name}")
    if manifest["status"] == "rule_and_target_tokenizer_validated_release_candidate":
        preflight = manifest.get("target_tokenizer_preflight", {})
        if (
            preflight.get("status") != "passed"
            or int(preflight.get("totals", {}).get("over_max_length_rows", -1)) != 0
            or int(preflight.get("totals", {}).get("loss_mask_failures", -1)) != 0
            or "target_tokenizer_preflight_report" not in tracked
        ):
            raise ValueError("release status lacks a passing target-tokenizer report")
    rows_by_output = {
        name: assert_output_record(name, record)
        for name, record in manifest["outputs"].items()
    }
    return manifest, rows_by_output


def validate_tool_and_loss_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_count = 0
    config_count = 0
    max_action_count = 0
    source_coverage: Counter[str] = Counter()
    supervised_coverage: Counter[str] = Counter()
    lldp_actions = 0
    lldp_regression_rows: set[str] = set()
    for row in rows:
        metadata = row["metadata"]
        messages = row["messages"]
        if messages[0] != {"role": "system", "content": converter.CODEX_CLI_SYSTEM_PROMPT}:
            raise ValueError(f"{row['id']}: system prompt mismatch")
        tools = json.loads(row["tools"])
        if tools != converter.CODEX_CLI_TOOLS:
            raise ValueError(f"{row['id']}: tool schema mismatch")
        question_match = re.search(
            r"(CampusNetwork(?:-for-perf)?_\d+)",
            next(message["content"] for message in messages if message["role"] == "user"),
            re.IGNORECASE,
        )
        if not question_match:
            raise ValueError(f"{row['id']}: question snapshot missing")
        target_snapshot = question_match.group(1).lower()
        supervised_target = current_supervised_text(row)
        if WINDOWS_PATH.search(supervised_target):
            raise ValueError(f"{row['id']}: trainable assistant target contains Windows path")
        for message in messages:
            loss_scale = float(message.get("loss_scale", 0) or 0)
            if message["role"] in {"system", "user", "tool_response"} and loss_scale > 0:
                raise ValueError(f"{row['id']}: context role participates in loss")
            if message["role"] != "tool_call":
                continue
            payload = json.loads(message["content"])
            if payload.get("name") != "exec_command" or set(payload.get("arguments", {})) != {"cmd"}:
                raise ValueError(f"{row['id']}: invalid tool call protocol")
            cmd = str(payload["arguments"]["cmd"])
            if (
                WINDOWS_PATH.search(cmd)
                or "powershell" in cmd.lower()
                or "saved_configs\\" in cmd
            ):
                raise ValueError(f"{row['id']}: non-Linux supervised command")
            if loss_scale > 0 and abs(loss_scale - converter.TOOL_CALL_LOSS_SCALE) > 1e-12:
                raise ValueError(f"{row['id']}: current tool loss mismatch")
        actions = metadata["current_actions"]
        max_action_count = max(max_action_count, len(actions))
        for action in actions:
            action_count += 1
            semantics = action["causal_semantics"]
            independently_inferred = independently_infer_families(
                f"{action['command']} {' '.join(semantics.get('filenames', []))}"
            )
            if independently_inferred != set(semantics["families"]):
                raise ValueError(
                    f"{row['id']}: metadata family differs from command/filename "
                    f"({sorted(semantics['families'])} != {sorted(independently_inferred)})"
                )
            if "lldp" in independently_inferred:
                lldp_actions += 1
                if row["id"] in KNOWN_LLDP_REGRESSION_ROWS:
                    lldp_regression_rows.add(row["id"])
                if (
                    "lldp" not in supervised_target.lower()
                    or re.search(r"(?<![a-z0-9])(?:mpls|lsp|ldp)(?![a-z0-9])", supervised_target.lower())
                ):
                    raise ValueError(f"{row['id']}: LLDP action has an MPLS/LSP supervised intent")
            if set(semantics["snapshots"]) != {target_snapshot} or semantics["has_path_glob"]:
                raise ValueError(f"{row['id']}: current action crosses snapshots")
            if "config" in semantics["families"]:
                config_count += 1
                semantic_surface = re.sub(r"\bsaved_configs\b", "", action["source_command"].lower())
                if "config" not in semantic_surface and "configuration" not in semantic_surface:
                    raise ValueError(f"{row['id']}: config family is directory-name pollution")
        thinking_source = metadata["thinking_source"]
        if thinking_source == "fixed_bridge_template" and metadata["loss_policy"]["thinking"] != 0:
            raise ValueError(f"{row['id']}: fixed thinking bridge participates in loss")
        selection = metadata["action_selection"]
        if int(selection["kept_action_count"]) != len(actions):
            raise ValueError(f"{row['id']}: action selection count is stale")
        if actions:
            if abs(
                float(selection["claim_coverage_after"])
                - float(selection["claim_coverage_before"])
            ) > 1e-12:
                raise ValueError(f"{row['id']}: minimal claim cover lost source coverage")
            source_status = str(selection["claim_coverage_status_after"])
            supervised_status = str(selection["supervised_intent_coverage_status"])
            if source_status not in {"full", "partial", "zero", "unscoped"}:
                raise ValueError(f"{row['id']}: invalid source coverage status")
            if supervised_status not in {"full", "partial", "zero", "unscoped"}:
                raise ValueError(f"{row['id']}: invalid supervised coverage status")
            if supervised_status not in {"full", "unscoped"}:
                raise ValueError(
                    f"{row['id']}: supervised intent is not fully covered by current actions"
                )
            source_coverage[source_status] += 1
            supervised_coverage[supervised_status] += 1
            bindings = selection.get("selected_action_bindings") or []
            if {
                str(binding.get("action_id")) for binding in bindings
            } != {str(action["action_id"]) for action in actions}:
                raise ValueError(f"{row['id']}: final action bindings are incomplete")
            if source_status == "zero" and selection.get(
                "claim_cover_retained_without_decrease"
            ) is not True:
                raise ValueError(f"{row['id']}: zero coverage was mislabeled as success")
        if metadata["target_type"] == "endpoint_bundle":
            component_loss = metadata.get("endpoint_component_loss", {})
            if set(component_loss) != set(converter.ENDPOINT_COMPONENT_TYPES):
                raise ValueError(f"{row['id']}: endpoint component loss is incomplete")
            positive = [
                message for message in messages
                if message["role"] == "assistant"
                and float(message.get("loss_scale", 0) or 0) > 0
            ]
            if len(positive) != 5 or sum("<result>" in message["content"] for message in positive) != 1:
                raise ValueError(f"{row['id']}: endpoint bundle does not supervise 2+2+1 messages")
    if not action_count or config_count / action_count >= 0.30:
        raise ValueError("config family remains overrepresented")
    return {
        "actions": action_count,
        "config_actions": config_count,
        "max_actions_in_stage": max_action_count,
        "source_intent_coverage": dict(sorted(source_coverage.items())),
        "supervised_intent_coverage": dict(sorted(supervised_coverage.items())),
        "lldp_actions": lldp_actions,
        "lldp_regression_rows_checked": len(lldp_regression_rows),
    }


def validate_visible_reasoning_grounding(rows: list[dict[str, Any]]) -> dict[str, int]:
    grounded = procedural = dropped = 0
    for row in rows:
        metadata = row["metadata"]
        if metadata["target_type"] == "hypothesis_elimination":
            continue
        optimization = metadata["text_optimization"]
        records = optimization.get("retained_thinking_sentence_records", [])
        dropped_sentences = optimization.get("dropped_unsupported_thinking_sentences", [])
        target = normalized(current_supervised_text(row))
        prior_responses = prior_visible_tool_text(row)
        for record in records:
            sentence = normalized(str(record["sentence"]))
            if sentence not in target:
                raise ValueError(f"{row['id']}: retained source sentence is absent from target")
            if record["kind"] == "procedural_plan":
                procedural += 1
                if record["action_ids"]:
                    raise ValueError(f"{row['id']}: procedural plan has evidence IDs")
                if record.get("claim_bindings"):
                    raise ValueError(f"{row['id']}: procedural plan has factual bindings")
                if not independently_safe_procedural(str(record.get("source_sentence") or "")):
                    raise ValueError(f"{row['id']}: mixed factual sentence bypassed grounding")
            elif record["kind"] == "observation_bound_reasoning":
                grounded += 1
                source_sentence = normalized(str(record.get("source_sentence") or ""))
                if not source_sentence or source_sentence == sentence:
                    raise ValueError(f"{row['id']}: factual source was not rewritten")
                if source_sentence in target:
                    raise ValueError(f"{row['id']}: unsupported source inference leaked into loss")
                bindings = record.get("claim_bindings") or []
                if not record["action_ids"] or not bindings:
                    raise ValueError(f"{row['id']}: factual sentence lacks atomic bindings")
                bound_ids: set[str] = set()
                for binding in bindings:
                    claim = normalized(str(binding.get("claim") or ""))
                    evidence = binding.get("evidence") or []
                    if not claim or claim not in sentence or not evidence:
                        raise ValueError(f"{row['id']}: factual claim binding is incomplete")
                    for item in evidence:
                        action_id = str(item.get("action_id") or "")
                        span = str(item.get("observation_span") or "")
                        bound_ids.add(action_id)
                        if (
                            not action_id
                            or not converter.is_substantive_observation_line(span)
                            or normalized(span) not in prior_responses
                        ):
                            raise ValueError(
                                f"{row['id']}: factual span is empty/header/future evidence"
                            )
                if bound_ids != set(record["action_ids"]):
                    raise ValueError(f"{row['id']}: factual action binding set mismatch")
            else:
                raise ValueError(f"{row['id']}: unknown sentence grounding kind")
        for sentence in dropped_sentences:
            dropped += 1
            if normalized(sentence) and normalized(sentence) in target:
                raise ValueError(f"{row['id']}: unsupported sentence leaked into target")
        if any(fragment in current_supervised_text(row) for fragment in UNSAFE_PROCEDURAL_TARGET_FRAGMENTS):
            raise ValueError(f"{row['id']}: known mixed procedural regression leaked into loss")
    return {
        "observation_bound_factual_sentences": grounded,
        "procedural_source_sentences": procedural,
        "dropped_unsupported_sentences": dropped,
    }


def validate_elimination_grounding(
    rows: list[dict[str, Any]], source: dict[str, Any]
) -> dict[str, int]:
    annotations = {
        str(item["id"]): item for item in source["trajectories"] if item.get("selected")
    }
    count = auxiliary = 0
    for row in rows:
        metadata = row["metadata"]
        if metadata["target_type"] != "hypothesis_elimination":
            continue
        count += 1
        auxiliary += bool(metadata.get("auxiliary_elimination_from_non_endpoint_path"))
        statements = metadata.get("rejected_candidate_statements")
        if (
            not isinstance(statements, list)
            or not 1 <= len(statements) <= 2
            or not all(isinstance(value, str) and value.strip() for value in statements)
        ):
            raise ValueError(f"{row['id']}: invalid elimination statements")
        if not all(
            any(marker in statement for marker in converter.ELIMINATION_MARKERS)
            for statement in statements
        ):
            raise ValueError(f"{row['id']}: elimination lacks a rejection marker")
        if (
            metadata.get("thinking_source") != "claim_bound_hypothesis_elimination"
            or metadata.get("conclusion_source")
            != "source_grounded_hypothesis_elimination"
            or metadata.get("elimination_derived_from_visible_source") is not True
        ):
            raise ValueError(f"{row['id']}: elimination provenance is invalid")
        trajectory_id = str(metadata["trajectory_id"])
        annotation = annotations.get(trajectory_id)
        if annotation is None:
            raise ValueError(f"{row['id']}: elimination trajectory is absent from curation")
        messages, _ = base.parse_events(ROOT / annotation["events_file"])
        source_index = int(metadata["source_message_index"])
        if source_index >= len(messages):
            raise ValueError(f"{row['id']}: elimination source message index is invalid")
        source_text = str(messages[source_index]["text"])
        if not all(statement in source_text for statement in statements):
            raise ValueError(f"{row['id']}: elimination statement is absent from source")
        target = current_supervised_text(row)
        claims = metadata.get("supervised_elimination_claims")
        bindings = metadata.get("elimination_claim_bindings")
        if (
            not isinstance(claims, list)
            or not claims
            or not isinstance(bindings, list)
            or len(bindings) != len(claims)
        ):
            raise ValueError(f"{row['id']}: elimination claim bindings are missing")
        if any(str(claim) not in target for claim in claims):
            raise ValueError(f"{row['id']}: supervised elimination fact is absent from target")
        evidence_ids = metadata.get("elimination_evidence_action_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids)
            != int(metadata.get("elimination_evidence_action_count", 0))
        ):
            raise ValueError(f"{row['id']}: invalid elimination evidence IDs")
        for action_id in evidence_ids:
            match = re.fullmatch(r"S(\d+)-A\d+", str(action_id))
            if not match or int(match.group(1)) - 1 >= source_index:
                raise ValueError(f"{row['id']}: elimination cites current/future evidence")
        prior = prior_visible_tool_text(row)
        bound_ids: set[str] = set()
        for binding, supervised_fact in zip(bindings, claims):
            if binding.get("supervised_fact") != supervised_fact:
                raise ValueError(f"{row['id']}: elimination supervised fact is stale")
            anchors = binding.get("anchors")
            evidence = binding.get("evidence")
            if not anchors or not evidence:
                raise ValueError(f"{row['id']}: empty elimination claim binding")
            combined = normalized(" ".join(
                str(item.get("observation_span") or "") for item in evidence
            ))
            for anchor in anchors:
                if not converter.observation_contains_anchor(combined, str(anchor)):
                    raise ValueError(f"{row['id']}: elimination anchor is not visible")
            for item in evidence:
                action_id = str(item.get("action_id") or "")
                observation = str(item.get("observation_span") or "")
                bound_ids.add(action_id)
                if action_id not in evidence_ids:
                    raise ValueError(f"{row['id']}: elimination binding cites unknown action")
                if (
                    normalized(observation) not in prior
                    or not converter.is_substantive_observation_line(observation)
                ):
                    raise ValueError(
                        f"{row['id']}: elimination span is absent, empty, or a header"
                    )
        if bound_ids != set(evidence_ids):
            raise ValueError(f"{row['id']}: elimination evidence IDs are not exact")
        if row["id"] in {
            "q0004_path_05_success_08_step_03",
            "q0008_path_04_success_06_step_03",
            "q0032_path_01_success_06_step_03",
        }:
            raise ValueError(f"{row['id']}: known unsupported elimination regression")
    return {"nodes": count, "auxiliary_non_endpoint_nodes": auxiliary}


def validate_role_misalignment_closure(
    row: dict[str, Any], facts: list[dict[str, Any]], expected_devices: set[str]
) -> int:
    host_facts = [fact for fact in facts if fact.get("kind") == "source_host_ipv4"]
    if len(host_facts) != 1:
        raise ValueError(f"{row['id']}: role gate needs exactly one source-host fact")
    host = host_facts[0]
    vlan = int(host["vlan"])
    user = normalized(next(
        message["content"] for message in row["messages"] if message["role"] == "user"
    ))
    if str(host["device"]).lower() not in user:
        raise ValueError(f"{row['id']}: source-host fact is not the question source")
    closed: set[str] = set()
    for device in expected_devices:
        masters = [
            fact for fact in facts
            if fact.get("kind") == "vrrp_master" and fact.get("device") == device
        ]
        master_matches = False
        for master in masters:
            if master.get("vlan") is not None:
                master_matches = int(master["vlan"]) == vlan
            else:
                master_matches = any(
                    fact.get("kind") == "vrrp_interface_context"
                    and fact.get("device") == device
                    and fact.get("action_id") == master.get("action_id")
                    and int(fact.get("vlan", -1)) == vlan
                    for fact in facts
                )
            if master_matches:
                break
        mappings = [
            fact for fact in facts
            if fact.get("kind") == "mst_vlan_instance_mapping"
            and fact.get("device") == device
            and vlan in {int(value) for value in fact.get("vlans") or []}
        ]
        same_instance = any(
            fact.get("kind") == "stp_alternate_discarding"
            and fact.get("device") == device
            and int(fact.get("instance", -1)) == int(mapping["instance"])
            for mapping in mappings
            for fact in facts
        )
        if master_matches and mappings and same_instance:
            closed.add(device)
    if closed != expected_devices:
        raise ValueError(
            f"{row['id']}: role evidence does not close source VLAN {vlan} "
            f"for every answer device ({sorted(closed)} != {sorted(expected_devices)})"
        )
    return vlan


def validate_endpoint_gate(
    rows: list[dict[str, Any]], expected_cases: set[int]
) -> dict[str, Any]:
    endpoints: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        target_type = row["metadata"]["target_type"]
        if target_type == "endpoint_bundle":
            key = (int(row["metadata"]["case_id"]), str(row["metadata"]["path_cluster_id"]))
            if key in endpoints:
                raise ValueError(f"{key}: duplicate endpoint bundle")
            endpoints[key] = row
    if {case_id for case_id, _ in endpoints} != expected_cases:
        raise ValueError("not every case has an evidence-qualified endpoint")
    semantic_paths = {
        (int(row["metadata"]["case_id"]), str(row["metadata"]["path_cluster_id"]))
        for row in rows
        if row["metadata"]["target_type"] != "hypothesis_elimination"
    }
    if semantic_paths != set(endpoints):
        raise ValueError("a retained semantic path has no evidence-qualified endpoint")
    gate_rules: Counter[str] = Counter()
    recovery_count = 0
    role_closure_count = 0
    inclusive_or_endpoint_counts: Counter[int] = Counter()
    for key, row in endpoints.items():
        metadata = row["metadata"]
        verified_items = [str(value) for value in metadata["actual_result_items"]]
        expected_devices = {item.split(";", 1)[0].lower() for item in verified_items}
        gate = metadata["endpoint_evidence_gate"]
        gate_rules[gate["gate_rule"]] += 1
        if not gate["passed"] or metadata["derived_from_verified_final_answer"]:
            raise ValueError(f"{row['id']}: endpoint is label-derived or gate failed")
        if set(gate["covered_devices"]) != expected_devices:
            raise ValueError(f"{row['id']}: endpoint does not cover every answer device")
        facts = gate["selected_facts"]
        if not facts:
            raise ValueError(f"{row['id']}: endpoint has no direct facts")
        prior = normalized("\n".join(
            str(message["content"])
            for message in row["messages"]
            if message["role"] == "tool_response"
        ))
        target = normalized(current_supervised_text(row))
        nondecision_target = normalized("\n".join(
            str(message["content"])
            for message in row["messages"]
            if message["role"] == "assistant"
            and float(message.get("loss_scale", 0) or 0) > 0
            and "<result>" not in str(message["content"])
        ))
        if gate["gate_rule"] in nondecision_target:
            raise ValueError(f"{row['id']}: endpoint exposes an internal gate ID")
        for fact in facts:
            if fact["snapshot"] != gate["target_snapshot"]:
                raise ValueError(f"{row['id']}: decisive fact crosses snapshots")
            observation = normalized(str(fact["observation"]))
            if observation not in prior or observation not in target:
                raise ValueError(f"{row['id']}: decisive fact is not visible and quoted")
            if WINDOWS_PATH.search(str(fact["observation"])) or "for-perf" in observation:
                raise ValueError(f"{row['id']}: decisive fact contains environment/cross-snapshot data")
        if any(normalized(item) in nondecision_target for item in verified_items):
            raise ValueError(f"{row['id']}: summary/stop leaks the exact verified answer")
        rule = gate["gate_rule"]
        if rule == "three_device_same_prefix_static_route_next_hop_cycle":
            if converter.select_cycle_facts(facts, mpls=False) is None:
                raise ValueError(f"{key}: IP evidence does not form a cycle")
        if rule == "three_device_static_lsp_next_hop_and_label_cycle":
            if converter.select_cycle_facts(facts, mpls=True) is None:
                raise ValueError(f"{key}: MPLS evidence does not form a label cycle")
        if rule == "same_snapshot_source_vlan_vrrp_mst_instance_role_misalignment":
            validate_role_misalignment_closure(row, facts, expected_devices)
            role_closure_count += 1
        case_id = int(metadata["case_id"])
        if case_id in INCLUSIVE_OR_CASES:
            inclusive_or_endpoint_counts[case_id] += 1
            if (
                rule
                != "same_snapshot_source_vlan_vrrp_mst_instance_role_misalignment"
                or len(verified_items) != 1
                or metadata.get("inclusive_or_singleton_selected_by_evidence") is not True
            ):
                raise ValueError(
                    f"{row['id']}: q73-q86 endpoint is not the evidence-selected singleton"
                )
        recovery_count += any(
            str(fact["action_id"]).startswith("REC-") for fact in facts
        )
    if (
        set(inclusive_or_endpoint_counts) != INCLUSIVE_OR_CASES
        or set(inclusive_or_endpoint_counts.values()) != {1}
    ):
        raise ValueError("q73-q86 do not each have exactly one strict endpoint")
    q1 = [
        row["metadata"]["endpoint_evidence_gate"]
        for (case_id, _), row in endpoints.items() if case_id == 1
    ]
    if not q1 or any(
        gate["target_snapshot"] != "campusnetwork_02"
        or {fact["kind"] for fact in gate["selected_facts"]} != {"stp_global_disabled"}
        or {fact["device"] for fact in gate["selected_facts"]} != {"core_sw_01"}
        for gate in q1
    ):
        raise ValueError("q0001 endpoint is not grounded in target Core_SW_01 Disabled evidence")
    return {
        "qualified_paths": len(endpoints),
        "qualified_cases": len({case_id for case_id, _ in endpoints}),
        "gate_rules": dict(sorted(gate_rules.items())),
        "same_query_recovery_paths": recovery_count,
        "strict_role_closure_paths": role_closure_count,
        "inclusive_or_singleton_endpoint_cases": len(inclusive_or_endpoint_counts),
    }


def validate_schedules(
    rows_by_output: dict[str, list[dict[str, Any]]], train_cases: set[int]
) -> dict[str, int]:
    core_pool = rows_by_output["train_core"]
    endpoint_pool = rows_by_output["train_endpoint_pool"]
    core_pool_counts = Counter(int(row["metadata"]["case_id"]) for row in core_pool)
    single_core_replay_cases = {
        case_id for case_id, count in core_pool_counts.items() if count == 1
    }
    core_exposure: Counter[str] = Counter()
    endpoint_exposure: Counter[tuple[int, str]] = Counter()
    for epoch in range(1, 6):
        core = rows_by_output[f"train_core_epoch_{epoch:02d}"]
        endpoint = rows_by_output[f"train_endpoint_epoch_{epoch:02d}"]
        core_by_case = Counter(int(row["metadata"]["case_id"]) for row in core)
        endpoint_by_case = Counter(int(row["metadata"]["case_id"]) for row in endpoint)
        if set(core_by_case) != train_cases or set(core_by_case.values()) != {
            converter.CORE_EXPOSURES_PER_QUERY_PER_EPOCH
        }:
            raise ValueError(f"epoch {epoch}: core schedule is not case-balanced")
        if set(endpoint_by_case) != train_cases or set(endpoint_by_case.values()) != {
            converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH * len(converter.ENDPOINT_TARGET_TYPES)
        }:
            raise ValueError(f"epoch {epoch}: endpoint schedule is not case-balanced")
        for row in core:
            source_id = str(row["metadata"]["sampling_source_row_id"])
            core_exposure[source_id] += 1
        core_sources_by_case: dict[int, list[str]] = defaultdict(list)
        for row in core:
            core_sources_by_case[int(row["metadata"]["case_id"])].append(
                str(row["metadata"]["sampling_source_row_id"])
            )
        for case_id, values in core_sources_by_case.items():
            if len(values) == len(set(values)):
                continue
            if case_id not in single_core_replay_cases or len(set(values)) != 1:
                raise ValueError(
                    f"epoch {epoch}: duplicate core row without single-row pool fallback"
                )
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in endpoint:
            grouped[(int(row["metadata"]["case_id"]), int(row["metadata"]["endpoint_schedule_slot"]))].append(row)
        if len(grouped) != len(train_cases) * converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH:
            raise ValueError(f"epoch {epoch}: endpoint group count mismatch")
        for (case_id, _), group in grouped.items():
            if {row["metadata"]["target_type"] for row in group} != set(converter.ENDPOINT_TARGET_TYPES):
                raise ValueError(f"epoch {epoch}/q{case_id}: incomplete endpoint group")
            path_ids = {str(row["metadata"]["path_cluster_id"]) for row in group}
            if len(path_ids) != 1:
                raise ValueError(f"epoch {epoch}/q{case_id}: endpoint path mismatch")
            endpoint_exposure[(case_id, next(iter(path_ids)))] += 1
        total_by_case = Counter(core_by_case)
        total_by_case.update(endpoint_by_case)
        expected_per_query = (
            converter.CORE_EXPOSURES_PER_QUERY_PER_EPOCH
            + converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
            * len(converter.ENDPOINT_TARGET_TYPES)
        )
        if set(total_by_case.values()) != {expected_per_query}:
            raise ValueError(
                f"epoch {epoch}: total query exposure is not exactly {expected_per_query}"
            )
    core_pool_by_case: dict[int, list[str]] = defaultdict(list)
    for row in core_pool:
        core_pool_by_case[int(row["metadata"]["case_id"])].append(str(row["id"]))
    schedule_capacity = (
        converter.CORE_EXPOSURES_PER_QUERY_PER_EPOCH
        * converter.ENDPOINT_SCHEDULE_EPOCHS
    )
    for case_id, source_ids in core_pool_by_case.items():
        exposed = {source_id for source_id in source_ids if core_exposure[source_id]}
        if len(exposed) != min(len(source_ids), schedule_capacity):
            raise ValueError(f"q{case_id}: core schedule does not maximize unique coverage")
        counts = [core_exposure[source_id] for source_id in source_ids if core_exposure[source_id]]
        if counts and max(counts) - min(counts) > 1:
            raise ValueError(f"q{case_id}: core source exposure is not round-robin balanced")
    pool_paths = {
        (int(row["metadata"]["case_id"]), str(row["metadata"]["path_cluster_id"]))
        for row in endpoint_pool
    }
    if pool_paths - set(endpoint_exposure):
        raise ValueError("five endpoint schedules do not cover every qualified path")
    return {
        "core_rows_per_epoch": len(rows_by_output["train_core_epoch_01"]),
        "endpoint_rows_per_epoch": len(rows_by_output["train_endpoint_epoch_01"]),
        "total_rows_per_epoch": len(rows_by_output["train_core_epoch_01"])
        + len(rows_by_output["train_endpoint_epoch_01"]),
        "rows_per_query_per_epoch": converter.CORE_EXPOSURES_PER_QUERY_PER_EPOCH
        + converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
        * len(converter.ENDPOINT_TARGET_TYPES),
        "single_core_replay_case_count": len(single_core_replay_cases),
    }


def validate_training_contract(manifest: dict[str, Any]) -> None:
    profile = manifest["comparison_experiment_plan"]
    if (
        profile.get("distributed_strategy") != "ddp"
        or profile.get("world_size") != 2
        or profile.get("cuda_visible_devices") != "0,1"
        or profile["gradient_accumulation_steps"] != 4
        or profile["effective_batch_size"] != 8
    ):
        raise ValueError("manifest batch contract mismatch")
    if profile["fixed_learning_rate_by_epoch"] != [2e-5, 1.5e-5, 1e-5, 6e-6, 3e-6]:
        raise ValueError("manifest learning-rate contract mismatch")
    if profile["lr_scheduler_type"] != "constant" or profile["warmup_ratio"] != 0:
        raise ValueError("manifest fixed-LR scheduler contract mismatch")
    if (
        profile.get("train_rows_per_stage") != 216
        or profile.get("optimizer_steps_per_stage") != 27
        or profile.get("expected_global_step_boundaries") != [0, 27, 54, 81, 108, 135]
        or profile.get("checkpoint_suffix_by_stage") != [27, 54, 81, 108, 135]
    ):
        raise ValueError("manifest optimizer-step boundary contract mismatch")
    if (
        profile.get("checkpoint_selection_strategy") != "fixed_epoch"
        or profile.get("fixed_validation_epoch") != 3
        or profile.get("fixed_validation_checkpoint_suffix") != 81
        or profile.get("agent_checkpoint_selection") is not False
        or "checkpoint_agent_selection" in profile
    ):
        raise ValueError("manifest checkpoint selection differs from 0805 fixed epoch 3")
    script = (ROOT / "scripts" / "train_qwen36_0807_evidence_gated_5epoch.sh").read_text(encoding="utf-8")
    required = (
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"',
        'NPROC_PER_NODE="${NPROC_PER_NODE:-2}"',
        "WORLD_SIZE=2", "GRADIENT_ACCUMULATION_STEPS=4",
        "EFFECTIVE_BATCH_SIZE=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))",
        "--lr_scheduler_type constant",
        "--warmup_ratio 0", "--resume_from_checkpoint", "--resume_only_model false",
        "qwen3_6_27b_0807_core_epoch_", "qwen3_6_27b_0807_endpoint_epoch_",
        "LEARNING_RATES=(2e-5 1.5e-5 1e-5 6e-6 3e-6)",
        "--callbacks forced_epoch_lr",
        "EXPECTED_TRAIN_ROWS_PER_STAGE=216",
        "EXPECTED_OPTIMIZER_STEPS_PER_STAGE=27",
        "FIXED_VALIDATION_EPOCH=3",
        "FIXED_VALIDATION_CHECKPOINT_SUFFIX=81",
        "step_globals != list(range(expected_start, expected_end))",
        'resume_checkpoint="${epoch_dir}/checkpoint-${expected_end_step}"',
        "Fixed Agent validation checkpoint:",
    )
    if any(value not in script for value in required):
        raise ValueError("training entry does not implement the manifest contract")
    callback = (ROOT / "scripts" / "qwen36_0807_epoch_lr_callback.py").read_text(
        encoding="utf-8"
    )
    if 'PROCESS_RANK = int(os.environ.get("RANK", "0"))' not in callback or (
        "if PROCESS_RANK != 0:" not in callback
    ):
        raise ValueError("DDP learning-rate audit is not rank-0 write exclusive")


def main() -> None:
    if independently_infer_families(
        "cat saved_configs/CampusNetwork_01/PE1/display_lldp_neighbor_brief.txt"
    ) != {"lldp"}:
        raise ValueError("LLDP negative regression was classified as MPLS/LDP")
    if "mpls" not in independently_infer_families(
        "cat saved_configs/CampusNetwork_01/PE1/display_mpls_ldp_lsp.txt"
    ):
        raise ValueError("standalone MPLS/LDP positive regression was missed")
    train_cases, validation_cases = validate_source()
    source = load_json(CURATION)
    manifest, rows_by_output = validate_manifest_and_files()
    train = rows_by_output["train"]
    validation = rows_by_output["validation"]
    if {int(row["metadata"]["case_id"]) for row in train} != train_cases:
        raise ValueError("train semantic pool case inventory mismatch")
    if {int(row["metadata"]["case_id"]) for row in validation} != validation_cases:
        raise ValueError("validation case inventory mismatch")
    all_semantic = [*train, *validation]
    protocol = validate_tool_and_loss_protocol(all_semantic)
    grounding = validate_visible_reasoning_grounding(all_semantic)
    elimination = validate_elimination_grounding(all_semantic, source)
    endpoint = validate_endpoint_gate(all_semantic, VALID_CASES)
    schedules = validate_schedules(rows_by_output, train_cases)
    validate_training_contract(manifest)
    max_auto = max(
        audit["auto_summary_stop_weighted_percent"]
        for audit in manifest["heuristic_training_signal_by_epoch"].values()
    )
    if max_auto >= 25:
        raise ValueError("automatic summary/stop weighted loss reaches the 25% cap")
    print("0807 evidence-gated SFT validation passed")
    print("source=840 trajectories / 84 cases / 10 each; split=72 train + 12 validation")
    print(
        f"semantic rows: train={len(train)}, validation={len(validation)}; "
        f"qualified endpoints={endpoint['qualified_paths']} paths/{endpoint['qualified_cases']} cases"
    )
    print(
        "q73-q86 inclusive-OR archive=140 synchronized singleton trajectories; "
        f"strict singleton endpoints={endpoint['inclusive_or_singleton_endpoint_cases']}; "
        f"all strict role closures={endpoint['strict_role_closure_paths']}"
    )
    print(
        f"actions={protocol['actions']}; config={protocol['config_actions']} "
        f"({100 * protocol['config_actions'] / protocol['actions']:.2f}%); "
        f"LLDP={protocol['lldp_actions']}; cross-snapshot=0"
    )
    print(
        f"observation-bound factual sentences="
        f"{grounding['observation_bound_factual_sentences']}; "
        f"procedural={grounding['procedural_source_sentences']}; "
        f"dropped unsupported={grounding['dropped_unsupported_sentences']}"
    )
    print(
        f"source intent coverage={protocol['source_intent_coverage']}; "
        f"supervised intent coverage={protocol['supervised_intent_coverage']}"
    )
    print(
        f"grounded eliminations={elimination['nodes']}; "
        f"auxiliary={elimination['auxiliary_non_endpoint_nodes']}"
    )
    print(
        f"per epoch={schedules['total_rows_per_epoch']} rows; "
        f"{schedules['rows_per_query_per_epoch']} per query; "
        f"single-core replay cases={schedules['single_core_replay_case_count']}; "
        f"max auto summary+stop loss={max_auto:.2f}%"
    )


if __name__ == "__main__":
    main()
