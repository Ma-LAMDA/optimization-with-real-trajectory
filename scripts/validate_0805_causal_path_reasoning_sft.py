#!/usr/bin/env python3
"""Validate the 0805 causal-path-clustered weighted multi-turn SFT data."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import convert_0804_best_trajectory_reasoning_sft as base
import convert_0805_causal_path_reasoning_sft as converter


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-05"
SOURCE_CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
FROZEN_0804_CURATION = (
    ROOT / "data" / "2026-08-04" / "curation" / "accepted_trajectory_selection.json"
)
SELECTION = DATA_ROOT / "curation" / "causal_path_clusters_per_case.json"
MANIFEST = DATA_ROOT / "sft" / "reasoning_causal_path_manifest.json"
TRAIN = DATA_ROOT / "sft" / "qwen3_6_27b_reasoning_causal_path_train.jsonl"
TRAIN_CORE = DATA_ROOT / "sft" / "qwen3_6_27b_reasoning_causal_path_train_core.jsonl"
TRAIN_ENDPOINT_POOL = (
    DATA_ROOT / "sft" / "qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl"
)
VALIDATION = DATA_ROOT / "sft" / "qwen3_6_27b_reasoning_causal_path_validation.jsonl"


def train_endpoint_epoch_path(epoch: int) -> Path:
    return DATA_ROOT / "sft" / f"qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_{epoch:02d}.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}: expected JSON objects")
    return rows


def digest_output(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_formal_training_contract(manifest: dict[str, Any]) -> None:
    config = load_json(converter.FORMAL_TRAINING_CONFIG_PATH)
    expected_critical = {
        "schema_version": "qwen36-0805-formal-training.v2",
        "distributed_strategy": "ddp",
        "world_size": 2,
        "cuda_visible_devices": "0,1",
        "epochs": 5,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "max_length": 16384,
        "fixed_learning_rate_by_epoch": [2e-5, 1.5e-5, 1e-5, 6e-6, 3e-6],
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 5,
        "load_best_model_at_end": False,
        "resume_only_model": False,
        "epoch_specific_endpoint_schedule_required": True,
        "sequential_full_state_resume_required": True,
        "learning_rate_runtime_audit_required": True,
        "checkpoint_selection_strategy": "fixed_epoch",
        "fixed_validation_epoch": 3,
        "agent_checkpoint_selection": False,
    }
    for key, expected in expected_critical.items():
        if config.get(key) != expected:
            raise ValueError(
                f"formal training config {key}={config.get(key)!r}, expected {expected!r}"
            )
    calculated_effective_batch = (
        config["world_size"]
        * config["per_device_train_batch_size"]
        * config["gradient_accumulation_steps"]
    )
    if calculated_effective_batch != config["effective_batch_size"]:
        raise ValueError(
            "formal training effective-batch arithmetic is inconsistent: "
            f"{calculated_effective_batch} != {config['effective_batch_size']}"
        )
    if not converter.FORMAL_TRAINING_REFERENCE.is_file():
        raise ValueError("formal 0804 reference training entry is missing")

    profile = manifest["training_profile"]
    expected_profile_values = {
        "distributed_strategy": "ddp",
        "world_size": 2,
        "cuda_visible_devices": "0,1",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
    }
    for key, expected in expected_profile_values.items():
        if profile.get(key) != expected:
            raise ValueError(f"manifest training profile has stale {key}")
    expected_profile_paths = {
        "formal_training_config": converter.FORMAL_TRAINING_CONFIG_PATH,
        "formal_training_entry": converter.FORMAL_TRAINING_ENTRY,
        "fixed_stage_lr_plugin": converter.FIXED_STAGE_LR_PLUGIN,
        "quick_smoke_entry": converter.QUICK_SMOKE_ENTRY,
    }
    for key, path in expected_profile_paths.items():
        if profile.get(key) != path.relative_to(ROOT).as_posix():
            raise ValueError(f"manifest training profile has stale {key}")
    if "independent one-epoch" not in profile.get("quick_smoke_scope", ""):
        raise ValueError("quick entry is not explicitly isolated from formal training")

    plan = manifest["comparison_experiment_plan"]
    for key, expected in config.items():
        if plan.get(key) != expected:
            raise ValueError(f"manifest formal plan differs from config at {key}")
    expected_plan_paths = {
        "config_path": converter.FORMAL_TRAINING_CONFIG_PATH,
        "entry_path": converter.FORMAL_TRAINING_ENTRY,
        "lr_audit_plugin_path": converter.FIXED_STAGE_LR_PLUGIN,
    }
    for key, path in expected_plan_paths.items():
        if plan.get(key) != path.relative_to(ROOT).as_posix():
            raise ValueError(f"manifest formal plan has stale {key}")
    expected_plan_hashes = {
        "config_sha256_lf_normalized": converter.FORMAL_TRAINING_CONFIG_PATH,
        "entry_sha256_lf_normalized": converter.FORMAL_TRAINING_ENTRY,
        "lr_audit_plugin_sha256_lf_normalized": converter.FIXED_STAGE_LR_PLUGIN,
    }
    for key, path in expected_plan_hashes.items():
        if plan.get(key) != base.digest_file(path):
            raise ValueError(f"manifest formal plan has stale {key}")

    entry = converter.FORMAL_TRAINING_ENTRY.read_text(encoding="utf-8")
    required_entry_snippets = (
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"',
        'NPROC_PER_NODE="${NPROC_PER_NODE:-2}"',
        "export NPROC_PER_NODE",
        "for ((stage = start_stage; stage <= 5; stage++))",
        '--resume_from_checkpoint "${existing_checkpoint}"',
        '--resume_only_model "${CFG[resume_only_model]}"',
        '--num_train_epochs "${stage}"',
        '--gradient_accumulation_steps "${CFG[gradient_accumulation_steps]}"',
        '--learning_rate "${target_lr}"',
        '--lr_scheduler_type "${CFG[lr_scheduler_type]}"',
        '--warmup_ratio "${CFG[warmup_ratio]}"',
        '--max-length "${FORMAL_MAX_LENGTH}"',
        '--eval_strategy "${CFG[eval_strategy]}"',
        '--save_strategy "${CFG[save_strategy]}"',
        "--callbacks qwen36_0805_fixed_stage_lr",
        'endpoint_epoch_${stage_tag}.jsonl',
        'model_files_sha256.txt',
        'sha256sum --check',
        'find "${MODEL_PATH}" -type f -print0',
        'capture_training_environment',
        'current_environment="$(capture_training_environment)"',
    )
    missing_entry = [item for item in required_entry_snippets if item not in entry]
    if missing_entry:
        raise ValueError(f"formal training entry lacks required controls: {missing_entry}")
    forbidden_entry_snippets = (
        'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
        "--lr_scheduler_type cosine",
        "--gradient_accumulation_steps 2",
        "--warmup_ratio 0.1",
        "--early_stop_interval",
    )
    leaked_entry = [item for item in forbidden_entry_snippets if item in entry]
    if leaked_entry:
        raise ValueError(f"quick-only settings leaked into formal training: {leaked_entry}")

    plugin = converter.FIXED_STAGE_LR_PLUGIN.read_text(encoding="utf-8")
    required_plugin_snippets = (
        'QWEN36_0805_TRAIN_STAGE',
        'QWEN36_0805_TARGET_LR',
        'QWEN36_0805_LR_AUDIT_PATH',
        'PROCESS_RANK = int(os.environ.get("RANK", "0"))',
        'if PROCESS_RANK != 0:',
        'def _assert_target',
        'def on_train_begin',
        'def on_epoch_begin',
        'def on_step_begin',
        'callbacks_map["qwen36_0805_fixed_stage_lr"]',
    )
    missing_plugin = [item for item in required_plugin_snippets if item not in plugin]
    if missing_plugin:
        raise ValueError(f"formal LR plugin lacks required enforcement: {missing_plugin}")


def independent_training_signal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_counts: Counter[str] = Counter()
    raw_tokens_by_target: Counter[str] = Counter()
    weighted_tokens_by_target: Counter[str] = Counter()
    weighted_tokens_by_message_kind: Counter[str] = Counter()
    for row in rows:
        target_type = str(row["metadata"]["target_type"])
        row_counts[target_type] += 1
        for message in row["messages"]:
            scale = float(message.get("loss_scale", 0) or 0)
            if scale <= 0:
                continue
            token_count = converter.estimate_context_token_count(
                str(message["content"])
            )
            raw_tokens_by_target[target_type] += token_count
            weighted_tokens_by_target[target_type] += token_count * scale
            if message["role"] == "tool_call":
                kind = "tool_call"
            elif message["role"] == "assistant" and "<think>" in message["content"]:
                kind = "thinking"
            else:
                kind = "conclusion_or_answer"
            weighted_tokens_by_message_kind[kind] += token_count * scale
    raw_total = sum(raw_tokens_by_target.values())
    weighted_total = sum(weighted_tokens_by_target.values())
    auto_endpoint_weight = sum(
        weighted_tokens_by_target[target_type]
        for target_type in ("evidence_summary", "decision_ready")
    )
    return {
        "method": "one CJK code point is approximately one token; other text is approximately four characters per token",
        "row_count": len(rows),
        "target_type_rows": dict(sorted(row_counts.items())),
        "raw_target_token_estimate": raw_total,
        "weighted_token_estimate": round(weighted_total, 6),
        "weighted_message_kind_percent": {
            key: round(100.0 * value / weighted_total, 6)
            for key, value in sorted(weighted_tokens_by_message_kind.items())
        },
        "weighted_target_type_percent": {
            key: round(100.0 * value / weighted_total, 6)
            for key, value in sorted(weighted_tokens_by_target.items())
        },
        "auto_summary_stop_weighted_percent": round(
            100.0 * auto_endpoint_weight / weighted_total, 6
        ),
    }


def source_command_inventory(path: Path) -> set[str]:
    _, commands = base.parse_events(path)
    return {base.normalize_text(str(command["command"])) for command in commands}


def validate_loss_and_sequence(row: dict[str, Any]) -> None:
    metadata = row["metadata"]
    messages = row["messages"]
    if len(messages) < 4 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"{row['id']}: malformed initial messages")
    if messages[0]["content"] != converter.CODEX_CLI_SYSTEM_PROMPT:
        raise ValueError(f"{row['id']}: system prompt differs from Codex CLI model metadata")
    if metadata["system_prompt_source"] != converter.CODEX_MODEL_CATALOG.relative_to(ROOT).as_posix():
        raise ValueError(f"{row['id']}: system prompt source mismatch")
    if metadata["system_prompt_model_slug"] != converter.CODEX_CLI_MODEL_SLUG:
        raise ValueError(f"{row['id']}: system prompt model slug mismatch")
    if metadata["system_prompt_sha256"] != converter.CODEX_CLI_SYSTEM_PROMPT_SHA256:
        raise ValueError(f"{row['id']}: system prompt hash mismatch")
    expected_tools = json.dumps(
        converter.CODEX_CLI_TOOLS, ensure_ascii=False, separators=(",", ":")
    )
    if row["tools"] != expected_tools:
        raise ValueError(f"{row['id']}: tool schema differs from Codex CLI exec_command")
    if metadata.get("tool_protocol") != {
        "name": "exec_command",
        "command_argument": "cmd",
        "operating_system": "linux",
        "path_style": "repository_relative_saved_configs",
        "source_command_audit": "exact PowerShell retained in current_actions.source_command",
        "tool_response_transport": "Codex CLI shape with Linux-normalized discovery/search output; token count is heuristic and context-only",
    }:
        raise ValueError(f"{row['id']}: tool protocol metadata mismatch")

    assistant = [message for message in messages if message["role"] == "assistant"]
    if len(assistant) % 2 or len(assistant) < 2:
        raise ValueError(f"{row['id']}: assistant messages are not thinking/conclusion pairs")
    thinking_source = str(metadata["thinking_source"])
    expected_thinking_scale = converter.THINKING_LOSS_SCALE_BY_SOURCE.get(thinking_source)
    if expected_thinking_scale is None:
        raise ValueError(f"{row['id']}: unknown thinking source {thinking_source}")
    if metadata["loss_policy"]["thinking"] != expected_thinking_scale:
        raise ValueError(f"{row['id']}: thinking loss policy mismatch")
    conclusion_source = str(metadata["conclusion_source"])
    expected_conclusion_scale = converter.CONCLUSION_LOSS_SCALE_BY_SOURCE.get(
        conclusion_source
    )
    if expected_conclusion_scale is None:
        raise ValueError(f"{row['id']}: unknown conclusion source {conclusion_source}")
    if metadata["loss_policy"]["conclusion_or_result"] != expected_conclusion_scale:
        raise ValueError(f"{row['id']}: conclusion loss policy mismatch")
    expected_scales = [base.HISTORY_LOSS_SCALE] * (len(assistant) - 2) + [
        expected_thinking_scale,
        expected_conclusion_scale,
    ]
    actual_scales = [message.get("loss_scale") for message in assistant]
    if actual_scales != expected_scales:
        raise ValueError(f"{row['id']}: assistant loss scales {actual_scales} != {expected_scales}")
    if not assistant[-2]["content"].startswith("<think>\n") or "</think>" not in assistant[-2]["content"]:
        raise ValueError(f"{row['id']}: current thinking wrapper is missing")

    tool_calls = [message for message in messages if message["role"] == "tool_call"]
    current_action_count = int(metadata["current_action_count"])
    if current_action_count > len(tool_calls):
        raise ValueError(f"{row['id']}: current action count exceeds tool calls")
    expected_tool_scales = [base.HISTORY_LOSS_SCALE] * (
        len(tool_calls) - current_action_count
    ) + [converter.TOOL_CALL_LOSS_SCALE] * current_action_count
    actual_tool_scales = [message.get("loss_scale") for message in tool_calls]
    if actual_tool_scales != expected_tool_scales:
        raise ValueError(f"{row['id']}: tool-call loss scales are invalid")
    for message in tool_calls:
        payload = json.loads(message["content"])
        if payload.get("name") != "exec_command":
            raise ValueError(f"{row['id']}: non-Codex tool name in supervised messages")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != {"cmd"}:
            raise ValueError(f"{row['id']}: exec_command arguments are malformed")
        command = str(arguments["cmd"])
        if (
            "powershell" in command.lower()
            or re.search(r"\b[A-Za-z]:[\\/]", command)
            or "saved_configs\\" in command
        ):
            raise ValueError(f"{row['id']}: Windows command syntax leaked into supervision")

    response_prefix = re.compile(
        r"^Chunk ID: [0-9a-f]{6}\n"
        r"Wall time: 0\.0000 seconds\n"
        r"Process exited with code -?\d+\n"
        r"Original token count: (?P<count>\d+)\n"
        r"Output:\n",
    )
    for message in messages:
        if message["role"] != "tool_response":
            continue
        match = response_prefix.match(message["content"])
        if not match:
            raise ValueError(f"{row['id']}: tool response is not in Codex CLI transport shape")
        output = message["content"][match.end() :]
        if int(match.group("count")) != converter.estimate_context_token_count(output):
            raise ValueError(f"{row['id']}: context-only response token estimate mismatch")
        if (
            converter.WINDOWS_SAVED_CONFIGS_ROOT_PATTERN.search(output)
            or re.search(r"^(?:目录|Directory)\s*:", output, re.MULTILINE | re.IGNORECASE)
        ):
            raise ValueError(f"{row['id']}: PowerShell result presentation leaked into context")

    for message in messages:
        if message["role"] in {"system", "user", "tool_response"} and (
            "loss_scale" in message or "loss" in message
        ):
            raise ValueError(f"{row['id']}: context-only role is marked trainable")
        if "loss" in message:
            raise ValueError(f"{row['id']}: binary loss field is not allowed")

    last_target_assistant = max(
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    )
    if any(message["role"] == "tool_response" for message in messages[last_target_assistant + 1 :]):
        raise ValueError(f"{row['id']}: future tool result leaked into current target")
    no_tool_targets = {"decision", "decision_ready", "evidence_summary"}
    if metadata["target_type"] in no_tool_targets and messages[-1]["role"] != "assistant":
        raise ValueError(f"{row['id']}: no-tool target must end in an assistant answer")
    if current_action_count and messages[-1]["role"] != "tool_call":
        raise ValueError(f"{row['id']}: action target must end with actual tool calls")
    if metadata["target_type"] in no_tool_targets and current_action_count:
        raise ValueError(f"{row['id']}: no-tool target must not have an action")
    if metadata["target_type"] not in no_tool_targets and not current_action_count:
        raise ValueError(f"{row['id']}: actionable node has no actual action")


def main() -> None:
    source = load_json(SOURCE_CURATION)
    frozen = load_json(FROZEN_0804_CURATION)
    selection = load_json(SELECTION)
    manifest = load_json(MANIFEST)
    if manifest["schema_version"] != "qwen36-0805-causal-path-reasoning-sft.v10":
        raise ValueError("manifest schema version is not the formal-training-contract revision")
    if selection["schema_version"] != "0805-causal-path-cluster-selection.v4":
        raise ValueError("selection schema version is not the elimination/endpoint-group revision")
    expected_reproducibility = {
        "document": converter.REPRODUCIBILITY_DOC.relative_to(ROOT).as_posix(),
        "document_sha256_lf_normalized": base.digest_file(
            converter.REPRODUCIBILITY_DOC
        ),
        "change_policy": "every source, curation, split, conversion, loss, sampling, prompt, tool protocol, tokenizer, training-entry, validator, or generated-data change must update documentation, regenerate all artifacts, and pass independent validation",
        "tracked_files": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256_lf_normalized": base.digest_file(path),
            }
            for name, path in converter.REPRODUCIBILITY_FILES.items()
        },
    }
    if manifest.get("reproducibility") != expected_reproducibility:
        raise ValueError("manifest reproducibility document or tracked-file hashes are stale")
    validate_formal_training_contract(manifest)
    if manifest["conversion"]["max_actions_per_investigative_stage"] != converter.MAX_ACTIONS_PER_STAGE:
        raise ValueError("manifest action cap differs from converter policy")
    expected_system_prompt = {
        "role": "system",
        "source": converter.CODEX_MODEL_CATALOG.relative_to(ROOT).as_posix(),
        "source_sha256_lf_normalized": base.digest_file(converter.CODEX_MODEL_CATALOG),
        "model_slug": converter.CODEX_CLI_MODEL_SLUG,
        "field": "base_instructions",
        "content_sha256": converter.CODEX_CLI_SYSTEM_PROMPT_SHA256,
        "content": converter.CODEX_CLI_SYSTEM_PROMPT,
    }
    if manifest["system_prompt"] != expected_system_prompt:
        raise ValueError("manifest Codex CLI system prompt metadata mismatch")
    if manifest["loss_policy"]["thinking_by_source"] != converter.THINKING_LOSS_SCALE_BY_SOURCE:
        raise ValueError("manifest thinking loss policy mismatch")
    if manifest["loss_policy"]["conclusion_by_source"] != converter.CONCLUSION_LOSS_SCALE_BY_SOURCE:
        raise ValueError("manifest conclusion loss policy mismatch")
    if manifest["conversion"]["endpoint_group_exposures_per_query_per_epoch"] != converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH:
        raise ValueError("manifest endpoint-group exposure count mismatch")
    if manifest["conversion"]["endpoint_schedule_epochs"] != converter.ENDPOINT_SCHEDULE_EPOCHS:
        raise ValueError("manifest endpoint schedule epoch count mismatch")
    train_rows = load_jsonl(TRAIN)
    train_core_rows = load_jsonl(TRAIN_CORE)
    train_endpoint_pool_rows = load_jsonl(TRAIN_ENDPOINT_POOL)
    endpoint_schedules = {
        epoch: load_jsonl(train_endpoint_epoch_path(epoch))
        for epoch in range(1, converter.ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    validation_rows = load_jsonl(VALIDATION)
    obsolete_outputs = [
        path.relative_to(ROOT).as_posix()
        for path in converter.OBSOLETE_DECISION_OUTPUTS
        if path.exists()
    ]
    if obsolete_outputs:
        raise ValueError(f"obsolete decision-only schedules still exist: {obsolete_outputs}")

    if selection["source_curation_sha256_lf_normalized"] != base.digest_file(SOURCE_CURATION):
        raise ValueError("selection source curation hash mismatch")
    if selection["frozen_0804_curation_sha256_lf_normalized"] != base.digest_file(FROZEN_0804_CURATION):
        raise ValueError("selection frozen 0804 hash mismatch")
    if manifest["source_curation_sha256_lf_normalized"] != base.digest_file(SOURCE_CURATION):
        raise ValueError("manifest source curation hash mismatch")
    if manifest["cluster_selection_sha256_lf_normalized"] != base.digest_file(SELECTION):
        raise ValueError("manifest cluster selection hash mismatch")

    source_train = set(source["split"]["train_case_ids"])
    source_validation = set(source["split"]["validation_case_ids"])
    frozen_train = set(frozen["split"]["train_case_ids"])
    frozen_validation = set(frozen["split"]["validation_case_ids"])
    if (source_train, source_validation) != (frozen_train, frozen_validation):
        raise ValueError("0805 split differs from frozen 0804 split")
    if source_train & source_validation:
        raise ValueError("source train/validation cases overlap")
    if sorted(source_validation) != manifest["split"]["validation_case_ids"]:
        raise ValueError("manifest validation IDs differ from source")

    annotations = [item for item in source["trajectories"] if item.get("selected")]
    if len(annotations) != 840:
        raise ValueError(f"expected 840 selected source trajectories, found {len(annotations)}")
    annotations_by_id = {str(item["id"]): item for item in annotations}
    by_case: dict[int, set[str]] = defaultdict(set)
    raw_by_trajectory: dict[str, dict[str, Any]] = {}
    messages_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    stages_by_trajectory: dict[str, list[dict[str, Any]]] = {}
    commands_by_trajectory: dict[str, set[str]] = {}
    supervised_commands_by_trajectory: dict[str, dict[str, str]] = {}
    for annotation in annotations:
        trajectory_id = str(annotation["id"])
        case_id = int(annotation["case_id"])
        by_case[case_id].add(trajectory_id)
        raw_path = ROOT / annotation["raw_file"]
        event_path = ROOT / annotation["events_file"]
        raw = load_json(raw_path)
        if not raw.get("answer_matches_reference") or not raw.get("independent_judgment", {}).get("correct"):
            raise ValueError(f"{trajectory_id}: source trajectory is not strictly correct")
        if base.digest_file(raw_path) != annotation["raw_sha256_lf_normalized"]:
            raise ValueError(f"{trajectory_id}: raw hash mismatch")
        if base.digest_file(event_path) != annotation["events_sha256_lf_normalized"]:
            raise ValueError(f"{trajectory_id}: event hash mismatch")
        raw_by_trajectory[trajectory_id] = raw
        source_messages, source_commands = base.parse_events(event_path)
        messages_by_trajectory[trajectory_id] = source_messages
        prepared_stages, _ = converter.prepare_stages(
            raw, source_messages, source_commands
        )
        stages_by_trajectory[trajectory_id] = prepared_stages
        commands_by_trajectory[trajectory_id] = {
            base.normalize_text(str(command["command"]))
            for command in source_commands
        }
        translated_commands: dict[str, str] = {}
        for command in commands_by_trajectory[trajectory_id]:
            inner = converter.normalize_supervised_command(command)
            try:
                translation = converter.translate_powershell_to_exec_command(inner)
            except ValueError:
                continue
            translated_commands[translation["cmd"]] = translation["justification"]
        supervised_commands_by_trajectory[trajectory_id] = translated_commands
    if set(by_case) != source_train | source_validation:
        raise ValueError("source case inventory differs from split")
    if any(len(trajectory_ids) != 10 for trajectory_ids in by_case.values()):
        raise ValueError("not every source case has exactly 10 trajectories")

    selection_cases = selection["cases"]
    if len(selection_cases) != 84 or {int(case["case_id"]) for case in selection_cases} != set(by_case):
        raise ValueError("selection does not cover all 84 cases")
    selected_row_ids: set[str] = set()
    selected_cluster_by_case: dict[int, set[str]] = defaultdict(set)
    for case in selection_cases:
        case_id = int(case["case_id"])
        if int(case["source_trajectory_count"]) != 10:
            raise ValueError(f"q{case_id}: selection source count is not 10")
        clusters = case["clusters"]
        cluster_members = {
            member
            for cluster in clusters
            for member in cluster["member_trajectory_ids"]
        }
        if cluster_members != by_case[case_id]:
            raise ValueError(f"q{case_id}: path clusters do not partition the source trajectories")
        if sum(int(cluster["member_count"]) for cluster in clusters) != 10:
            raise ValueError(f"q{case_id}: cluster member counts do not sum to 10")
        retained = {cluster["cluster_id"] for cluster in clusters if cluster["retained"]}
        if not 1 <= len(retained) <= converter.MAX_RETAINED_PATHS_PER_CASE:
            raise ValueError(f"q{case_id}: invalid retained path count")
        if len(retained) != int(case["retained_path_cluster_count"]):
            raise ValueError(f"q{case_id}: retained path count mismatch")
        nodes = case["selected_nodes"]
        if len(nodes) != int(case["selected_sft_node_count"]):
            raise ValueError(f"q{case_id}: selected node count mismatch")
        if int(case["selected_sft_node_count"]) >= int(case["raw_visible_checkpoint_count"]):
            raise ValueError(f"q{case_id}: clustering did not reduce checkpoint count")
        decisions = [node for node in nodes if node["target_type"] == "decision"]
        eliminations = [
            node for node in nodes
            if node["target_type"] == "hypothesis_elimination"
        ]
        if len(decisions) != len(retained):
            raise ValueError(f"q{case_id}: decisions do not cover every retained path")
        for cluster_id in retained:
            summaries = [
                node
                for node in nodes
                if node["path_cluster_id"] == cluster_id
                and node["target_type"] == "evidence_summary"
            ]
            stops = [
                node
                for node in nodes
                if node["path_cluster_id"] == cluster_id
                and node["target_type"] == "decision_ready"
            ]
            if len(summaries) != 1 or len(stops) != 1:
                raise ValueError(
                    f"q{case_id}/{cluster_id}: retained path lacks exactly one summary and stop"
                )
            if not summaries[0].get("synthetic_stage") or not stops[0].get("synthetic_stage"):
                raise ValueError(f"q{case_id}/{cluster_id}: endpoint node is not marked synthetic")
            path_decisions = [
                node
                for node in decisions
                if node["path_cluster_id"] == cluster_id
            ]
            if len(path_decisions) != 1:
                raise ValueError(f"q{case_id}/{cluster_id}: path lacks exactly one decision")
            path_eliminations = [
                node
                for node in eliminations
                if node["path_cluster_id"] == cluster_id
            ]
            if len(path_eliminations) > 1:
                raise ValueError(
                    f"q{case_id}/{cluster_id}: more than one elimination node was mined"
                )
            if any(node.get("synthetic_stage") for node in path_eliminations):
                raise ValueError(
                    f"q{case_id}/{cluster_id}: source elimination was marked synthetic"
                )
        if int(case["evidence_summary_node_count"]) != len(retained):
            raise ValueError(f"q{case_id}: evidence-summary count differs from retained paths")
        if int(case["decision_ready_node_count"]) != len(retained):
            raise ValueError(f"q{case_id}: stop count differs from retained paths")
        if int(case["decision_node_count"]) != len(retained):
            raise ValueError(f"q{case_id}: decision count differs from retained paths")
        if int(case["hypothesis_elimination_node_count"]) != len(eliminations):
            raise ValueError(f"q{case_id}: hypothesis-elimination count mismatch")
        if int(case["synthetic_path_endpoint_node_count"]) != 2 * len(retained):
            raise ValueError(f"q{case_id}: synthetic endpoint count mismatch")
        node_clusters = {node["path_cluster_id"] for node in nodes}
        if retained - node_clusters:
            raise ValueError(f"q{case_id}: a retained path has no representative node")
        for node in nodes:
            row_id = str(node["row_id"])
            if row_id in selected_row_ids:
                raise ValueError(f"duplicate selected row ID: {row_id}")
            selected_row_ids.add(row_id)
            selected_cluster_by_case[case_id].add(str(node["path_cluster_id"]))

    rows_by_split = {"train": train_rows, "validation": validation_rows}
    expected_outputs = {
        "train": (TRAIN, train_rows),
        "train_core": (TRAIN_CORE, train_core_rows),
        "train_endpoint_pool": (TRAIN_ENDPOINT_POOL, train_endpoint_pool_rows),
        **{
            f"train_endpoint_epoch_{epoch:02d}": (
                train_endpoint_epoch_path(epoch),
                rows,
            )
            for epoch, rows in endpoint_schedules.items()
        },
        "validation": (VALIDATION, validation_rows),
    }
    for output_name, (path, output_rows) in expected_outputs.items():
        expected = manifest["outputs"][output_name]
        if expected["rows"] != len(output_rows):
            raise ValueError(f"{output_name}: row count mismatch")
        if expected["bytes"] != path.stat().st_size:
            raise ValueError(f"{output_name}: byte count mismatch")
        if expected["sha256_lf_normalized"] != digest_output(path):
            raise ValueError(f"{output_name}: output hash mismatch")

    all_rows = train_rows + validation_rows
    if {str(row["id"]) for row in all_rows} != selected_row_ids:
        raise ValueError("SFT row IDs differ from the cluster selection")
    if len(all_rows) != len(selected_row_ids):
        raise ValueError("SFT row IDs are not unique")

    rows_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    split_by_case: dict[int, set[str]] = defaultdict(set)
    target_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    for expected_split, rows in rows_by_split.items():
        for row in rows:
            metadata = row["metadata"]
            case_id = int(metadata["case_id"])
            trajectory_id = str(metadata["trajectory_id"])
            if metadata["split"] != expected_split:
                raise ValueError(f"{row['id']}: split metadata mismatch")
            expected_cases = source_train if expected_split == "train" else source_validation
            if case_id not in expected_cases:
                raise ValueError(f"{row['id']}: case is in the wrong split")
            if trajectory_id not in by_case[case_id]:
                raise ValueError(f"{row['id']}: trajectory is not a source member of the case")
            annotation = annotations_by_id[trajectory_id]
            if metadata["source_event_sha256_lf_normalized"] != annotation["events_sha256_lf_normalized"]:
                raise ValueError(f"{row['id']}: event provenance hash mismatch")
            if metadata["path_cluster_id"] not in selected_cluster_by_case[case_id]:
                raise ValueError(f"{row['id']}: path cluster is not retained")
            if metadata.get("thinking_is_original_hidden_chain_of_thought") is not False:
                raise ValueError(f"{row['id']}: hidden thinking provenance is misrepresented")
            if int(metadata["current_action_count"]) > converter.MAX_ACTIONS_PER_STAGE:
                raise ValueError(f"{row['id']}: too many current actions")
            if metadata["loss_policy"]["tool_calls"] != converter.TOOL_CALL_LOSS_SCALE:
                raise ValueError(f"{row['id']}: tool-call loss policy mismatch")
            validate_loss_and_sequence(row)
            if metadata.get("training_sampling_role") != "primary":
                raise ValueError(f"{row['id']}: semantic dataset row has a schedule role")
            if metadata.get("sampling_source_row_id") is not None:
                raise ValueError(f"{row['id']}: primary row claims a sampling source")

            target_type = str(metadata["target_type"])
            expected_exposures = (
                converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
                if target_type in converter.ENDPOINT_TARGET_TYPES
                else 0
            )
            if metadata.get("endpoint_group_exposures_per_query_per_epoch") != expected_exposures:
                raise ValueError(f"{row['id']}: endpoint exposure metadata mismatch")
            if metadata.get("endpoint_schedule_epoch") is not None or metadata.get("endpoint_schedule_slot") is not None:
                raise ValueError(f"{row['id']}: semantic row claims schedule placement")
            synthetic = bool(metadata.get("synthetic_stage"))
            endpoint_policy = metadata.get("path_endpoint_policy")
            if (
                not isinstance(endpoint_policy, dict)
                or endpoint_policy.get("evidence_summary_nodes") != 1
                or endpoint_policy.get("decision_ready_nodes") != 1
                or endpoint_policy.get("decision_nodes") != 1
                or endpoint_policy.get("sampling_unit") != "path_endpoint_group"
            ):
                raise ValueError(f"{row['id']}: path endpoint policy metadata mismatch")
            if target_type in {"evidence_summary", "decision_ready"}:
                if not synthetic or metadata.get("synthetic_stage_type") != target_type:
                    raise ValueError(f"{row['id']}: synthetic endpoint type mismatch")
                if any(
                    metadata.get(key) is not None
                    for key in (
                        "source_message_index",
                        "source_message_event_line",
                        "source_message_item_id",
                    )
                ):
                    raise ValueError(f"{row['id']}: synthetic endpoint claims a source message")
                if not metadata.get("derived_from_verified_final_answer"):
                    raise ValueError(f"{row['id']}: endpoint lacks verified-final provenance")
                if not metadata.get("source_evidence_message_indices"):
                    raise ValueError(f"{row['id']}: endpoint lacks source evidence indices")
                if int(metadata.get("source_evidence_action_count", 0)) < 1:
                    raise ValueError(f"{row['id']}: endpoint has no source evidence action")
                if int(metadata.get("source_decisive_action_count", 0)) < 1:
                    raise ValueError(f"{row['id']}: endpoint has no decisive evidence action")
                if len(metadata.get("source_decisive_action_ids", [])) != int(
                    metadata["source_decisive_action_count"]
                ):
                    raise ValueError(f"{row['id']}: decisive evidence provenance mismatch")
                if int(metadata["current_action_count"]) != 0:
                    raise ValueError(f"{row['id']}: endpoint unexpectedly calls a tool")
                for key in (
                    "grounded_evidence_description",
                    "grounded_evidence_observation",
                    "excluded_evidence_description",
                    "excluded_evidence_observation",
                ):
                    if not str(metadata.get(key) or "").strip():
                        raise ValueError(f"{row['id']}: endpoint lacks {key}")
            elif synthetic:
                raise ValueError(f"{row['id']}: non-endpoint row is marked synthetic")

            if target_type == "evidence_summary":
                if metadata["history_contains_evidence_summary"] or metadata["history_contains_stop_judgment"]:
                    raise ValueError(f"{row['id']}: evidence summary has future endpoint history")
                if not any(message["role"] == "tool_response" for message in row["messages"]):
                    raise ValueError(f"{row['id']}: evidence summary does not inherit tool results")
                if metadata["conclusion_source"] != "path_evidence_synthesis":
                    raise ValueError(f"{row['id']}: evidence summary provenance mismatch")
            elif target_type == "decision_ready":
                if not metadata["history_contains_evidence_summary"] or metadata["history_contains_stop_judgment"]:
                    raise ValueError(f"{row['id']}: stop judgment history is malformed")
                if metadata["conclusion_source"] != "verified_path_stop_judgment":
                    raise ValueError(f"{row['id']}: stop judgment provenance mismatch")
            elif target_type == "decision":
                if not metadata["history_contains_evidence_summary"] or not metadata["history_contains_stop_judgment"]:
                    raise ValueError(f"{row['id']}: final decision does not inherit summary and stop")
            elif target_type == "hypothesis_elimination":
                statements = metadata.get("rejected_candidate_statements")
                if (
                    not isinstance(statements, list)
                    or not 1 <= len(statements) <= 2
                    or not all(isinstance(value, str) and value.strip() for value in statements)
                ):
                    raise ValueError(f"{row['id']}: rejected candidate statements are invalid")
                if metadata.get("thinking_source") != "pruned_original_visible_agent_message":
                    raise ValueError(f"{row['id']}: elimination thinking is not visible-source text")
                if metadata.get("conclusion_source") != "source_grounded_hypothesis_elimination":
                    raise ValueError(f"{row['id']}: elimination conclusion provenance mismatch")
                if metadata.get("elimination_derived_from_visible_source") is not True:
                    raise ValueError(f"{row['id']}: elimination is not marked visible-source derived")
                evidence_ids = metadata.get("elimination_evidence_action_ids")
                if (
                    not isinstance(evidence_ids, list)
                    or not 1 <= len(evidence_ids) <= 2
                    or len(evidence_ids)
                    != int(metadata.get("elimination_evidence_action_count", 0))
                ):
                    raise ValueError(f"{row['id']}: elimination evidence IDs are invalid")
                source_index = int(metadata["source_message_index"])
                source_text = str(messages_by_trajectory[trajectory_id][source_index]["text"])
                if not all(statement in source_text for statement in statements):
                    raise ValueError(f"{row['id']}: elimination statement is absent from source message")
                if not all(
                    any(marker in statement for marker in converter.ELIMINATION_MARKERS)
                    for statement in statements
                ):
                    raise ValueError(f"{row['id']}: elimination lacks an explicit rejection marker")
                prior_actions = [
                    action
                    for source_stage in stages_by_trajectory[trajectory_id]
                    if int(source_stage["source_message_index"]) < source_index
                    for action in source_stage["actions"]
                ]
                prior_actions_by_id = {
                    str(action["action_id"]): action for action in prior_actions
                }
                if any(action_id not in prior_actions_by_id for action_id in evidence_ids):
                    raise ValueError(f"{row['id']}: elimination cites future or absent evidence")
                evidence_actions = [prior_actions_by_id[action_id] for action_id in evidence_ids]
                if metadata.get("elimination_evidence_description") != converter.elimination_action_description(evidence_actions):
                    raise ValueError(f"{row['id']}: elimination evidence description mismatch")
                if metadata.get("elimination_evidence_observation") != converter.elimination_observation_description(
                    evidence_actions,
                    raw_by_trajectory[trajectory_id],
                    statements,
                ):
                    raise ValueError(f"{row['id']}: elimination evidence observation mismatch")

            for action in metadata["current_actions"]:
                source_command = base.normalize_text(str(action["source_command"]))
                source_powershell_command = base.normalize_text(
                    str(action["source_powershell_command"])
                )
                supervised_command = base.normalize_text(str(action["command"]))
                if source_command not in commands_by_trajectory[trajectory_id]:
                    raise ValueError(f"{row['id']}: action source command is not in source events")
                expected_inner = converter.normalize_supervised_command(source_command)
                if source_powershell_command != expected_inner:
                    raise ValueError(f"{row['id']}: source PowerShell normalization mismatch")
                translation = converter.translate_powershell_to_exec_command(expected_inner)
                if supervised_command != translation["cmd"]:
                    raise ValueError(f"{row['id']}: supervised Linux command translation mismatch")
                if action["justification"] != translation["justification"]:
                    raise ValueError(f"{row['id']}: exec_command justification mismatch")
                if action["command_translation"] != translation["kind"]:
                    raise ValueError(f"{row['id']}: command translation kind mismatch")
                if action["command_protocol"] != "codex_cli.exec_command.arguments.cmd":
                    raise ValueError(f"{row['id']}: command protocol audit mismatch")
                if action["result_protocol"] != "codex_cli.function_call_output.linux_normalized":
                    raise ValueError(f"{row['id']}: result protocol audit mismatch")
                if action["source_powershell_command_sha256_lf_normalized"] != base.digest_text(expected_inner):
                    raise ValueError(f"{row['id']}: source PowerShell hash mismatch")
                if action["supervised_cmd_sha256_lf_normalized"] != base.digest_text(supervised_command):
                    raise ValueError(f"{row['id']}: supervised cmd hash mismatch")

            for message in row["messages"]:
                if message["role"] != "tool_call":
                    continue
                payload = json.loads(message["content"])
                command = base.normalize_text(str(payload["arguments"]["cmd"]))
                if command not in supervised_commands_by_trajectory[trajectory_id]:
                    raise ValueError(f"{row['id']}: Linux tool call is not derived from source events")

            current_assistant_messages = [
                message for message in row["messages"] if message["role"] == "assistant"
            ][-2:]
            target_thinking = current_assistant_messages[0]["content"]
            target_answer = current_assistant_messages[1]["content"]
            if target_type == "decision":
                if base.normalize_text(target_answer) != base.normalize_text(str(raw_by_trajectory[trajectory_id]["final_answer"])):
                    raise ValueError(f"{row['id']}: final answer differs from source")
                if metadata.get("reference_answer_match") is not True:
                    raise ValueError(f"{row['id']}: decision is not marked strictly correct")
            elif target_type == "evidence_summary":
                if not all(
                    str(item) in target_answer
                    for item in raw_by_trajectory[trajectory_id]["actual_result_items"]
                ):
                    raise ValueError(f"{row['id']}: evidence summary omits the verified root")
                if metadata["grounded_evidence_observation"] not in target_answer:
                    raise ValueError(f"{row['id']}: evidence summary omits the observed result")
                if metadata["excluded_evidence_observation"] not in target_answer:
                    raise ValueError(f"{row['id']}: evidence summary omits checked exclusions")
            elif target_type == "decision_ready":
                if (
                    "停止调用工具" not in target_answer
                    or "最小根因集合已经收敛" not in target_answer
                ):
                    raise ValueError(f"{row['id']}: stop judgment is not explicit")
                if metadata["grounded_evidence_description"] not in target_answer:
                    raise ValueError(f"{row['id']}: stop judgment omits decisive evidence")
                if metadata["excluded_evidence_observation"] not in target_answer:
                    raise ValueError(f"{row['id']}: stop judgment omits checked exclusions")
                if metadata["grounded_evidence_observation"] not in target_thinking:
                    raise ValueError(f"{row['id']}: stop reasoning omits the observed result")
                if not all(
                    str(item) in target_answer
                    for item in raw_by_trajectory[trajectory_id]["actual_result_items"]
                ):
                    raise ValueError(f"{row['id']}: stop judgment omits the verified root")
            elif target_type == "hypothesis_elimination":
                if not all(
                    statement in target_thinking and statement in target_answer
                    for statement in metadata["rejected_candidate_statements"]
                ):
                    raise ValueError(f"{row['id']}: exact source rejection is not supervised")
                if "候选排除" not in target_answer:
                    raise ValueError(f"{row['id']}: elimination target is not explicit")
            elif base.normalize_text(target_answer) == base.normalize_text(str(raw_by_trajectory[trajectory_id]["final_answer"])):
                raise ValueError(f"{row['id']}: final answer leaked into a non-decision target")

            rows_by_case[case_id].append(row)
            split_by_case[case_id].add(expected_split)
            target_counts[expected_split][str(metadata["target_type"])] += 1

    primary_endpoints = {
        str(row["id"]): row
        for row in train_rows
        if row["metadata"]["target_type"] in converter.ENDPOINT_TARGET_TYPES
    }
    expected_core = [
        row
        for row in train_rows
        if row["metadata"]["target_type"] not in converter.ENDPOINT_TARGET_TYPES
    ]
    expected_endpoint_pool = [
        row
        for row in train_rows
        if row["metadata"]["target_type"] in converter.ENDPOINT_TARGET_TYPES
    ]
    if train_core_rows != expected_core:
        raise ValueError("train core is not the exact non-endpoint partition")
    if train_endpoint_pool_rows != expected_endpoint_pool:
        raise ValueError("endpoint pool is not the exact all-path endpoint partition")

    expected_schedule_count = (
        len(source_train)
        * converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
        * len(converter.ENDPOINT_TARGET_TYPES)
    )
    schedule_source_counts: Counter[str] = Counter()
    path_group_counts: Counter[tuple[int, str]] = Counter()
    all_primary_ids = {str(row["id"]) for row in all_rows}
    schedule_ids: set[str] = set()
    for epoch, schedule_rows in endpoint_schedules.items():
        if len(schedule_rows) != expected_schedule_count:
            raise ValueError(f"epoch {epoch}: endpoint schedule row count mismatch")
        per_case: Counter[int] = Counter()
        per_case_slots: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for scheduled in schedule_rows:
            scheduled_id = str(scheduled["id"])
            metadata = scheduled["metadata"]
            source_id = str(metadata.get("sampling_source_row_id"))
            case_id = int(metadata["case_id"])
            slot = int(metadata.get("endpoint_schedule_slot"))
            if scheduled_id in all_primary_ids or scheduled_id in schedule_ids:
                raise ValueError(f"duplicate scheduled endpoint row ID: {scheduled_id}")
            schedule_ids.add(scheduled_id)
            if metadata.get("training_sampling_role") != "endpoint_group_epoch_schedule":
                raise ValueError(f"{scheduled_id}: scheduled endpoint lacks sampling role")
            if (
                metadata.get("target_type") not in converter.ENDPOINT_TARGET_TYPES
                or source_id not in primary_endpoints
            ):
                raise ValueError(f"{scheduled_id}: schedule source is not a path endpoint")
            if metadata.get("endpoint_schedule_epoch") != epoch:
                raise ValueError(f"{scheduled_id}: endpoint schedule epoch mismatch")
            if not 1 <= slot <= converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH:
                raise ValueError(f"{scheduled_id}: invalid endpoint schedule slot")
            validate_loss_and_sequence(scheduled)
            expected_scheduled = copy.deepcopy(primary_endpoints[source_id])
            expected_scheduled["id"] = scheduled_id
            expected_scheduled["metadata"]["training_sampling_role"] = "endpoint_group_epoch_schedule"
            expected_scheduled["metadata"]["sampling_source_row_id"] = source_id
            expected_scheduled["metadata"]["endpoint_schedule_epoch"] = epoch
            expected_scheduled["metadata"]["endpoint_schedule_slot"] = slot
            if scheduled != expected_scheduled:
                raise ValueError(f"{scheduled_id}: schedule row differs from its source endpoint")
            per_case[case_id] += 1
            per_case_slots[case_id][slot].append(scheduled)
            schedule_source_counts[source_id] += 1
        if set(per_case) != source_train or any(
            count
            != converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
            * len(converter.ENDPOINT_TARGET_TYPES)
            for count in per_case.values()
        ):
            raise ValueError(f"epoch {epoch}: endpoint exposure is not query-balanced")
        for case_id, slots in per_case_slots.items():
            if set(slots) != set(
                range(1, converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH + 1)
            ):
                raise ValueError(f"epoch {epoch}/q{case_id}: endpoint slots are incomplete")
            for slot, group in slots.items():
                target_types = {row["metadata"]["target_type"] for row in group}
                path_ids = {row["metadata"]["path_cluster_id"] for row in group}
                if target_types != set(converter.ENDPOINT_TARGET_TYPES) or len(path_ids) != 1:
                    raise ValueError(
                        f"epoch {epoch}/q{case_id}/slot {slot}: endpoint group is not one complete path"
                    )
                path_group_counts[(case_id, str(next(iter(path_ids))))] += 1
    if set(schedule_source_counts) != set(primary_endpoints):
        raise ValueError("five-epoch endpoint schedules do not cover every retained path endpoint")
    for case_id in source_train:
        case_path_counts = [
            count
            for (scheduled_case_id, _), count in path_group_counts.items()
            if scheduled_case_id == case_id
        ]
        if not case_path_counts or max(case_path_counts) - min(case_path_counts) > 1:
            raise ValueError(f"q{case_id}: path endpoint rotation is imbalanced across epochs")

    expected_signal_audits = {
        f"epoch_{epoch:02d}": independent_training_signal_audit(
            [*train_core_rows, *endpoint_schedules[epoch]]
        )
        for epoch in range(1, converter.ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    if manifest.get("heuristic_training_signal_by_epoch") != expected_signal_audits:
        raise ValueError("manifest heuristic training-signal audit is stale")
    if max(
        audit["auto_summary_stop_weighted_percent"]
        for audit in expected_signal_audits.values()
    ) >= 30.0:
        raise ValueError("automatic summary/stop supervision exceeds 30% weighted loss")

    if set(rows_by_case) != set(by_case):
        raise ValueError("SFT rows do not cover all source cases")
    if any(len(values) != 1 for values in split_by_case.values()):
        raise ValueError("a case appears in multiple splits")
    if any(
        sum(row["metadata"]["target_type"] == "decision" for row in rows)
        != sum(row["metadata"]["target_type"] == "decision_ready" for row in rows)
        for rows in rows_by_case.values()
    ):
        raise ValueError("each retained path must have one final decision row")
    current_thinking_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        current_thinking = [
            message
            for message in row["messages"]
            if message["role"] == "assistant"
        ][-2]
        current_thinking_groups[current_thinking["content"]].append(current_thinking)
    for content, messages in current_thinking_groups.items():
        if len(messages) >= 10 and any(float(message["loss_scale"]) != 0 for message in messages):
            raise ValueError(
                f"high-frequency thinking template remains trainable: {content[:80]!r}"
            )
    endpoint_uniqueness: dict[str, tuple[int, int, int]] = {}
    for target_type, minimum_ratio in (("evidence_summary", 0.95), ("decision_ready", 0.85)):
        conclusions = [
            [message for message in row["messages"] if message["role"] == "assistant"][-1]["content"]
            for row in train_rows
            if row["metadata"]["target_type"] == target_type
        ]
        counts = Counter(conclusions)
        unique_count = len(counts)
        max_repeat = max(counts.values())
        if unique_count / len(conclusions) < minimum_ratio or max_repeat > 4:
            raise ValueError(f"{target_type}: endpoint text remains too repetitive")
        endpoint_uniqueness[target_type] = (len(conclusions), unique_count, max_repeat)
    for case_id, rows in rows_by_case.items():
        retained_paths = {
            str(row["metadata"]["path_cluster_id"])
            for row in rows
            if row["metadata"]["target_type"] == "evidence_summary"
        }
        for cluster_id in retained_paths:
            path_rows = [
                row for row in rows if row["metadata"]["path_cluster_id"] == cluster_id
            ]
            if sum(row["metadata"]["target_type"] == "evidence_summary" for row in path_rows) != 1:
                raise ValueError(f"q{case_id}/{cluster_id}: invalid summary row count")
            if sum(row["metadata"]["target_type"] == "decision_ready" for row in path_rows) != 1:
                raise ValueError(f"q{case_id}/{cluster_id}: invalid stop row count")
            if sum(row["metadata"]["target_type"] == "decision" for row in path_rows) != 1:
                raise ValueError(f"q{case_id}/{cluster_id}: invalid decision row count")
    if manifest["counts"]["train_sft_rows"] != len(train_rows) or manifest["counts"]["validation_sft_rows"] != len(validation_rows):
        raise ValueError("manifest SFT row counts are inconsistent")
    if manifest["counts"]["train_core_rows"] != len(train_core_rows):
        raise ValueError("manifest train core count is inconsistent")
    if manifest["counts"]["train_endpoint_pool_rows"] != len(train_endpoint_pool_rows):
        raise ValueError("manifest endpoint pool count is inconsistent")
    if manifest["counts"]["train_endpoint_schedule_rows_per_epoch"] != len(endpoint_schedules[1]):
        raise ValueError("manifest endpoint schedule count is inconsistent")
    if manifest["counts"]["train_endpoint_groups_per_epoch"] != len(endpoint_schedules[1]) // len(converter.ENDPOINT_TARGET_TYPES):
        raise ValueError("manifest endpoint group count is inconsistent")
    if manifest["counts"]["effective_train_row_exposures_per_epoch"] != len(train_core_rows) + len(endpoint_schedules[1]):
        raise ValueError("manifest per-epoch exposure count is inconsistent")
    expected_target_counts = {
        split: dict(sorted(counts.items())) for split, counts in target_counts.items()
    }
    if manifest["counts"]["target_types"] != expected_target_counts:
        raise ValueError("manifest target-type counts are inconsistent")
    if manifest["counts"]["evidence_summary_nodes"] != manifest["counts"]["retained_path_clusters"]:
        raise ValueError("manifest evidence-summary count differs from retained paths")
    if manifest["counts"]["decision_ready_nodes"] != manifest["counts"]["retained_path_clusters"]:
        raise ValueError("manifest stop count differs from retained paths")
    if manifest["counts"]["decision_nodes"] != manifest["counts"]["retained_path_clusters"]:
        raise ValueError("manifest decision count differs from retained paths")
    actual_eliminations = sum(
        row["metadata"]["target_type"] == "hypothesis_elimination"
        for row in all_rows
    )
    if manifest["counts"]["hypothesis_elimination_nodes"] != actual_eliminations:
        raise ValueError("manifest hypothesis-elimination count mismatch")
    if manifest["counts"]["synthetic_path_endpoint_nodes"] != 2 * manifest["counts"]["retained_path_clusters"]:
        raise ValueError("manifest synthetic endpoint count mismatch")

    q1 = next(case for case in selection_cases if int(case["case_id"]) == 1)
    print("0805 causal-path reasoning SFT validation passed")
    print("- source: 840 strictly correct trajectories, 84 cases, 10 per case")
    print(
        f"- semantic split: train={len(source_train)} cases/{len(train_rows)} nodes; "
        f"validation={len(source_validation)} cases/{len(validation_rows)} nodes"
    )
    print(
        f"- sampling per epoch: core={len(train_core_rows)} + "
        f"balanced path endpoints={len(endpoint_schedules[1])} = "
        f"{len(train_core_rows) + len(endpoint_schedules[1])} exposures; "
        f"{converter.ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH} path groups/query"
    )
    print(
        f"- paths: clustered={selection['counts']['path_clusters']}, "
        f"retained={selection['counts']['retained_path_clusters']}"
    )
    print(
        f"- endpoints: evidence summaries={manifest['counts']['evidence_summary_nodes']}, "
        f"stop decisions={manifest['counts']['decision_ready_nodes']}, "
        f"path decisions={manifest['counts']['decision_nodes']}; "
        f"max tools per node={manifest['conversion']['max_actions_per_investigative_stage']}"
    )
    print(
        f"- source-grounded hypothesis eliminations={actual_eliminations}; "
        f"train={target_counts['train']['hypothesis_elimination']}, "
        f"validation={target_counts['validation']['hypothesis_elimination']}"
    )
    auto_endpoint_ratios = [
        audit["auto_summary_stop_weighted_percent"]
        for audit in expected_signal_audits.values()
    ]
    print(
        "- heuristic auto summary+stop weighted loss: "
        f"{min(auto_endpoint_ratios):.2f}%..{max(auto_endpoint_ratios):.2f}% across five epochs"
    )
    print(
        "- endpoint text uniqueness (train): "
        f"summary={endpoint_uniqueness['evidence_summary'][1]}/{endpoint_uniqueness['evidence_summary'][0]}, "
        f"stop={endpoint_uniqueness['decision_ready'][1]}/{endpoint_uniqueness['decision_ready'][0]}"
    )
    print(
        f"- q0001: raw checkpoints={q1['raw_visible_checkpoint_count']}, "
        f"retained paths={q1['retained_path_cluster_count']}, "
        f"selected nodes={q1['selected_sft_node_count']}"
    )
    print(f"- validation IDs: {sorted(source_validation)}")


if __name__ == "__main__":
    main()
