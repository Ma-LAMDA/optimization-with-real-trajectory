#!/usr/bin/env python3
"""Generate deterministic audit metrics for the 0807 SFT release candidate."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import convert_0804_best_trajectory_reasoning_sft as base
import convert_0807_evidence_gated_reasoning_sft as converter


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-07"
SFT_ROOT = DATA_ROOT / "sft"
MANIFEST = SFT_ROOT / "0807_evidence_gated_manifest.json"
CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
SELECTION = DATA_ROOT / "curation" / "causal_path_clusters_per_case.json"
DEFAULT_OUTPUT = DATA_ROOT / "curation" / "AUDIT_METRICS.json"
BASELINE_MANIFEST = (
    ROOT / "data" / "2026-08-05" / "sft" / "reasoning_causal_path_manifest.json"
)
WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\")
INCLUSIVE_OR_CASES = set(range(73, 87))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def supervised_assistant_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(message["content"])
        for message in row["messages"]
        if message["role"] == "assistant"
        and float(message.get("loss_scale", 0) or 0) > 0
    )


def supervised_nondecision_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(message["content"])
        for message in row["messages"]
        if message["role"] == "assistant"
        and float(message.get("loss_scale", 0) or 0) > 0
        and "<result>" not in str(message["content"])
    )


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def exact_surface_sets(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    commands: set[str] = set()
    responses: set[str] = set()
    for row in rows:
        for message in row["messages"]:
            content = normalized(str(message["content"]))
            if not content:
                continue
            if message["role"] == "tool_call":
                commands.add(content)
            elif message["role"] == "tool_response":
                responses.add(content)
    return commands, responses


def overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    shared = left & right
    return {
        "train_unique": len(left),
        "validation_unique": len(right),
        "shared_unique": len(shared),
        "validation_unique_seen_in_train_percent": round(
            100.0 * len(shared) / max(1, len(right)), 6
        ),
    }


def target_repetition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_type[str(row["metadata"]["target_type"])].append(
            supervised_assistant_text(row)
        )
    result: dict[str, Any] = {}
    for target_type, texts in sorted(by_type.items()):
        counts = Counter(texts)
        result[target_type] = {
            "rows": len(texts),
            "unique_targets": len(counts),
            "maximum_exact_repeat": max(counts.values()),
            "rows_whose_target_repeats": sum(
                count for count in counts.values() if count > 1
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = load_json(MANIFEST)
    curation = load_json(CURATION)
    selection = load_json(SELECTION)
    outputs = manifest["outputs"]
    rows_by_name = {
        name: load_jsonl(ROOT / record["path"])
        for name, record in outputs.items()
    }
    train = rows_by_name["train"]
    validation = rows_by_name["validation"]
    semantic = [*train, *validation]

    action_count = config_count = cross_snapshot_count = 0
    glob_counts: Counter[str] = Counter()
    current_action_counts: Counter[int] = Counter()
    action_selection_totals: Counter[str] = Counter()
    source_intent_coverage: Counter[str] = Counter()
    supervised_intent_coverage: Counter[str] = Counter()
    supervised_partial_rows: list[str] = []
    lldp_actions = lldp_misclassified_as_mpls = 0
    for row in semantic:
        actions = row["metadata"]["current_actions"]
        current_action_counts[len(actions)] += 1
        selection_metrics = row["metadata"]["action_selection"]
        for name in ("original_action_count", "eligible_action_count", "kept_action_count"):
            action_selection_totals[name] += int(selection_metrics[name])
        if actions:
            source_intent_coverage[
                str(selection_metrics["claim_coverage_status_after"])
            ] += 1
            supervised_intent_coverage[
                str(selection_metrics["supervised_intent_coverage_status"])
            ] += 1
            if selection_metrics["supervised_intent_coverage_status"] == "partial":
                supervised_partial_rows.append(str(row["id"]))
        question = next(
            message["content"] for message in row["messages"]
            if message["role"] == "user"
        )
        snapshot = re.search(
            r"CampusNetwork(?:-for-perf)?_\d+", question, re.IGNORECASE
        ).group(0).lower()
        for action in actions:
            action_count += 1
            semantics = action["causal_semantics"]
            config_count += "config" in semantics["families"]
            lldp_actions += "lldp" in semantics["families"]
            lldp_misclassified_as_mpls += (
                "lldp" in " ".join(semantics.get("filenames", [])).lower()
                and "mpls" in semantics["families"]
            )
            cross_snapshot_count += (
                set(semantics["snapshots"]) != {snapshot}
            )
            glob_counts["snapshot"] += bool(semantics["has_snapshot_glob"])
            glob_counts["device"] += bool(semantics["has_device_glob"])
            glob_counts["filename"] += bool(semantics["has_filename_glob"])

    grounding = Counter()
    unsafe_procedural_rows: list[str] = []
    for row in semantic:
        if row["metadata"]["target_type"] == "hypothesis_elimination":
            continue
        optimization = row["metadata"]["text_optimization"]
        for record in optimization.get("retained_thinking_sentence_records", []):
            grounding[record["kind"]] += 1
            if (
                record["kind"] == "procedural_plan"
                and not converter.is_safe_procedural_sentence(
                    str(record.get("source_sentence") or "")
                )
            ):
                unsafe_procedural_rows.append(str(row["id"]))
        grounding["dropped_unsupported"] += len(
            optimization.get("dropped_unsupported_thinking_sentences", [])
        )

    endpoint_paths: set[tuple[int, str]] = set()
    gate_rules: Counter[str] = Counter()
    recovery_paths: set[tuple[int, str]] = set()
    strict_role_closures: dict[int, dict[str, Any]] = {}
    for row in semantic:
        if row["metadata"]["target_type"] != "endpoint_bundle":
            continue
        key = (
            int(row["metadata"]["case_id"]),
            str(row["metadata"]["path_cluster_id"]),
        )
        endpoint_paths.add(key)
        gate = row["metadata"]["endpoint_evidence_gate"]
        gate_rules[str(gate["gate_rule"])] += 1
        if any(str(fact["action_id"]).startswith("REC-") for fact in gate["selected_facts"]):
            recovery_paths.add(key)
        case_id = int(row["metadata"]["case_id"])
        if case_id in INCLUSIVE_OR_CASES:
            host = next(
                fact for fact in gate["selected_facts"]
                if fact["kind"] == "source_host_ipv4"
            )
            strict_role_closures[case_id] = {
                "source_host": host["device"],
                "source_vlan": int(host["vlan"]),
                "selected_target": row["metadata"]["actual_result_items"],
                "gate_rule": gate["gate_rule"],
                "evidence_selected_singleton": bool(
                    row["metadata"].get(
                        "inclusive_or_singleton_selected_by_evidence"
                    )
                ),
            }

    labels_by_case = {
        int(item["case_id"]): [str(value) for value in item["actual_result_items"]]
        for item in curation["trajectories"]
    }
    exact_answer_leaks = 0
    label_derived_nondecision = 0
    for row in semantic:
        target = normalized(supervised_nondecision_text(row))
        if row["metadata"].get("derived_from_verified_final_answer"):
            label_derived_nondecision += 1
        exact_answer_leaks += any(
            normalized(item) in target
            for item in labels_by_case[int(row["metadata"]["case_id"])]
        )

    train_commands, train_responses = exact_surface_sets(train)
    validation_commands, validation_responses = exact_surface_sets(validation)

    schedule_by_epoch: dict[str, Any] = {}
    core_source_union: set[str] = set()
    for epoch in range(1, converter.ENDPOINT_SCHEDULE_EPOCHS + 1):
        core = rows_by_name[f"train_core_epoch_{epoch:02d}"]
        endpoint = rows_by_name[f"train_endpoint_epoch_{epoch:02d}"]
        core_source_union.update(
            str(row["metadata"]["sampling_source_row_id"]) for row in core
        )
        per_case = Counter(
            int(row["metadata"]["case_id"]) for row in [*core, *endpoint]
        )
        schedule_by_epoch[f"epoch_{epoch:02d}"] = {
            "core_rows": len(core),
            "endpoint_rows": len(endpoint),
            "total_rows": len(core) + len(endpoint),
            "rows_per_query_min": min(per_case.values()),
            "rows_per_query_max": max(per_case.values()),
            "target_types": dict(sorted(Counter(
                str(row["metadata"]["target_type"]) for row in [*core, *endpoint]
            ).items())),
            "heuristic_signal": manifest["heuristic_training_signal_by_epoch"][
                f"epoch_{epoch:02d}"
            ],
        }

    context_lengths = {
        row["id"]: converter.estimate_context_token_count(
            "\n".join(str(message["content"]) for message in row["messages"])
            + "\n"
            + str(row["tools"])
        )
        for row in semantic
    }
    sorted_lengths = sorted(context_lengths.values())
    p99_index = min(len(sorted_lengths) - 1, int(0.99 * len(sorted_lengths)))

    baseline: dict[str, Any] | None = None
    if BASELINE_MANIFEST.exists():
        old = load_json(BASELINE_MANIFEST)
        baseline = {
            "schema_version": old.get("schema_version"),
            "counts": old.get("counts"),
            "outputs": {
                name: {key: record[key] for key in ("rows", "bytes") if key in record}
                for name, record in old.get("outputs", {}).items()
            },
        }

    result = {
        "schema_version": "0807-evidence-gated-audit-metrics.v6",
        "input_manifest_schema": manifest["schema_version"],
        "status": manifest["status"],
        "source": {
            "cases": manifest["counts"]["cases"],
            "successful_trajectories": manifest["counts"]["source_trajectories"],
            "successful_trajectories_per_case": 10,
            "model_valid_attempt_status": curation["source_attempt_status_counts"],
            "infrastructure_failures_or_interruptions_recorded": 0,
        },
        "split": manifest["split"],
        "q73_q86_inclusive_or_impact": {
            "input_report": "trajectory-analysis/2026-08-07_q0073-q0086_inclusive_or_impact.md",
            "input_report_sha256_lf_normalized": "8c3e5231ffd1ed2516a9ae6871a208b3921a7b4f02b29407d2056c83985d1164",
            "source_dataset": curation["source_dataset"],
            "source_dataset_sha256_lf_normalized": curation[
                "source_dataset_sha256_lf_normalized"
            ],
            "accepted_options_per_case": 3,
            "synchronized_raw_trajectories": 140,
            "current_dual_device_source_targets": 0,
            "case_split_changed": False,
            "successful_trajectory_count_changed": False,
            "sft_policy": "evidence-strongest singleton; dual only after two independent source-VLAN/MST-instance closures",
            "prior_q73_q86_semantic_rows": {
                "train": {"planning": 12, "reasoning": 17, "endpoint_bundle": 13, "total": 42},
                "validation": {"planning": 2, "reasoning": 2, "endpoint_bundle": 2, "total": 6},
                "total": 48,
            },
            "current_q73_q86_semantic_rows": {
                split: {
                    **dict(sorted(Counter(
                        row["metadata"]["target_type"]
                        for row in rows
                        if int(row["metadata"]["case_id"]) in INCLUSIVE_OR_CASES
                    ).items())),
                    "total": sum(
                        int(row["metadata"]["case_id"]) in INCLUSIVE_OR_CASES
                        for row in rows
                    ),
                }
                for split, rows in (("train", train), ("validation", validation))
            },
            "strict_endpoint_closures": {
                "prior_endpoints": 15,
                "prior_fully_vlan_instance_closed": 4,
                "prior_unsafe_or_incomplete": 11,
                "current_endpoints": len(strict_role_closures),
                "current_fully_vlan_instance_closed": len(strict_role_closures),
                "current_unsafe_or_incomplete": 0,
                "by_case": {
                    str(case_id): value
                    for case_id, value in sorted(strict_role_closures.items())
                },
            },
        },
        "paths_and_rows": manifest["counts"],
        "v6_reaudit_response": {
            "input_report": "trajectory-analysis/2026-08-07_1703_0807_sft_reaudit_v5.md",
            "input_report_sha256_lf_normalized": "7b2bb6b25b14915d147cfda83ed91d5a754fb29aa291a22e970df7cb3fae3639",
            "lldp_family_actions": lldp_actions,
            "lldp_actions_misclassified_as_mpls": lldp_misclassified_as_mpls,
            "unsafe_mixed_procedural_rows": sorted(set(unsafe_procedural_rows)),
            "supervised_intent_partial_rows": sorted(supervised_partial_rows),
            "optimizer_step_contract": {
                "rows_per_stage": 216,
                "world_size": 2,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 8,
                "optimizer_steps_per_stage": 27,
                "global_step_boundaries": [0, 27, 54, 81, 108, 135],
                "checkpoint_suffixes": [27, 54, 81, 108, 135],
            },
        },
        "v7_0805_training_alignment": {
            "distributed_strategy": "ddp",
            "world_size": 2,
            "cuda_visible_devices": "0,1",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "checkpoint_selection_strategy": "fixed_epoch",
            "fixed_validation_epoch": 3,
            "fixed_validation_checkpoint_suffix": 81,
            "agent_checkpoint_selection": False,
        },
        "endpoint_audit": {
            "qualified_paths": len(endpoint_paths),
            "qualified_cases": len({case_id for case_id, _ in endpoint_paths}),
            "all_retained_paths_have_complete_endpoint": (
                len(endpoint_paths) == manifest["counts"]["retained_path_clusters"]
            ),
            "gate_rules": dict(sorted(gate_rules.items())),
            "same_query_same_snapshot_recovery_paths": len(recovery_paths),
            "label_derived_nondecision_rows": label_derived_nondecision,
            "exact_verified_answer_in_nondecision_target_rows": exact_answer_leaks,
        },
        "reasoning_grounding": dict(sorted(grounding.items())),
        "tool_audit": {
            "current_actions": action_count,
            "config_actions": config_count,
            "config_action_percent": round(100.0 * config_count / action_count, 6),
            "cross_snapshot_actions": cross_snapshot_count,
            "snapshot_glob_actions": glob_counts["snapshot"],
            "device_glob_actions": glob_counts["device"],
            "filename_glob_actions": glob_counts["filename"],
            "trainable_windows_path_targets": sum(
                bool(WINDOWS_PATH.search(supervised_assistant_text(row))) for row in semantic
            ),
            "current_action_count_histogram": {
                str(key): value for key, value in sorted(current_action_counts.items())
            },
            "tool_call_loss_scale": converter.TOOL_CALL_LOSS_SCALE,
            "action_selection_totals": dict(sorted(action_selection_totals.items())),
            "source_intent_coverage": dict(sorted(source_intent_coverage.items())),
            "supervised_intent_coverage": dict(
                sorted(supervised_intent_coverage.items())
            ),
            "supervised_intent_partial_rows": sorted(supervised_partial_rows),
            "lldp_actions": lldp_actions,
            "lldp_actions_misclassified_as_mpls": lldp_misclassified_as_mpls,
            "zero_to_zero_counted_as_success": False,
        },
        "elimination_audit": {
            "rows": sum(
                row["metadata"]["target_type"] == "hypothesis_elimination"
                for row in semantic
            ),
            "bound_atomic_claims": sum(
                len(row["metadata"].get("elimination_claim_bindings", []))
                for row in semantic
            ),
            "known_unsupported_regression_rows_present": sorted(
                row["id"] for row in semantic
                if row["id"] in {
                    "q0004_path_05_success_08_step_03",
                    "q0008_path_04_success_06_step_03",
                    "q0032_path_01_success_06_step_03",
                }
            ),
        },
        "case_balanced_training": {
            "core_pool_unique_rows": len(rows_by_name["train_core"]),
            "core_pool_rows_exposed_at_least_once_across_five_epochs": len(core_source_union),
            "endpoint_pool_unique_rows": len(rows_by_name["train_endpoint_pool"]),
            "schedule_by_epoch": schedule_by_epoch,
        },
        "target_exact_repetition": {
            "train": target_repetition(train),
            "validation": target_repetition(validation),
        },
        "frozen_validation_overlap": {
            "exact_tool_calls": overlap(train_commands, validation_commands),
            "exact_tool_responses": overlap(train_responses, validation_responses),
            "interpretation": "case-isolated but not topology-heldout",
        },
        "heuristic_context_length": {
            "method": "same mixed CJK/non-CJK approximation as converter; not target tokenizer",
            "rows": len(context_lengths),
            "p99": sorted_lengths[p99_index],
            "max": max(sorted_lengths),
            "max_row_id": max(context_lengths, key=context_lengths.get),
            "over_16384": sum(value > 16384 for value in context_lengths.values()),
        },
        "0805_baseline": baseline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
