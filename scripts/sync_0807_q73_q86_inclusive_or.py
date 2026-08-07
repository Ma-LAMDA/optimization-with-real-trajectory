#!/usr/bin/env python3
"""Synchronize the q73-q86 inclusive-OR reference revision into the 0807 archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
CURATION = ROOT / "data" / "2026-08-07" / "curation" / "accepted_trajectory_selection.json"
QUESTION_IDS = set(range(73, 87))
POLICY = "q73_q86_inclusive_or_single_or_dual_exact_fault_set.v1"


def digest_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def option_key(items: list[str]) -> frozenset[str]:
    return frozenset(str(item) for item in items)


def source_options() -> dict[int, list[list[str]]]:
    rows = [
        json.loads(line)
        for line in SOURCE_DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id: dict[int, list[list[str]]] = {}
    for row in rows:
        question_id = int(row["id"])
        if question_id not in QUESTION_IDS:
            continue
        value = json.loads(row["answer"])
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"q{question_id}: expected exactly three inclusive-OR options")
        options = [[str(item) for item in option] for option in value]
        if [len(option) for option in options] != [1, 1, 2]:
            raise ValueError(f"q{question_id}: expected singleton A, singleton B, and A+B")
        answer_a, answer_b = options[0][0], options[1][0]
        if (
            not answer_a.startswith("Core_SW_01;")
            or not answer_b.startswith("Core_SW_02;")
            or answer_a.split(";", 1)[1] != answer_b.split(";", 1)[1]
            or option_key(options[2]) != frozenset({answer_a, answer_b})
            or len({option_key(option) for option in options}) != 3
        ):
            raise ValueError(f"q{question_id}: malformed inclusive-OR answer options")
        by_id[question_id] = options
    if set(by_id) != QUESTION_IDS:
        raise ValueError(f"source dataset is missing q73-q86: {sorted(QUESTION_IDS - set(by_id))}")
    return by_id


def main() -> None:
    options_by_id = source_options()
    source_hash = digest_file(SOURCE_DATASET)
    curation = load_json(CURATION)
    updated = 0
    dual_target_count = 0
    for annotation in curation["trajectories"]:
        question_id = int(annotation["case_id"])
        if question_id not in QUESTION_IDS:
            continue
        raw_path = ROOT / str(annotation["raw_file"])
        raw = load_json(raw_path)
        options = options_by_id[question_id]
        prediction = [str(item) for item in raw.get("actual_result_items") or []]
        matches = option_key(prediction) in {option_key(option) for option in options}
        if not matches:
            raise ValueError(f"{raw['id']}: accepted result no longer matches inclusive OR")
        dual_target_count += len(prediction) == 2
        revision = {
            "policy": POLICY,
            "source_dataset": SOURCE_DATASET.relative_to(ROOT).as_posix(),
            "source_dataset_sha256_lf_normalized": source_hash,
            "original_independent_judgment_remains_correct": True,
            "accepted_target_device_counts": [1, 2],
            "sft_endpoint_preference": "single evidence-strongest device; dual only after two independent VLAN-instance closures",
        }
        raw["reference_answer_options"] = options
        raw["answer_matches_reference"] = True
        raw["reference_answer_revision"] = revision
        write_json(raw_path, raw)
        annotation["reference_answer_options"] = options
        annotation["reference_answer_revision"] = revision
        annotation["raw_sha256_lf_normalized"] = digest_file(raw_path)
        updated += 1
    if updated != 140:
        raise ValueError(f"expected 140 q73-q86 trajectories, updated {updated}")
    curation["schema_version"] = "codex-ip-accepted-trajectory-curation.v5"
    curation["source_dataset_sha256_lf_normalized"] = source_hash
    curation["selection"]["answer_filter"] = (
        "independent_exact_fault_set_match_with_q73_q86_inclusive_or_revision"
    )
    curation["reference_answer_policy"] = {
        "policy": POLICY,
        "source_dataset": SOURCE_DATASET.relative_to(ROOT).as_posix(),
        "source_dataset_sha256_lf_normalized": source_hash,
        "affected_case_ids": sorted(QUESTION_IDS),
        "accepted_options_per_case": 3,
        "updated_trajectory_count": updated,
        "existing_dual_device_trajectory_count": dual_target_count,
        "split_or_success_count_changed": False,
        "sft_endpoint_preference": "single evidence-strongest device; dual only after two independent VLAN-instance closures",
    }
    write_json(CURATION, curation)
    print(
        f"Synchronized {updated} q73-q86 raw trajectories; "
        f"dual targets={dual_target_count}; source={source_hash}"
    )


if __name__ == "__main__":
    main()
