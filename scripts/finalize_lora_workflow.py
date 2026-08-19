#!/usr/bin/env python3
"""Combine training and validation summaries into one workflow result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--model-path", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    args = parse_args()
    training_path = args.training_summary.resolve()
    validation_path = args.validation_summary.resolve()
    manifest_path = args.manifest.resolve()
    training = load_json(training_path)
    validation = load_json(validation_path)
    manifest = load_json(manifest_path)

    if training.get("status") != "completed":
        raise ValueError("training summary is not complete")
    if validation.get("status") != "completed":
        raise ValueError("Agent validation summary is not complete")
    if validation.get("evaluation_method") != "full_codex_agent_with_tools":
        raise ValueError("final validation did not use the full Codex Agent runner")
    if validation.get("topology") != {
        "instance_count": 1,
        "tensor_parallel_size": 2,
        "worker_count": 2,
        "request_concurrency": 2,
    }:
        raise ValueError("validation topology is not single-instance dual-concurrency")
    validation_case_ids = manifest.get("split", {}).get("validation_case_ids")
    if validation.get("case_ids") != validation_case_ids:
        raise ValueError("Agent validation cases differ from the manifest holdout cases")
    expected_attempts = len(validation_case_ids) * int(
        validation.get("repeats_per_case") or 0
    )
    if validation.get("counts", {}).get("attempts") != expected_attempts:
        raise ValueError("Agent validation attempt count is incomplete")

    result = {
        "schema_version": "qwen36-lora-sft-workflow.v2",
        "status": "completed",
        "run_id": args.run_id,
        "branch": args.branch,
        "git_commit": args.git_commit,
        "model_path": args.model_path,
        "data": {
            "manifest": manifest_path.as_posix(),
            "manifest_sha256_lf_normalized": digest(manifest_path),
            "train_samples": manifest.get("split", {}).get("train"),
            "validation_samples": manifest.get("split", {}).get("validation"),
            "validation_case_ids": validation_case_ids,
            "case_groups_disjoint": manifest.get("split", {}).get(
                "case_groups_disjoint"
            ),
        },
        "training": training,
        "validation_evaluation": validation,
        "artifacts": {
            "training_summary": training_path.as_posix(),
            "validation_summary": validation_path.as_posix(),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Workflow completed: {output}")


if __name__ == "__main__":
    main()
