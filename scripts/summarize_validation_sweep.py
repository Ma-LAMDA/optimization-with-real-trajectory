#!/usr/bin/env python3
"""Aggregate base-model and LoRA validation targets into one comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    targets: list[dict[str, Any]] = []
    for target_root in sorted(path for path in input_root.iterdir() if path.is_dir()):
        summary_path = target_root / "validation_summary.json"
        if not summary_path.is_file():
            continue
        summary = load_object(summary_path)
        counts = summary.get("counts")
        rates = summary.get("rates")
        topology = summary.get("topology")
        if not isinstance(counts, dict) or not isinstance(rates, dict):
            raise ValueError(f"{summary_path}: counts or rates are missing")
        if topology != {
            "instance_count": 1,
            "worker_count": 2,
            "request_concurrency": 2,
        }:
            raise ValueError(f"{summary_path}: unexpected evaluation topology")
        targets.append(
            {
                "target": target_root.name,
                "model": summary.get("model"),
                "checkpoint": summary.get("checkpoint"),
                "repeat_count": summary.get("repeat_count", 1),
                "counts": counts,
                "rates": rates,
                "latency_seconds": summary.get("latency_seconds"),
                "summary": summary_path.as_posix(),
            }
        )
    if not targets:
        raise ValueError(f"{input_root}: no validation summaries found")

    base = next(
        (target for target in targets if target["target"].endswith("-base")),
        None,
    )
    base_rate = (
        float(base["rates"]["exact_match"])
        if base is not None
        else None
    )
    for target in targets:
        rate = float(target["rates"]["exact_match"])
        target["exact_match_delta_vs_base"] = (
            rate - base_rate if base_rate is not None else None
        )

    aggregate = {
        "schema_version": "qwen36-sft-validation-sweep.v1",
        "input_root": input_root.as_posix(),
        "base_target": base["target"] if base is not None else None,
        "targets": targets,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for target in targets:
        counts = target["counts"]
        print(
            f"{target['target']}: "
            f"{counts['exact_match']}/{counts['total']} exact matches"
        )
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
