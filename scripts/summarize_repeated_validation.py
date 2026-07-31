#!/usr/bin/env python3
"""Validate and aggregate repeated SFT validation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


TOPOLOGY = {
    "instance_count": 1,
    "worker_count": 2,
    "request_concurrency": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            values.append(value)
    return values


def percentile(values: list[float], proportion: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def common_value(
    summaries: list[dict[str, Any]], field: str
) -> Any:
    values = [summary.get(field) for summary in summaries]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"repeated validation field differs: {field}")
    return values[0]


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("repeated validation aggregation requires at least 2 runs")

    input_root = args.input_root.resolve()
    summaries: list[dict[str, Any]] = []
    per_repeat: list[dict[str, Any]] = []
    prediction_paths: list[str] = []
    durations: list[float] = []

    count_fields = (
        "total",
        "completed",
        "failed",
        "format_valid",
        "exact_match",
        "leak_free",
    )
    aggregate_counts = {field: 0 for field in count_fields}

    for repeat_index in range(1, args.repeats + 1):
        repeat_root = input_root / f"repeat_{repeat_index:02d}"
        summary_path = repeat_root / "validation_summary.json"
        prediction_path = repeat_root / "validation_predictions.jsonl"
        summary = load_object(summary_path)
        predictions = load_jsonl(prediction_path)
        counts = summary.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"{summary_path}: counts are missing")
        if counts.get("failed") != 0:
            raise ValueError(f"{summary_path}: validation has failed requests")
        if summary.get("topology") != TOPOLOGY:
            raise ValueError(f"{summary_path}: invalid evaluation topology")
        if counts.get("total") != len(predictions):
            raise ValueError(f"{summary_path}: prediction count differs")

        for field in count_fields:
            value = counts.get(field)
            if not isinstance(value, int):
                raise ValueError(f"{summary_path}: invalid count {field}")
            aggregate_counts[field] += value

        for prediction in predictions:
            duration = prediction.get("duration_seconds")
            if prediction.get("status") == "completed" and isinstance(
                duration, (int, float)
            ):
                durations.append(float(duration))

        summaries.append(summary)
        prediction_paths.append(prediction_path.as_posix())
        per_repeat.append(
            {
                "repeat_index": repeat_index,
                "summary": summary_path.as_posix(),
                "predictions": prediction_path.as_posix(),
                "counts": counts,
                "rates": summary.get("rates"),
                "latency_seconds": summary.get("latency_seconds"),
            }
        )

    total = aggregate_counts["total"]
    if total <= 0:
        raise ValueError("repeated validation has no predictions")
    aggregate = {
        "schema_version": "qwen36-sft-repeated-validation-eval.v1",
        "git_commit": common_value(summaries, "git_commit"),
        "checkpoint": common_value(summaries, "checkpoint"),
        "dataset": common_value(summaries, "dataset"),
        "model": common_value(summaries, "model"),
        "topology": TOPOLOGY,
        "sampling": common_value(summaries, "sampling"),
        "repeat_count": args.repeats,
        "counts": aggregate_counts,
        "rates": {
            "format_valid": aggregate_counts["format_valid"] / total,
            "exact_match": aggregate_counts["exact_match"] / total,
            "leak_free": aggregate_counts["leak_free"] / total,
        },
        "latency_seconds": {
            "total": sum(
                float(summary["latency_seconds"]["total"])
                for summary in summaries
            ),
            "mean": statistics.fmean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p95": percentile(durations, 0.95) if durations else None,
        },
        "per_repeat": per_repeat,
        "predictions": prediction_paths,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Repeated validation: {args.repeats} runs, "
        f"{aggregate_counts['exact_match']}/{total} exact matches"
    )
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
