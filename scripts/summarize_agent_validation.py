#!/usr/bin/env python3
"""Strictly score full Codex Agent validation attempts and write a compact report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from final_answer_scoring import parse_final_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_options(expected: Any) -> list[list[str]]:
    if isinstance(expected, list) and all(isinstance(item, str) for item in expected):
        return [expected]
    if (
        isinstance(expected, list)
        and expected
        and all(isinstance(option, list) for option in expected)
        and all(all(isinstance(item, str) for item in option) for option in expected)
    ):
        return expected
    raise TypeError("expected answer must be a JSON list of strings or alternatives")


def prediction_matches(prediction: Any, expected: Any) -> bool:
    return isinstance(prediction, list) and any(
        prediction == option for option in expected_options(expected)
    )


def false_counts(prediction: Any, expected: Any) -> tuple[int, int]:
    prediction_set = set(prediction) if isinstance(prediction, list) else set()
    differences = []
    for option in expected_options(expected):
        option_set = set(option)
        differences.append(
            (
                len(prediction_set - option_set),
                len(option_set - prediction_set),
            )
        )
    return min(differences, key=lambda counts: (sum(counts), counts))


def load_expected(path: Path) -> dict[int, Any]:
    rows: dict[int, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            identifier = row.get("id")
            answer = json.loads(row.get("answer", "null"))
            if not isinstance(identifier, int):
                raise ValueError(f"{path}:{line_number}: invalid id or answer")
            expected_options(answer)
            rows[identifier] = answer
    return rows


def selected_attempt(run: dict[str, Any]) -> int | None:
    selected = run.get("successful_attempt")
    attempts = run.get("attempts") or []
    if selected is None and attempts:
        selected = attempts[-1].get("attempt_index")
    return int(selected) if selected is not None else None


def event_metrics(slot: Path) -> dict[str, int]:
    metrics = Counter()
    for event_path in sorted(slot.glob("attempt_*/events.jsonl")):
        for line in event_path.open(encoding="utf-8", errors="ignore"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                metrics["invalid_events"] += 1
                continue
            metrics["events"] += 1
            item = event.get("item") or {}
            if event.get("type") == "item.completed":
                if item.get("type") == "command_execution":
                    metrics["commands"] += 1
                elif item.get("type") == "agent_message":
                    metrics["agent_messages"] += 1
                elif item.get("type") == "reasoning":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        metrics["reasoning_items"] += 1
                        metrics["reasoning_characters"] += len(text)
    return dict(metrics)


def token_metrics(slot: Path) -> dict[str, int]:
    totals = Counter()
    for metadata_path in sorted(slot.glob("attempt_*/metadata.json")):
        try:
            usage = load_json(metadata_path).get("usage") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        ):
            totals[key] += int(usage.get(key) or 0)
    return dict(totals)


def parse_attempt(
    output_root: Path,
    prefix: str,
    case_id: int,
    repeat: int,
    expected: Any,
    timeout_seconds: int,
    model: str,
) -> dict[str, Any]:
    root = output_root / f"{prefix}-q{case_id}-r{repeat:02d}"
    manifest_path = root / "manifest.json"
    timeout_marker = (root / f".timeout_{timeout_seconds}s").exists()
    exit_marker = root / ".runner_exit_code"
    if not manifest_path.is_file() and not timeout_marker and not exit_marker.is_file():
        raise FileNotFoundError(f"attempt is incomplete: {root}")

    manifest: dict[str, Any] = {}
    run: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        runs = manifest.get("runs") or []
        run = runs[0] if runs else {}
    slot = root / f"q{case_id:04d}_r01"
    attempts = run.get("attempts") or []
    duration = sum(float(item.get("duration_seconds") or 0) for item in attempts)
    if not duration:
        for metadata_path in sorted(slot.glob("attempt_*/metadata.json")):
            try:
                duration += float(load_json(metadata_path).get("duration_seconds") or 0)
            except (OSError, json.JSONDecodeError):
                pass
    timeout = timeout_marker or duration > timeout_seconds
    if timeout:
        duration = max(duration, float(timeout_seconds))

    attempt_index = selected_attempt(run)
    answer_path = (
        slot / f"attempt_{attempt_index:03d}" / "final_answer.txt"
        if attempt_index is not None
        else None
    )
    answer_text = (
        answer_path.read_text(encoding="utf-8", errors="replace")
        if answer_path and answer_path.is_file()
        else ""
    )
    parsed_answer = parse_final_answer(answer_text, expected)
    prediction = parsed_answer.value
    runner_status = manifest.get(
        "status",
        "timeout" if timeout else "failed_before_manifest",
    )
    metadata: dict[str, Any] = {}
    if attempt_index is not None:
        metadata_path = slot / f"attempt_{attempt_index:03d}" / "metadata.json"
        if metadata_path.is_file():
            metadata = load_json(metadata_path)
    event_type_counts = metadata.get("event_type_counts") or {}
    completed_model_turn = (
        runner_status == "failed"
        and metadata.get("exit_code") == 0
        and int(event_type_counts.get("turn.completed") or 0) > 0
        and not metadata.get("error_events")
        and not metadata.get("invalid_jsonl_events")
        and not metadata.get("launch_error")
    )
    model_completed_without_valid_answer = completed_model_turn and prediction is None
    infrastructure_failure = (
        runner_status != "succeeded"
        and not timeout
        and not completed_model_turn
    )
    correct = (
        (runner_status == "succeeded" or completed_model_turn)
        and not timeout
        and prediction_matches(prediction, expected)
    )
    false_positive_count, false_negative_count = false_counts(prediction, expected)
    events = event_metrics(slot)
    tokens = token_metrics(slot)
    return {
        "model": model,
        "case_id": case_id,
        "repeat": repeat,
        "runner_status": runner_status,
        "model_completed_without_valid_answer": model_completed_without_valid_answer,
        "infrastructure_failure": infrastructure_failure,
        "duration_seconds": round(duration, 3),
        "capped_minutes": round(min(duration, timeout_seconds) / 60, 3),
        "timeout": timeout,
        "completed_within_limit": not timeout,
        "correct": correct,
        "result": "timeout" if timeout else ("correct" if correct else "wrong"),
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "events": events.get("events", 0),
        "commands": events.get("commands", 0),
        "agent_messages": events.get("agent_messages", 0),
        "reasoning_items": events.get("reasoning_items", 0),
        "reasoning_characters": events.get("reasoning_characters", 0),
        "input_tokens": tokens.get("input_tokens", 0),
        "cached_input_tokens": tokens.get("cached_input_tokens", 0),
        "output_tokens": tokens.get("output_tokens", 0),
        "reasoning_output_tokens": tokens.get("reasoning_output_tokens", 0),
        "prediction": prediction,
        "prediction_source": parsed_answer.source,
        "format_recovered": parsed_answer.recovered,
        "expected": expected,
        "artifact_dir": str(root),
    }


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(row["capped_minutes"]) for row in rows]
    mean = statistics.mean(durations)
    deviation = statistics.pstdev(durations)
    return {
        "attempts": len(rows),
        "completed_within_limit": sum(row["completed_within_limit"] for row in rows),
        "timeouts": sum(row["timeout"] for row in rows),
        "runner_failures": sum(
            row["infrastructure_failure"] for row in rows
        ),
        "strict_correct": sum(row["correct"] for row in rows),
        "accuracy_percent": 100 * sum(row["correct"] for row in rows) / len(rows),
        "false_positives": sum(row["false_positive_count"] for row in rows),
        "false_negatives": sum(row["false_negative_count"] for row in rows),
        "runtime_minutes": {
            "mean": mean,
            "median": statistics.median(durations),
            "p95": percentile_95(durations),
            "population_stddev": deviation,
            "coefficient_of_variation": deviation / mean if mean else None,
        },
        "events": sum(row["events"] for row in rows),
        "commands": sum(row["commands"] for row in rows),
        "agent_messages": sum(row["agent_messages"] for row in rows),
        "reasoning_items": sum(row["reasoning_items"] for row in rows),
        "reasoning_characters": sum(row["reasoning_characters"] for row in rows),
        "attempts_with_captured_reasoning": sum(
            row["reasoning_items"] > 0 for row in rows
        ),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "cached_input_tokens": sum(row["cached_input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "reasoning_output_tokens": sum(row["reasoning_output_tokens"] for row in rows),
        "attempts_with_reasoning_output": sum(
            row["reasoning_output_tokens"] > 0 for row in rows
        ),
    }


def baseline_rows(path: Path, case_ids: list[int]) -> list[dict[str, Any]]:
    payload = load_json(path)
    if payload.get("cases") != case_ids:
        raise ValueError("baseline cases differ from candidate cases")
    rows = [row for row in payload.get("runs", []) if row.get("condition") == "tp2x1"]
    if not rows:
        raise ValueError("baseline has no tp2x1 runs")
    normalized = []
    for row in rows:
        prediction = row.get("prediction")
        expected = row.get("expected") or []
        false_positive_count, false_negative_count = false_counts(prediction, expected)
        normalized.append(
            {
                **row,
                "completed_within_limit": row.get("completed_within_60m", not row.get("timeout")),
                "runner_failures": 0,
                "false_positive_count": false_positive_count,
                "false_negative_count": false_negative_count,
                "agent_messages": 0,
                "reasoning_items": 0,
                "reasoning_characters": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
            }
        )
    return normalized


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = row.copy()
            item["prediction"] = json.dumps(item["prediction"], ensure_ascii=False)
            item["expected"] = json.dumps(item["expected"], ensure_ascii=False)
            writer.writerow(item)


def report_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Qwen3.6-27B 完整 Agent 验证",
        "",
        f"- 模型：`{summary['model']}`",
        "- 方法：原始 Codex CLI Agent runner，允许读取离线 `saved_configs` 并执行完整工具循环。",
        "- 部署：单个 vLLM TP=2 实例，2 个 Agent worker，总并发 2。",
        f"- Thinking：已显式请求，reasoning effort=`{summary['thinking']['reasoning_effort']}`；"
        f"原始 reasoning 已回填 {overall['attempts_with_captured_reasoning']}/{overall['attempts']} 次、"
        f"共 {overall['reasoning_items']} 个节点 / {overall['reasoning_characters']} 字符；"
        f"provider token 字段另报 {overall['reasoning_output_tokens']} tokens。",
        f"- 单次硬上限：{summary['timeout_seconds']} 秒；超时和 runner 失败均按错误计。",
        "- 严格判分：最终 `<result>` 中的 JSON 列表必须与独立 label 完全一致。",
        "",
        "| 题号 | 运行结果 | 严格正确 | 准确率 | 平均封顶耗时/分 | 超时 |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for case_id in summary["case_ids"]:
        item = summary["per_case"][str(case_id)]
        symbols = {"correct": "✅", "wrong": "❌", "timeout": "⏱"}
        runs = " ".join(symbols[value] for value in item["runs"])
        lines.append(
            f"| {case_id} | {runs} | {item['strict_correct']}/{item['attempts']} | "
            f"{item['accuracy_percent']:.2f}% | {item['runtime_minutes']['mean']:.2f} | "
            f"{item['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "| 汇总 | 数值 |",
            "|:---|---:|",
            f"| 严格正确率 | {overall['strict_correct']}/{overall['attempts']} ({overall['accuracy_percent']:.2f}%) |",
            f"| 60 分钟内完成 | {overall['completed_within_limit']}/{overall['attempts']} |",
            f"| 超时 / runner 失败 | {overall['timeouts']} / {overall['runner_failures']} |",
            f"| false positive / false negative | {overall['false_positives']} / {overall['false_negatives']} |",
            f"| 耗时均值 / 中位数 / P95 | {overall['runtime_minutes']['mean']:.2f} / {overall['runtime_minutes']['median']:.2f} / {overall['runtime_minutes']['p95']:.2f} 分钟 |",
        ]
    )
    comparison = summary.get("baseline_comparison")
    if comparison:
        base = comparison["baseline"]
        candidate = comparison["candidate"]
        lines.extend(
            [
                "",
                "## 与已归档 base-eval 的同条件 A/B",
                "",
                "| 条件 | 严格正确 | 准确率 | 超时 | 平均封顶耗时/分 |",
                "|:---|---:|---:|---:|---:|",
                f"| Base（复用） | {base['strict_correct']}/{base['attempts']} | {base['accuracy_percent']:.2f}% | {base['timeouts']} | {base['runtime_minutes']['mean']:.2f} |",
                f"| LoRA | {candidate['strict_correct']}/{candidate['attempts']} | {candidate['accuracy_percent']:.2f}% | {candidate['timeouts']} | {candidate['runtime_minutes']['mean']:.2f} |",
                f"| LoRA - Base | {comparison['strict_correct_delta']:+d} | {comparison['accuracy_point_delta']:+.2f} pp | {comparison['timeout_delta']:+d} | {comparison['mean_runtime_minutes_delta']:+.2f} |",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.timeout_seconds <= 0:
        raise ValueError("repeats and timeout must be positive")
    if len(set(args.case_ids)) != len(args.case_ids):
        raise ValueError("case ids must be unique")
    expected_by_case = load_expected(args.dataset.resolve())
    missing = [case_id for case_id in args.case_ids if case_id not in expected_by_case]
    if missing:
        raise ValueError(f"case ids absent from dataset: {missing}")

    rows = [
        parse_attempt(
            args.output_root.resolve(),
            args.run_prefix,
            case_id,
            repeat,
            expected_by_case[case_id],
            args.timeout_seconds,
            args.model,
        )
        for repeat in range(1, args.repeats + 1)
        for case_id in args.case_ids
    ]
    overall = aggregate(rows)
    summary: dict[str, Any] = {
        "schema_version": "qwen36-codex-agent-validation.v1",
        # A runner failure is an infrastructure-level incomplete slot, not an
        # incorrect Agent answer.  Keep the row for audit, but force the
        # orchestration layer to retry before accepting this summary.
        "status": "completed" if overall["runner_failures"] == 0 else "incomplete",
        "evaluation_method": "full_codex_agent_with_tools",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "git_commit": args.git_commit,
        "dataset": str(args.dataset.resolve()),
        "case_ids": args.case_ids,
        "repeats_per_case": args.repeats,
        "timeout_seconds": args.timeout_seconds,
        "thinking": {
            "requested": True,
            "reasoning_effort": args.reasoning_effort,
            "verification": (
                "raw reasoning items are restored from the matching Codex session rollout "
                "into events.jsonl and counted per attempt"
            ),
        },
        "topology": {
            "instance_count": 1,
            "tensor_parallel_size": 2,
            "worker_count": 2,
            "request_concurrency": 2,
        },
        "scoring": (
            "exact accepted-answer equality; normally parsed from one <result> JSON list; "
            "when that wrapper is wholly absent, one unique non-conflicting fenced exact "
            "match may be recovered; timeouts and infrastructure runner failures require retry"
        ),
        "overall": overall,
        "counts": {
            "attempts": overall["attempts"],
            "strict_correct": overall["strict_correct"],
            "timeouts": overall["timeouts"],
            "runner_failures": overall["runner_failures"],
        },
        "per_case": {},
        "runs": rows,
    }
    for case_id in args.case_ids:
        selected = [row for row in rows if row["case_id"] == case_id]
        summary["per_case"][str(case_id)] = {
            **aggregate(selected),
            "runs": [row["result"] for row in selected],
        }

    if args.baseline_summary:
        base = aggregate(baseline_rows(args.baseline_summary.resolve(), args.case_ids))
        if base["attempts"] != overall["attempts"]:
            raise ValueError("baseline and candidate attempt counts differ")
        summary["baseline_comparison"] = {
            "source": str(args.baseline_summary.resolve()),
            "baseline": base,
            "candidate": overall,
            "strict_correct_delta": overall["strict_correct"] - base["strict_correct"],
            "accuracy_point_delta": overall["accuracy_percent"] - base["accuracy_percent"],
            "timeout_delta": overall["timeouts"] - base["timeouts"],
            "mean_runtime_minutes_delta": (
                overall["runtime_minutes"]["mean"] - base["runtime_minutes"]["mean"]
            ),
        }

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "attempts.csv", rows)
    (report_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(
        f"Agent validation: {overall['strict_correct']}/{overall['attempts']} "
        f"({overall['accuracy_percent']:.2f}%)"
    )


if __name__ == "__main__":
    main()
