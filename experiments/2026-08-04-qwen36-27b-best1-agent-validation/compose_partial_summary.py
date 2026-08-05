#!/usr/bin/env python3
"""Compose uneven completed Agent-validation fragments after a graceful stop."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fragments",
        nargs="+",
        help="Summary spec PATH@REPEAT_OFFSET; offset is added to every run repeat.",
    )
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheduled-attempts", type=int, default=60)
    parser.add_argument("--stop-note", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_fragment_spec(spec: str) -> tuple[Path, int]:
    path_text, separator, offset_text = spec.rpartition("@")
    if not separator:
        raise ValueError(f"fragment lacks @REPEAT_OFFSET: {spec}")
    return Path(path_text), int(offset_text)


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
        "model_hard_timeouts": sum(row["timeout"] for row in rows),
        "infrastructure_failures": 0,
        "interruptions_recorded": 0,
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
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "reasoning_output_tokens": sum(row["reasoning_output_tokens"] for row in rows),
        "attempts_with_reasoning_output": sum(
            row["reasoning_output_tokens"] > 0 for row in rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = row.copy()
            item["prediction"] = json.dumps(item["prediction"], ensure_ascii=False)
            item["expected"] = json.dumps(item["expected"], ensure_ascii=False)
            writer.writerow(item)


def report_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    symbols = {"correct": "✅", "wrong": "❌", "timeout": "⏱"}
    lines = [
        "# 0804 best1 LoRA 部分 Agent 验证",
        "",
        f"- 状态：`{summary['status']}`；{summary['stop']['note']}",
        f"- 已执行：{overall['attempts']}/{summary['scheduled_attempts']}；"
        f"未启动：{summary['unstarted_attempts']}（不计失败）。",
        f"- 严格正确：{overall['strict_correct']}/{overall['attempts']} "
        f"({overall['accuracy_percent']:.2f}%)。",
        f"- 模型硬超时：{overall['model_hard_timeouts']}；基础设施失败：0；"
        "中断记录：0。",
        "- Thinking：Codex CLI 显式请求 `high`；本地 provider 未提供独立的 "
        "reasoning token 计数，不能据此推断未启用 thinking。",
        "- 判分：最终 `<result>` JSON 列表与参考答案完全一致；模型硬超时计错，"
        "基础设施失败和中断不入表。",
        "",
        "| 题号 | 已执行结果 | 严格正确 | 准确率 | 模型硬超时 |",
        "|---:|:---:|---:|---:|---:|",
    ]
    for case_id in summary["case_ids"]:
        item = summary["per_case"][str(case_id)]
        result_text = " ".join(symbols[result] for result in item["runs"])
        lines.append(
            f"| {case_id} | {result_text} | {item['strict_correct']}/{item['attempts']} | "
            f"{item['accuracy_percent']:.2f}% | {item['model_hard_timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## 主要观察",
            "",
            "- q85、q86 各命中 2/3，但第 3 次把两个核心设备同时输出，设备集合过宽。",
            "- q99 能找到 `preempt disabled` 证据，却映射成“VRRP Master 角色规划不合理”。",
            "- q2、q12 经常绕到 OSPF/ISIS/BGP 路径，未收敛到“全局 STP 未使能”。",
            "- 主要短板是过度搜索、停止判断不足，以及标签/设备集合映射不稳定。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fragments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    base: dict[str, Any] | None = None
    for spec in args.fragments:
        path, offset = parse_fragment_spec(spec)
        payload = load_json(path)
        if base is None:
            base = payload
        fragments.append(
            {
                "path": path.as_posix(),
                "repeat_offset": offset,
                "attempts": len(payload["runs"]),
            }
        )
        for source_row in payload["runs"]:
            row = source_row.copy()
            row["repeat"] = int(row["repeat"]) + offset
            rows.append(row)
    if base is None or not rows:
        raise ValueError("no fragment rows")
    rows.sort(key=lambda row: (int(row["case_id"]), int(row["repeat"])))
    keys = [(int(row["case_id"]), int(row["repeat"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case/repeat slots")
    if any(row["runner_status"] == "interrupted" for row in rows):
        raise ValueError("interrupted attempt must not be archived")

    case_ids = sorted({int(row["case_id"]) for row in rows})
    overall = aggregate(rows)
    if args.scheduled_attempts < overall["attempts"]:
        raise ValueError("scheduled attempts are fewer than completed attempts")
    per_case: dict[str, Any] = {}
    for case_id in case_ids:
        selected = [row for row in rows if int(row["case_id"]) == case_id]
        per_case[str(case_id)] = {
            **aggregate(selected),
            "repeats": [int(row["repeat"]) for row in selected],
            "runs": [row["result"] for row in selected],
        }

    summary = {
        "schema_version": "qwen36-codex-agent-partial-validation.v1",
        "status": "stopped_by_user_after_inflight_attempts",
        "evaluation_method": base["evaluation_method"],
        "model": base["model"],
        "checkpoint": base["checkpoint"],
        "git_commit": base["git_commit"],
        "dataset": base["dataset"],
        "case_ids": case_ids,
        "scheduled_attempts": args.scheduled_attempts,
        "unstarted_attempts": args.scheduled_attempts - overall["attempts"],
        "timeout_seconds": base["timeout_seconds"],
        "thinking": {
            **base["thinking"],
            "provider_limitation": (
                "The local provider reports reasoning_output_tokens=0; explicit high reasoning "
                "configuration is the auditable thinking-on signal."
            ),
        },
        "topology": base["topology"],
        "scoring": (
            "exact JSON list equality; model hard timeout is incorrect; infrastructure "
            "failure and interruption are excluded"
        ),
        "stop": {
            "note": args.stop_note,
            "inflight_slots_completed": [[12, 4], [19, 4]],
            "last_completed_slot": [19, 4],
        },
        "fragments": fragments,
        "overall": overall,
        "per_case": per_case,
        "runs": rows,
    }

    training = load_json(args.training_summary)
    workflow = {
        "schema_version": "qwen36-0804-best1-partial-workflow.v1",
        "status": summary["status"],
        "run_id": "20260804T141553Z",
        "training": training,
        "validation": {
            "attempts": overall["attempts"],
            "scheduled_attempts": args.scheduled_attempts,
            "strict_correct": overall["strict_correct"],
            "accuracy_percent": overall["accuracy_percent"],
            "model_hard_timeouts": overall["model_hard_timeouts"],
            "infrastructure_failures": 0,
            "unstarted_attempts": summary["unstarted_attempts"],
        },
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "attempts.csv", rows)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "workflow_summary.json").write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    print(
        f"Partial Agent validation: {overall['strict_correct']}/{overall['attempts']} "
        f"({overall['accuracy_percent']:.2f}%), unstarted={summary['unstarted_attempts']}"
    )


if __name__ == "__main__":
    main()
