#!/usr/bin/env python3
"""Compose the 100x5 base Agent evaluation from audited component runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from summarize_agent_validation import aggregate


PRIMARY_CASE_IDS = [*range(1, 89), 91, 92, 93, 94]
HELDOUT_CASE_IDS = [89, 90, 95, 96, 97, 98, 99, 100]
ALL_CASE_IDS = list(range(1, 101))
REPLACED_TIMEOUT_SLOTS = {(89, 3), (90, 3), (99, 2)}
CURRENT_TOPOLOGY = {
    "instance_count": 1,
    "tensor_parallel_size": 2,
    "worker_count": 2,
    "request_concurrency": 2,
}
HISTORICAL_TOPOLOGY = {"worker_count": 8, "request_concurrency": 8}

CSV_FIELDS = [
    "model",
    "case_id",
    "repeat",
    "runner_status",
    "duration_seconds",
    "capped_minutes",
    "timeout",
    "completed_within_limit",
    "correct",
    "result",
    "false_positive_count",
    "false_negative_count",
    "events",
    "commands",
    "agent_messages",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "prediction",
    "expected",
    "artifact_dir",
    "source",
    "source_repeat",
    "replaces_historical_timeout",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--historical-attempts", type=Path, required=True)
    parser.add_argument("--replacement-summary", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_json_cell(value: str) -> Any:
    if not value or value.strip().lower() in {"null", "none"}:
        return None
    return json.loads(value)


def false_counts(prediction: Any, expected: Any) -> tuple[int, int]:
    prediction_set = set(prediction) if isinstance(prediction, list) else set()
    expected_set = set(expected) if isinstance(expected, list) else set()
    return len(prediction_set - expected_set), len(expected_set - prediction_set)


def validate_primary(payload: dict[str, Any]) -> None:
    if payload.get("evaluation_method") != "full_codex_agent_with_tools":
        raise ValueError("primary evaluation method is not full Codex Agent")
    if payload.get("case_ids") != PRIMARY_CASE_IDS:
        raise ValueError("primary case ids are not the audited 92-case scope")
    if payload.get("repeats_per_case") != 5:
        raise ValueError("primary repeats_per_case must be 5")
    if payload.get("counts", {}).get("attempts") != 460:
        raise ValueError("primary summary must contain 460 attempts")
    if payload.get("topology") != CURRENT_TOPOLOGY:
        raise ValueError("primary topology is not tp2x1/concurrency2")


def validate_replacements(payload: dict[str, Any]) -> None:
    replacement_cases = sorted(case_id for case_id, _ in REPLACED_TIMEOUT_SLOTS)
    if payload.get("evaluation_method") != "full_codex_agent_with_tools":
        raise ValueError("replacement evaluation method is not full Codex Agent")
    if sorted(payload.get("case_ids", [])) != replacement_cases:
        raise ValueError("replacement summary must contain cases 89, 90, and 99")
    if payload.get("repeats_per_case") != 1:
        raise ValueError("replacement repeats_per_case must be 1")
    if payload.get("counts", {}).get("attempts") != 3:
        raise ValueError("replacement summary must contain exactly 3 attempts")
    if payload.get("topology") != CURRENT_TOPOLOGY:
        raise ValueError("replacement topology is not tp2x1/concurrency2")


def normalize_current_row(
    row: dict[str, Any],
    *,
    source: str,
    repeat: int | None = None,
    replaces_timeout: bool = False,
) -> dict[str, Any]:
    normalized = {field: row.get(field) for field in CSV_FIELDS}
    normalized["case_id"] = int(row["case_id"])
    normalized["source_repeat"] = int(row["repeat"])
    normalized["repeat"] = int(repeat if repeat is not None else row["repeat"])
    normalized["source"] = source
    normalized["replaces_historical_timeout"] = replaces_timeout
    return normalized


def load_historical_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for source_row in csv.DictReader(handle):
            if source_row.get("model") != "base":
                continue
            case_id = int(source_row["case_id"])
            if case_id not in HELDOUT_CASE_IDS:
                continue
            repeat = int(source_row["repeat"])
            timeout = parse_bool(source_row["timeout"])
            correct = parse_bool(source_row["correct"])
            prediction = parse_json_cell(source_row["prediction"])
            expected = parse_json_cell(source_row["expected"])
            false_positive_count, false_negative_count = false_counts(
                prediction, expected
            )
            duration_seconds = float(source_row["duration_seconds"])
            capped_seconds = float(source_row["capped_duration_seconds"])
            rows.append(
                {
                    "model": source_row["model"],
                    "case_id": case_id,
                    "repeat": repeat,
                    "runner_status": source_row["runner_status"],
                    "duration_seconds": duration_seconds,
                    "capped_minutes": capped_seconds / 60,
                    "timeout": timeout,
                    "completed_within_limit": not timeout,
                    "correct": correct,
                    "result": "timeout" if timeout else ("correct" if correct else "wrong"),
                    "false_positive_count": false_positive_count,
                    "false_negative_count": false_negative_count,
                    "events": 0,
                    "commands": 0,
                    "agent_messages": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "prediction": prediction,
                    "expected": expected,
                    "artifact_dir": source_row["artifact_dir"],
                    "source": "historical_heldout8_non_timeout",
                    "source_repeat": repeat,
                    "replaces_historical_timeout": False,
                }
            )
    keys = {(row["case_id"], row["repeat"]) for row in rows}
    expected_keys = {
        (case_id, repeat)
        for case_id in HELDOUT_CASE_IDS
        for repeat in range(1, 6)
    }
    if len(rows) != 40 or keys != expected_keys:
        raise ValueError("historical base rows are not exactly the expected 8x5 attempts")
    timeout_keys = {
        (row["case_id"], row["repeat"]) for row in rows if row["timeout"]
    }
    if timeout_keys != REPLACED_TIMEOUT_SLOTS:
        raise ValueError(
            f"historical timeout slots changed: expected {sorted(REPLACED_TIMEOUT_SLOTS)}, "
            f"found {sorted(timeout_keys)}"
        )
    return rows


def validate_final_rows(rows: list[dict[str, Any]]) -> None:
    keys = [(int(row["case_id"]), int(row["repeat"])) for row in rows]
    expected_keys = [
        (case_id, repeat)
        for case_id in ALL_CASE_IDS
        for repeat in range(1, 6)
    ]
    if len(rows) != 500 or sorted(keys) != expected_keys:
        raise ValueError("composed rows do not form an exact 100x5 evaluation")
    if len(set(keys)) != 500:
        raise ValueError("composed rows contain duplicate case/repeat slots")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            item = {field: row.get(field) for field in CSV_FIELDS}
            item["prediction"] = json.dumps(item["prediction"], ensure_ascii=False)
            item["expected"] = json.dumps(item["expected"], ensure_ascii=False)
            writer.writerow(item)


def report_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    symbols = {"correct": "✅", "wrong": "❌", "timeout": "⏱"}
    lines = [
        "# Qwen3.6-27B Base 100×5 完整 Agent 评测",
        "",
        "- 方法：完整 Codex CLI Agent runner，可读取离线 saved_configs 并执行工具循环。",
        "- 组合：当前单实例 TP=2/双并发 460 次；旧 8 并发留出题非超时结果 37 次；",
        "  单实例 TP=2/双并发替代运行 3 次。替代运行覆盖旧记录中的 q89-r3、q90-r3、q99-r2。",
        "- 注意：这是经用户确认允许历史非超时结果复用的组合结论，不代表全部 500 次都在双并发下运行。",
        f"- 单次上限：{summary['timeout_seconds']} 秒；超时和 runner 失败均按错误计。",
        "- 严格判分：最终 `<result>` 中 JSON 列表必须与 label 完全一致。",
        "",
        "| 题号 | 五次结果 | 严格正确 | 准确率 | 平均封顶耗时/分 | 超时 |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for case_id in summary["case_ids"]:
        item = summary["per_case"][str(case_id)]
        run_symbols = " ".join(symbols[result] for result in item["runs"])
        lines.append(
            f"| {case_id} | {run_symbols} | {item['strict_correct']}/{item['attempts']} | "
            f"{item['accuracy_percent']:.2f}% | {item['runtime_minutes']['mean']:.2f} | "
            f"{item['timeouts']} |"
        )
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            "| 指标 | 数值 |",
            "|:---|---:|",
            f"| 严格正确率 | {overall['strict_correct']}/{overall['attempts']} ({overall['accuracy_percent']:.2f}%) |",
            f"| 60 分钟内结束 | {overall['completed_within_limit']}/{overall['attempts']} |",
            f"| 超时 / runner 失败 | {overall['timeouts']} / {overall['runner_failures']} |",
            f"| false positive / false negative | {overall['false_positives']} / {overall['false_negatives']} |",
            f"| 耗时均值 / 中位数 / P95 | {overall['runtime_minutes']['mean']:.2f} / "
            f"{overall['runtime_minutes']['median']:.2f} / {overall['runtime_minutes']['p95']:.2f} 分钟 |",
            "",
            "事件、命令和 token 总量只覆盖仍保留完整结构化清单的当前运行；旧 37 次 CSV 复用结果的这些字段记为 0，",
            "因此不得把组合报告中的这些总量当作 500 次完整资源统计。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    primary_path = args.primary_summary.resolve()
    historical_path = args.historical_attempts.resolve()
    replacement_path = args.replacement_summary.resolve()
    primary = load_json(primary_path)
    replacements = load_json(replacement_path)
    validate_primary(primary)
    validate_replacements(replacements)

    rows = [
        normalize_current_row(row, source="current_tp2x1_concurrency2")
        for row in primary["runs"]
    ]
    historical_rows = load_historical_rows(historical_path)
    for row in historical_rows:
        row["model"] = primary["model"]
    rows.extend(
        row
        for row in historical_rows
        if (row["case_id"], row["repeat"]) not in REPLACED_TIMEOUT_SLOTS
    )

    replacement_by_case = {
        int(row["case_id"]): row for row in replacements["runs"]
    }
    for case_id, target_repeat in sorted(REPLACED_TIMEOUT_SLOTS):
        rows.append(
            normalize_current_row(
                replacement_by_case[case_id],
                source="timeout_replacement_tp2x1_concurrency2",
                repeat=target_repeat,
                replaces_timeout=True,
            )
        )

    rows.sort(key=lambda row: (int(row["case_id"]), int(row["repeat"])))
    validate_final_rows(rows)
    overall = aggregate(rows)
    summary: dict[str, Any] = {
        "schema_version": "qwen36-codex-agent-validation.composite.v1",
        "status": "completed",
        "evaluation_method": "full_codex_agent_with_tools",
        "model": primary["model"],
        "checkpoint": primary.get("checkpoint", ""),
        "git_commit": primary.get("git_commit", ""),
        "dataset": primary["dataset"],
        "case_ids": ALL_CASE_IDS,
        "repeats_per_case": 5,
        "timeout_seconds": primary["timeout_seconds"],
        "topology": {
            "mode": "composite_with_audited_historical_reuse",
            "current_and_replacement_runs": CURRENT_TOPOLOGY,
            "historical_reused_runs": HISTORICAL_TOPOLOGY,
        },
        "composition": {
            "primary": {
                "summary": str(primary_path),
                "attempts": 460,
                "case_ids": PRIMARY_CASE_IDS,
                "git_commit": primary.get("git_commit", ""),
            },
            "historical_reuse": {
                "attempts": 37,
                "attempts_csv": str(historical_path),
                "case_ids": HELDOUT_CASE_IDS,
                "excluded_timeout_slots": [
                    list(slot) for slot in sorted(REPLACED_TIMEOUT_SLOTS)
                ],
            },
            "timeout_replacements": {
                "summary": str(replacement_path),
                "attempts": 3,
                "git_commit": replacements.get("git_commit", ""),
                "slot_mapping": {
                    "89": {"source_repeat": 1, "target_repeat": 3},
                    "90": {"source_repeat": 1, "target_repeat": 3},
                    "99": {"source_repeat": 1, "target_repeat": 2},
                },
            },
        },
        "scoring": "exact JSON list equality; timeout and runner failure are incorrect",
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
    for case_id in ALL_CASE_IDS:
        selected = [row for row in rows if row["case_id"] == case_id]
        summary["per_case"][str(case_id)] = {
            **aggregate(selected),
            "runs": [row["result"] for row in selected],
        }

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "attempts.csv", rows)
    (report_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "report.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(
        f"Composed full base Agent eval: {overall['strict_correct']}/500 "
        f"({overall['accuracy_percent']:.2f}%), timeouts={overall['timeouts']}, "
        f"runner_failures={overall['runner_failures']}"
    )


if __name__ == "__main__":
    main()
