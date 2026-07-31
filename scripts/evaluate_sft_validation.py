#!/usr/bin/env python3
"""Evaluate a fixed SFT validation split through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RESULT_RE = re.compile(r"<result>\s*([\s\S]*?)\s*</result>")
LEAK_MARKERS = (
    "tool_call",
    "tool_response",
    "webfetch",
    "restore_tool_result",
    "powershell",
    "saved_configs",
    "调用工具",
    "调用接口",
    "执行命令",
    "读取文件",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--request-concurrency", type=int, default=2)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--git-commit")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: validation dataset is empty")
    return rows


def parse_result(text: str) -> list[str] | None:
    matches = RESULT_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return sorted(set(value))


def percentile(values: list[float], proportion: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def request_completion(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    identifier = row.get("id")
    messages = row.get("messages")
    if (
        not isinstance(identifier, str)
        or not isinstance(messages, list)
        or len(messages) < 2
        or messages[-1].get("role") != "assistant"
    ):
        raise ValueError("validation row is malformed")
    expected_text = messages[-1].get("content")
    if not isinstance(expected_text, str):
        raise ValueError(f"{identifier}: expected assistant text is missing")
    expected = parse_result(expected_text)
    if not expected:
        raise ValueError(f"{identifier}: expected result is malformed")
    request_messages = messages[:-1]
    payload = {
        "model": args.model,
        "messages": request_messages,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()
    response: dict[str, Any] | None = None
    error: str | None = None
    attempts = 0
    for attempts in range(1, args.retries + 2):
        try:
            request = urllib.request.Request(
                url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(
                request, timeout=args.timeout_seconds
            ) as handle:
                candidate = json.loads(handle.read().decode("utf-8"))
            if not isinstance(candidate, dict):
                raise TypeError("response is not a JSON object")
            response = candidate
            error = None
            break
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            TypeError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempts <= args.retries:
                time.sleep(min(2**attempts, 10))
    duration = time.perf_counter() - started

    if response is None:
        return {
            "id": identifier,
            "status": "request_failed",
            "attempts": attempts,
            "error": error,
            "duration_seconds": duration,
            "expected_result_items": expected,
        }

    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("response content is not a string")
    except (KeyError, IndexError, TypeError) as exc:
        return {
            "id": identifier,
            "status": "response_malformed",
            "attempts": attempts,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": duration,
            "expected_result_items": expected,
            "response": response,
        }

    actual = parse_result(content)
    lowered = content.lower()
    leak_hits = sorted(
        marker for marker in LEAK_MARKERS if marker.lower() in lowered
    )
    usage = response.get("usage")
    return {
        "id": identifier,
        "status": "completed",
        "attempts": attempts,
        "duration_seconds": duration,
        "finish_reason": choice.get("finish_reason"),
        "expected_result_items": expected,
        "actual_result_items": actual,
        "format_valid": actual is not None,
        "exact_match": actual == expected,
        "leak_marker_hits": leak_hits,
        "response_text": content,
        "usage": usage if isinstance(usage, dict) else None,
    }


def main() -> None:
    args = parse_args()
    if (
        args.instance_count != 1
        or args.workers != 2
        or args.request_concurrency != 2
    ):
        raise ValueError(
            "Qwen3.6-27B evaluation requires exactly one instance, "
            "two workers, and request concurrency two"
        )
    if args.max_tokens != 8000:
        raise ValueError("Qwen3.6-27B evaluation max_tokens must be 8000")

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(dataset)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        results = list(
            executor.map(
                lambda row: request_completion(row=row, args=args),
                rows,
            )
        )
    duration = time.perf_counter() - started

    predictions_path = output_dir / "validation_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    completed = [item for item in results if item["status"] == "completed"]
    durations = [
        float(item["duration_seconds"])
        for item in completed
        if isinstance(item.get("duration_seconds"), (int, float))
    ]
    exact = sum(item.get("exact_match") is True for item in completed)
    format_valid = sum(item.get("format_valid") is True for item in completed)
    leak_free = sum(not item.get("leak_marker_hits") for item in completed)
    summary = {
        "schema_version": "qwen36-sft-validation-eval.v1",
        "git_commit": args.git_commit,
        "checkpoint": args.checkpoint,
        "dataset": dataset.as_posix(),
        "model": args.model,
        "topology": {
            "instance_count": args.instance_count,
            "worker_count": args.workers,
            "request_concurrency": args.request_concurrency,
        },
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "max_tokens": args.max_tokens,
        },
        "counts": {
            "total": len(results),
            "completed": len(completed),
            "failed": len(results) - len(completed),
            "format_valid": format_valid,
            "exact_match": exact,
            "leak_free": leak_free,
        },
        "rates": {
            "format_valid": format_valid / len(results),
            "exact_match": exact / len(results),
            "leak_free": leak_free / len(results),
        },
        "latency_seconds": {
            "total": duration,
            "mean": statistics.fmean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p95": percentile(durations, 0.95) if durations else None,
        },
        "predictions": predictions_path.as_posix(),
    }
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Validation requests: {len(completed)}/{len(results)} completed")
    print(f"Format valid: {format_valid}/{len(results)}")
    print(f"Exact match: {exact}/{len(results)}")
    print(f"Leak free: {leak_free}/{len(results)}")
    print(f"Summary: {summary_path}")
    if len(completed) != len(results):
        raise RuntimeError("one or more validation requests failed")


if __name__ == "__main__":
    main()
