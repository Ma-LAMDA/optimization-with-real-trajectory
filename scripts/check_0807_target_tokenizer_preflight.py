#!/usr/bin/env python3
"""Archive the real Qwen/ms-swift 16K length and per-token loss-mask preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "2026-08-07" / "sft"
DEFAULT_OUTPUT = SFT / "TARGET_TOKENIZER_PREFLIGHT.json"
DEFAULT_DATASETS = {
    "train_core_pool": SFT / "qwen3_6_27b_0807_core_pool.jsonl",
    "train_endpoint_pool": SFT / "qwen3_6_27b_0807_endpoint_pool.jsonl",
    "validation": SFT / "qwen3_6_27b_0807_validation.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset", action="append", default=[], metavar="NAME=PATH",
        help="Override defaults; repeat for each logical dataset.",
    )
    return parser.parse_args()


def digest_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def raw_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[int(ratio * (len(ordered) - 1))]


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    datasets = dict(DEFAULT_DATASETS)
    if args.dataset:
        datasets = {}
        for value in args.dataset:
            if "=" not in value:
                raise ValueError(f"invalid --dataset {value!r}; expected NAME=PATH")
            name, path = value.split("=", 1)
            if not name or name in datasets:
                raise ValueError(f"invalid or duplicate dataset name: {name!r}")
            datasets[name] = Path(path)

    try:
        import swift
        import transformers
        from swift import get_processor, get_template
    except ImportError as exc:
        raise SystemExit("ms-swift and transformers are required") from exc

    processor = get_processor(str(args.model))
    template = get_template(
        processor,
        max_length=None,
        loss_scale="default",
        is_binary_loss_scale=False,
    )
    template.set_mode("train")

    dataset_reports: dict[str, Any] = {}
    all_lengths: list[int] = []
    overlong: list[dict[str, Any]] = []
    mask_failures: list[dict[str, Any]] = []
    for name, path in datasets.items():
        rows = load_rows(path)
        lengths: list[int] = []
        supervised_tokens = 0
        weighted_tokens = 0.0
        scale_tokens: Counter[str] = Counter()
        for row in rows:
            encoded = template.encode(row)
            input_ids = encoded.get("input_ids")
            labels = encoded.get("labels")
            loss_scale = encoded.get("loss_scale")
            if not isinstance(input_ids, list) or not isinstance(labels, list) or not isinstance(loss_scale, list):
                raise ValueError(f"{row.get('id')}: template omitted token arrays")
            if not (len(input_ids) == len(labels) == len(loss_scale)):
                raise ValueError(f"{row.get('id')}: token array lengths differ")
            length = len(input_ids)
            lengths.append(length)
            all_lengths.append(length)
            if length > args.max_length:
                overlong.append({"dataset": name, "row_id": row.get("id"), "tokens": length})
            expected_scales = {
                round(float(message.get("loss_scale", 0) or 0), 8)
                for message in row["messages"]
                if float(message.get("loss_scale", 0) or 0) > 0
            }
            observed_scales: set[float] = set()
            for index, (label, scale) in enumerate(zip(labels, loss_scale)):
                scale_value = round(float(scale), 8)
                supervised = int(label) != -100
                if supervised != (scale_value > 0):
                    mask_failures.append({
                        "dataset": name, "row_id": row.get("id"),
                        "token_index": index, "label": int(label), "loss_scale": scale_value,
                    })
                if supervised:
                    supervised_tokens += 1
                    weighted_tokens += scale_value
                    scale_tokens[f"{scale_value:.8g}"] += 1
                    observed_scales.add(scale_value)
            if expected_scales - observed_scales:
                mask_failures.append({
                    "dataset": name, "row_id": row.get("id"),
                    "missing_message_loss_scales": sorted(expected_scales - observed_scales),
                })
        if not rows or not lengths or supervised_tokens <= 0:
            raise ValueError(f"{name}: empty dataset or no supervised tokens")
        dataset_reports[name] = {
            "path": path.resolve().as_posix(),
            "sha256_lf_normalized": digest_file(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
            "token_length": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "p95": percentile(lengths, 0.95),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths),
            },
            "supervised_tokens": supervised_tokens,
            "weighted_supervised_token_sum": round(weighted_tokens, 8),
            "supervised_token_count_by_loss_scale": dict(sorted(scale_tokens.items())),
        }

    tokenizer_files: dict[str, dict[str, Any]] = {}
    for filename in (
        "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
        "special_tokens_map.json", "chat_template.jinja", "config.json",
    ):
        path = args.model / filename
        if path.exists():
            tokenizer_files[filename] = {
                "bytes": path.stat().st_size,
                "sha256_raw": raw_digest(path),
            }
    if not tokenizer_files:
        raise ValueError("no tokenizer/model identity files found")

    passed = not overlong and not mask_failures
    report = {
        "schema_version": "0807-target-tokenizer-loss-mask-preflight.v1",
        "status": "passed" if passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "ms_swift_version": getattr(swift, "__version__", "unknown"),
        "transformers_version": transformers.__version__,
        "model_path": args.model.resolve().as_posix(),
        "model_identity_files": tokenizer_files,
        "template": {
            "mode": "train",
            "loss_scale": "default",
            "is_binary_loss_scale": False,
            "max_length_during_encode": None,
            "release_max_length": args.max_length,
        },
        "datasets": dataset_reports,
        "totals": {
            "rows": sum(record["rows"] for record in dataset_reports.values()),
            "min_tokens": min(all_lengths),
            "p99_tokens": percentile(all_lengths, 0.99),
            "max_tokens": max(all_lengths),
            "over_max_length_rows": len(overlong),
            "loss_mask_failures": len(mask_failures),
        },
        "overlong_rows": overlong[:100],
        "loss_mask_failures": mask_failures[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["totals"], ensure_ascii=False))
    print(f"report={args.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
