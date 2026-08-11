#!/usr/bin/env python3
"""Archive the real Qwen/ms-swift 16K length and loss-mask preflight for 0809."""

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
DEFAULT_DATA_ROOT = ROOT / "data" / "2026-08-09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override defaults; repeat for each logical dataset.",
    )
    return parser.parse_args()


def resolve_data_root(value: Path) -> Path:
    path = value if value.is_absolute() else ROOT / value
    path = path.resolve()
    expected = (ROOT / "data" / "2026-08-09").resolve()
    if path != expected:
        raise ValueError(f"preflight expected {expected}, received {path}")
    return path


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
    data_root = resolve_data_root(args.data_root)
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    sft = data_root / "sft"
    output = args.output or sft / "TARGET_TOKENIZER_PREFLIGHT.json"
    datasets = {
        "train_semantic_pool": sft / "qwen3_6_27b_0809_train_semantic_pool.jsonl",
        "validation": sft / "qwen3_6_27b_0809_validation.jsonl",
    }
    for epoch in range(1, 6):
        datasets[f"core_epoch_{epoch:02d}"] = (
            sft / f"qwen3_6_27b_0809_core_epoch_{epoch:02d}.jsonl"
        )
        datasets[f"endpoint_epoch_{epoch:02d}"] = (
            sft / f"qwen3_6_27b_0809_endpoint_epoch_{epoch:02d}.jsonl"
        )
    if args.dataset:
        datasets = {}
        for value in args.dataset:
            if "=" not in value:
                raise ValueError(f"invalid --dataset {value!r}; expected NAME=PATH")
            name, raw_path = value.split("=", 1)
            path = Path(raw_path)
            path = path if path.is_absolute() else ROOT / path
            path = path.resolve()
            if not name or name in datasets:
                raise ValueError(f"invalid or duplicate dataset name: {name!r}")
            if data_root not in path.parents:
                raise ValueError(f"dataset {name} is outside resolved DATA_ROOT: {path}")
            datasets[name] = path

    try:
        import swift
        import transformers
        from swift import get_processor, get_template
    except ImportError as exc:
        raise SystemExit("ms-swift and transformers are required") from exc

    processor = get_processor(str(args.model.resolve()))
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
        source_rows = load_rows(path)
        lengths: list[int] = []
        supervised_tokens = 0
        weighted_tokens = 0.0
        scale_tokens: Counter[str] = Counter()
        weighted_by_target: Counter[str] = Counter()
        weighted_by_label: Counter[str] = Counter()
        weighted_by_target_label: Counter[str] = Counter()
        for row in source_rows:
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
            row_weighted_tokens = 0.0
            for index, (label, scale) in enumerate(zip(labels, loss_scale)):
                scale_value = round(float(scale), 8)
                supervised = int(label) != -100
                if supervised != (scale_value > 0):
                    mask_failures.append(
                        {
                            "dataset": name,
                            "row_id": row.get("id"),
                            "token_index": index,
                            "label": int(label),
                            "loss_scale": scale_value,
                        }
                    )
                if supervised:
                    supervised_tokens += 1
                    weighted_tokens += scale_value
                    row_weighted_tokens += scale_value
                    scale_tokens[f"{scale_value:.8g}"] += 1
                    observed_scales.add(scale_value)
            if expected_scales - observed_scales:
                mask_failures.append(
                    {
                        "dataset": name,
                        "row_id": row.get("id"),
                        "missing_message_loss_scales": sorted(expected_scales - observed_scales),
                    }
                )
            target_type = str(row.get("metadata", {}).get("target_type") or "unknown")
            label_values = row.get("metadata", {}).get("correct_labels") or []
            label_name = (
                str(label_values[0])
                if isinstance(label_values, list) and len(label_values) == 1
                else "unknown"
            )
            weighted_by_target[target_type] += row_weighted_tokens
            weighted_by_label[label_name] += row_weighted_tokens
            weighted_by_target_label[f"{target_type}||{label_name}"] += row_weighted_tokens
        if not source_rows or not lengths or supervised_tokens <= 0:
            raise ValueError(f"{name}: empty dataset or no supervised tokens")
        dataset_reports[name] = {
            "path": path.as_posix(),
            "sha256_lf_normalized": digest_file(path),
            "bytes": path.stat().st_size,
            "rows": len(source_rows),
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
            "weighted_supervised_token_sum_by_target": {
                key: round(value, 8) for key, value in sorted(weighted_by_target.items())
            },
            "weighted_supervised_token_sum_by_label": {
                key: round(value, 8) for key, value in sorted(weighted_by_label.items())
            },
            "weighted_supervised_token_sum_by_target_label": {
                key: round(value, 8)
                for key, value in sorted(weighted_by_target_label.items())
            },
        }

    tokenizer_files: dict[str, dict[str, Any]] = {}
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ):
        path = args.model.resolve() / filename
        if path.exists():
            tokenizer_files[filename] = {
                "bytes": path.stat().st_size,
                "sha256_raw": raw_digest(path),
            }
    if not tokenizer_files:
        raise ValueError("no tokenizer/model identity files found")

    passed = not overlong and not mask_failures
    report = {
        "schema_version": "0809-target-tokenizer-loss-mask-preflight.v1",
        "status": "passed" if passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_data_root": data_root.as_posix(),
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
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    if data_root not in output.parents:
        raise ValueError(f"preflight output is outside resolved DATA_ROOT: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"resolved_data_root={data_root}")
    print(json.dumps(report["totals"], ensure_ascii=False))
    print(f"report={output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
