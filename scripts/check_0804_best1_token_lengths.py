#!/usr/bin/env python3
"""Tokenize the 0804 best1 dataset with the target ms-swift template and enforce its limit."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "2026-08-04" / "sft" / "qwen3_6_27b_reasoning_trajectory_best1_train.jsonl"
DEFAULT_VALIDATION = ROOT / "data" / "2026-08-04" / "sft" / "qwen3_6_27b_reasoning_trajectory_best1_validation.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    try:
        from swift import get_processor, get_template
    except ImportError as exc:
        raise SystemExit("ms-swift is required for tokenizer preflight") from exc

    processor = get_processor(args.model)
    template = get_template(
        processor,
        max_length=None,
        loss_scale="default",
        is_binary_loss_scale=False,
    )
    template.set_mode("train")

    overlong: list[tuple[str, str, int]] = []
    all_lengths: list[int] = []
    for split, path in (("train", args.train), ("validation", args.validation)):
        lengths: list[int] = []
        for row in load_rows(path):
            encoded = template.encode(row)
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise ValueError(f"{row.get('id')}: template returned no input_ids")
            length = len(input_ids)
            lengths.append(length)
            all_lengths.append(length)
            if length > args.max_length:
                overlong.append((split, str(row.get("id")), length))
        ordered = sorted(lengths)
        p95 = ordered[int(0.95 * (len(ordered) - 1))]
        print(
            f"{split}: rows={len(lengths)} min={min(lengths)} "
            f"median={statistics.median(lengths):.1f} p95={p95} max={max(lengths)}"
        )

    if overlong:
        print(f"Overlong samples (> {args.max_length}): {len(overlong)}")
        for split, row_id, length in sorted(overlong, key=lambda item: -item[2])[:50]:
            print(f"- {split} {row_id}: {length}")
        raise SystemExit(1)
    print(f"All {len(all_lengths)} samples fit max_length={args.max_length}")


if __name__ == "__main__":
    main()
