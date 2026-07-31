#!/usr/bin/env python3
"""Summarize an ms-swift run and verify its best checkpoint by eval loss."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--print-best-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def state_rank(path: Path) -> tuple[int, int]:
    state = load_json(path)
    step = state.get("global_step")
    return (int(step) if isinstance(step, int) else -1, path.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    state_paths = sorted(output_dir.rglob("trainer_state.json"))
    if not state_paths:
        raise FileNotFoundError(f"No trainer_state.json found below {output_dir}")
    state_path = max(state_paths, key=state_rank)
    state = load_json(state_path)
    history = state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("trainer state has no log_history")

    evaluations: list[dict[str, int | float]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        loss = item.get("eval_loss")
        step = item.get("step")
        if (
            isinstance(loss, (int, float))
            and math.isfinite(float(loss))
            and isinstance(step, int)
        ):
            evaluation: dict[str, int | float] = {
                "step": step,
                "eval_loss": float(loss),
            }
            epoch = item.get("epoch")
            if isinstance(epoch, (int, float)) and math.isfinite(float(epoch)):
                evaluation["epoch"] = float(epoch)
            evaluations.append(evaluation)
    if not evaluations:
        raise ValueError("trainer state contains no finite eval_loss values")

    minimum = min(evaluations, key=lambda item: float(item["eval_loss"]))
    best_metric = state.get("best_metric")
    best_checkpoint_value = state.get("best_model_checkpoint")
    if not isinstance(best_metric, (int, float)) or not math.isfinite(
        float(best_metric)
    ):
        raise ValueError("trainer state best_metric is missing or not finite")
    if not isinstance(best_checkpoint_value, str) or not best_checkpoint_value:
        raise ValueError("trainer state best_model_checkpoint is missing")
    if not math.isclose(
        float(best_metric),
        float(minimum["eval_loss"]),
        rel_tol=1e-6,
        abs_tol=5e-8,
    ):
        raise ValueError(
            "best_metric does not equal the minimum observed validation loss"
        )

    best_checkpoint = Path(best_checkpoint_value)
    if not best_checkpoint.is_absolute():
        best_checkpoint = output_dir / best_checkpoint
    best_checkpoint = best_checkpoint.resolve()
    if not best_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint directory does not exist: {best_checkpoint}"
        )
    checkpoint_match = re.search(r"checkpoint-(\d+)$", best_checkpoint.name)
    if (
        checkpoint_match is None
        or int(checkpoint_match.group(1)) != int(minimum["step"])
    ):
        raise ValueError(
            "best checkpoint step does not equal the minimum validation-loss step"
        )

    summary = {
        "schema_version": "qwen36-lora-training-summary.v1",
        "status": "completed",
        "git_commit": args.git_commit,
        "output_dir": output_dir.as_posix(),
        "trainer_state": state_path.resolve().as_posix(),
        "global_step": state.get("global_step"),
        "best_model_checkpoint": best_checkpoint.as_posix(),
        "best_metric_name": "eval_loss",
        "best_metric": float(best_metric),
        "minimum_validation_loss": minimum,
        "validation_history": evaluations,
    }
    result_path = (
        args.result.resolve()
        if args.result is not None
        else output_dir / "training_summary.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.print_best_only:
        print(best_checkpoint.as_posix())
    else:
        print(f"Minimum validation loss: {best_metric}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Summary: {result_path}")


if __name__ == "__main__":
    main()
