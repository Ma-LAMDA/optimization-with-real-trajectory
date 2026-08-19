"""Force and audit the fixed learning rate for one resumed 0805 training stage."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required by the 0805 fixed-stage LR plugin")
    return value


STAGE = int(_required_env("QWEN36_0805_TRAIN_STAGE"))
TARGET_LR = float(_required_env("QWEN36_0805_TARGET_LR"))
AUDIT_PATH = Path(_required_env("QWEN36_0805_LR_AUDIT_PATH"))
PROCESS_RANK = int(os.environ.get("RANK", "0"))

if STAGE not in range(1, 6):
    raise RuntimeError(f"QWEN36_0805_TRAIN_STAGE must be 1..5, found {STAGE}")
if not math.isfinite(TARGET_LR) or TARGET_LR <= 0:
    raise RuntimeError(f"QWEN36_0805_TARGET_LR must be positive, found {TARGET_LR}")


def _optimizer_lrs(optimizer) -> list[float]:
    if optimizer is None:
        raise RuntimeError("Trainer did not provide the optimizer to the LR audit callback")
    return [float(group["lr"]) for group in optimizer.param_groups]


def _assert_target(values: list[float], event: str) -> None:
    if not values or any(
        not math.isclose(value, TARGET_LR, rel_tol=1e-9, abs_tol=1e-12)
        for value in values
    ):
        raise RuntimeError(
            f"stage {STAGE} {event}: optimizer LR {values} differs from target {TARGET_LR}"
        )


def _write(event: str, state, optimizer, logs=None) -> None:
    epoch = float(state.epoch or 0.0)
    expected_start = float(STAGE - 1)
    if event in {"train_begin", "epoch_begin"} and not math.isclose(
        epoch, expected_start, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"stage {STAGE} {event}: trainer epoch {epoch} differs from expected {expected_start}"
        )
    if event in {"step_begin", "log"} and not (
        expected_start - 1e-6 <= epoch <= float(STAGE) + 1e-6
    ):
        raise RuntimeError(
            f"stage {STAGE} {event}: trainer epoch {epoch} is outside the stage boundary"
        )
    values = _optimizer_lrs(optimizer)
    _assert_target(values, event)
    logged_lr = (logs or {}).get("learning_rate")
    if logged_lr is not None and not math.isclose(
        float(logged_lr), TARGET_LR, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"stage {STAGE} {event}: logged LR {logged_lr} differs from target {TARGET_LR}"
        )
    if PROCESS_RANK != 0:
        return
    payload = {
        "at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "event": event,
        "rank": PROCESS_RANK,
        "stage": STAGE,
        "epoch": epoch,
        "global_step": state.global_step,
        "target_lr": TARGET_LR,
        "optimizer_lrs": values,
        "logged_learning_rate": logged_lr,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class Qwen360805FixedStageLRCallback(TrainerCallback):
    @staticmethod
    def _apply(optimizer, lr_scheduler) -> None:
        if optimizer is None:
            raise RuntimeError("Trainer did not provide the optimizer to the LR callback")
        for group in optimizer.param_groups:
            group["lr"] = TARGET_LR
            group["initial_lr"] = TARGET_LR
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs = [TARGET_LR for _ in lr_scheduler.base_lrs]
            if hasattr(lr_scheduler, "_last_lr"):
                lr_scheduler._last_lr = [TARGET_LR for _ in lr_scheduler._last_lr]

    def on_train_begin(
        self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
    ):
        self._apply(optimizer, lr_scheduler)
        _write("train_begin", state, optimizer)
        return control

    def on_epoch_begin(
        self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
    ):
        self._apply(optimizer, lr_scheduler)
        _write("epoch_begin", state, optimizer)
        return control

    def on_step_begin(
        self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
    ):
        self._apply(optimizer, lr_scheduler)
        _write("step_begin", state, optimizer)
        return control

    def on_log(self, args, state, control, logs=None, optimizer=None, **kwargs):
        _write("log", state, optimizer, logs)
        return control


callbacks_map["qwen36_0805_fixed_stage_lr"] = Qwen360805FixedStageLRCallback
