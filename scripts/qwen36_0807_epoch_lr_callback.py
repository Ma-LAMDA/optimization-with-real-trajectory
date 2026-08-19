"""Force and audit the documented fixed learning rate after checkpoint resume."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map


TARGET_LR = float(os.environ["FORCED_EPOCH_LR"])
AUDIT_PATH = Path(os.environ["EPOCH_LR_AUDIT_PATH"])
PROCESS_RANK = int(os.environ.get("RANK", "0"))


def write_audit(event, state, optimizer) -> None:
    optimizer_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(abs(value - TARGET_LR) > 1e-12 for value in optimizer_lrs):
        raise RuntimeError(
            f"{event}: optimizer LR mismatch: {optimizer_lrs} != {TARGET_LR}"
        )
    payload = {
        "at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "event": event,
        "rank": PROCESS_RANK,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "target_lr": TARGET_LR,
        "optimizer_lrs": optimizer_lrs,
    }
    if PROCESS_RANK != 0:
        return
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ForcedEpochLearningRateCallback(TrainerCallback):
    def apply(self, optimizer, lr_scheduler) -> None:
        for group in optimizer.param_groups:
            group["lr"] = TARGET_LR
            group["initial_lr"] = TARGET_LR
        if hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [TARGET_LR for _ in lr_scheduler.base_lrs]
        if hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = [TARGET_LR for _ in lr_scheduler._last_lr]

    def on_train_begin(
        self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
    ):
        self.apply(optimizer, lr_scheduler)
        write_audit("train_begin", state, optimizer)
        return control

    def on_step_begin(
        self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
    ):
        self.apply(optimizer, lr_scheduler)
        write_audit("step_begin", state, optimizer)
        return control

    def on_train_end(
        self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs
    ):
        self.apply(optimizer, lr_scheduler)
        write_audit("train_end", state, optimizer)
        return control


callbacks_map["forced_epoch_lr"] = ForcedEpochLearningRateCallback
