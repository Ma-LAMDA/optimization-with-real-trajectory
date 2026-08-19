import json
import os
from datetime import datetime, timezone, timedelta
from swift.callbacks import TrainerCallback, callbacks_map

SCHEDULE = [2.0e-5, 1.5e-5, 1.0e-5, 6.0e-6, 3.0e-6]
AUDIT_PATH = os.environ.get("EPOCH_LR_AUDIT_PATH")

def _write(event, state, lr, optimizer=None, logs=None):
    if not AUDIT_PATH:
        return
    payload = {
        "at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "event": event,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "target_lr": lr,
        "optimizer_lrs": [group.get("lr") for group in optimizer.param_groups] if optimizer else None,
        "logged_learning_rate": (logs or {}).get("learning_rate"),
    }
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

class EpochStepLRCallback(TrainerCallback):
    def _target(self, state):
        index = max(0, min(len(SCHEDULE) - 1, int(state.epoch or 0)))
        return SCHEDULE[index]

    def _apply(self, state, optimizer=None, lr_scheduler=None):
        lr = self._target(state)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["initial_lr"] = lr
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs = [lr for _ in lr_scheduler.base_lrs]
            if hasattr(lr_scheduler, "_last_lr"):
                lr_scheduler._last_lr = [lr for _ in lr_scheduler._last_lr]
        return lr

    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
        lr = self._apply(state, optimizer, lr_scheduler)
        _write("train_begin", state, lr, optimizer)
        return control

    def on_epoch_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
        lr = self._apply(state, optimizer, lr_scheduler)
        _write("epoch_begin", state, lr, optimizer)
        return control

    def on_log(self, args, state, control, logs=None, optimizer=None, **kwargs):
        _write("log", state, self._target(state), optimizer, logs)
        return control

callbacks_map["epoch_step_lr"] = EpochStepLRCallback
