#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_ENV="${TRAIN_ENV:-/root/autodl-tmp/envs/qwen36-sft}"
PYTHON_BIN="${PYTHON_BIN:-${TRAIN_ENV}/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-${TRAIN_ENV}/bin/swift}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-08-05}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen36-27b-0805-causal-path-reasoning-5epoch}"
FORMAL_CONFIG_PATH="${FORMAL_CONFIG_PATH:-${REPO_ROOT}/config/qwen36_0805_formal_training.json}"
LR_PLUGIN_PATH="${LR_PLUGIN_PATH:-${REPO_ROOT}/scripts/qwen36_0805_fixed_stage_lr_plugin.py}"
CORE_DATASET_PATH="${CORE_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_core.jsonl}"
ENDPOINT_POOL_DATASET_PATH="${ENDPOINT_POOL_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_validation.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/sft/reasoning_causal_path_manifest.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export PYTORCH_CUDA_ALLOC_CONF

for path in \
  "${PYTHON_BIN}" \
  "${SWIFT_BIN}" \
  "${MODEL_PATH}" \
  "${FORMAL_CONFIG_PATH}" \
  "${LR_PLUGIN_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required formal-training path is missing: ${path}" >&2
    exit 1
  fi
done

FORMAL_MAX_LENGTH="$(
  "${PYTHON_BIN}" - "${FORMAL_CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(int(json.load(handle)["max_length"]))
PY
)"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -B scripts/convert_0805_causal_path_reasoning_sft.py
"${PYTHON_BIN}" -B scripts/validate_0805_causal_path_reasoning_sft.py
"${PYTHON_BIN}" -B scripts/check_0804_best1_token_lengths.py \
  --model "${MODEL_PATH}" \
  --max-length "${FORMAL_MAX_LENGTH}" \
  --train "${CORE_DATASET_PATH}" \
  --extra-train "${ENDPOINT_POOL_DATASET_PATH}" \
  --validation "${VALIDATION_DATASET_PATH}"

declare -A CFG=()
declare -a LEARNING_RATES=()
while IFS=$'\t' read -r key value; do
  if [[ "${key}" == "fixed_learning_rate_by_epoch" ]]; then
    LEARNING_RATES+=("${value}")
  else
    CFG["${key}"]="${value}"
  fi
done < <(
  "${PYTHON_BIN}" - "${FORMAL_CONFIG_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)

keys = (
    "schema_version",
    "distributed_strategy",
    "world_size",
    "cuda_visible_devices",
    "epochs",
    "check_model",
    "tuner_type",
    "torch_dtype",
    "freeze_vit",
    "freeze_aligner",
    "target_modules",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "max_length",
    "truncation_strategy",
    "loss_scale",
    "is_binary_loss_scale",
    "preserve_thinking",
    "add_non_thinking_prefix",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "lr_scheduler_type",
    "warmup_ratio",
    "gradient_checkpointing",
    "packing",
    "split_dataset_ratio",
    "dataset_num_proc",
    "dataloader_num_workers",
    "logging_steps",
    "eval_strategy",
    "save_strategy",
    "save_total_limit",
    "load_best_model_at_end",
    "resume_only_model",
    "seed",
    "data_seed",
    "report_to",
    "add_version",
)

def render(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)

for key in keys:
    print(f"{key}\t{render(config[key])}")
for value in config["fixed_learning_rate_by_epoch"]:
    print(f"fixed_learning_rate_by_epoch\t{value}")
PY
)

if [[ "${CFG[schema_version]:-}" != "qwen36-0805-formal-training.v2" ]]; then
  echo "Unexpected formal-training config schema: ${CFG[schema_version]:-missing}" >&2
  exit 1
fi
if [[ "${CFG[distributed_strategy]}" != "ddp" || "${CFG[world_size]}" != "2" ]]; then
  echo "Formal training requires two-process DDP." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "${CFG[cuda_visible_devices]}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} differs from config ${CFG[cuda_visible_devices]}." >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE}" != "${CFG[world_size]}" ]]; then
  echo "NPROC_PER_NODE=${NPROC_PER_NODE} differs from configured world_size=${CFG[world_size]}." >&2
  exit 1
fi
if [[ "${CFG[epochs]}" != "5" || "${#LEARNING_RATES[@]}" -ne 5 ]]; then
  echo "Formal training requires exactly five stages and five learning rates." >&2
  exit 1
fi
if [[ "${CFG[per_device_train_batch_size]}" != "1" || "${CFG[gradient_accumulation_steps]}" != "4" ]]; then
  echo "Formal training requires per-device batch 1 and gradient_accumulation_steps=4." >&2
  exit 1
fi
calculated_effective_batch="$((CFG[world_size] * CFG[per_device_train_batch_size] * CFG[gradient_accumulation_steps]))"
if [[ "${calculated_effective_batch}" != "${CFG[effective_batch_size]}" || "${CFG[effective_batch_size]}" != "8" ]]; then
  echo "Formal training effective-batch arithmetic is inconsistent: ${calculated_effective_batch} != ${CFG[effective_batch_size]}." >&2
  exit 1
fi
if [[ "${CFG[lr_scheduler_type]}" != "constant" || "${CFG[warmup_ratio]}" != "0.0" ]]; then
  echo "Formal training requires a constant scheduler and zero warmup." >&2
  exit 1
fi
if [[ "${CFG[resume_only_model]}" != "false" ]]; then
  echo "Formal training must resume the complete trainer state." >&2
  exit 1
fi

for path in \
  "${CORE_DATASET_PATH}" \
  "${ENDPOINT_POOL_DATASET_PATH}" \
  "${VALIDATION_DATASET_PATH}" \
  "${MANIFEST_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required generated artifact is missing: ${path}" >&2
    exit 1
  fi
done

latest_checkpoint() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
items = []
for path in root.glob("checkpoint-*"):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match and path.is_dir():
        items.append((int(match.group(1)), path))
if items:
    print(max(items)[1])
PY
}

checkpoint_epoch() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import math
import sys
from pathlib import Path

state_path = Path(sys.argv[1]) / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(f"Missing trainer state: {state_path}")
state = json.loads(state_path.read_text(encoding="utf-8"))
epoch = float(state.get("epoch", -1))
rounded = round(epoch)
if not math.isclose(epoch, rounded, abs_tol=1e-6):
    raise SystemExit(f"Checkpoint is not at an epoch boundary: epoch={epoch}")
print(rounded)
PY
}

capture_training_environment() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import platform

print(f"python={platform.python_version()}")
for name in ("ms-swift", "torch", "transformers", "peft", "accelerate"):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=missing")
PY
}

existing_checkpoint="$(latest_checkpoint)"
start_stage=1
if [[ -n "${existing_checkpoint}" ]]; then
  for path in \
    "${OUTPUT_DIR}/control/git_commit.txt" \
    "${OUTPUT_DIR}/control/input_sha256.txt" \
    "${OUTPUT_DIR}/control/model_files_sha256.txt" \
    "${OUTPUT_DIR}/control/environment.txt"; do
    if [[ ! -f "${path}" ]]; then
      echo "Existing checkpoint lacks formal-run provenance: ${path}" >&2
      exit 1
    fi
  done
  recorded_commit="$(<"${OUTPUT_DIR}/control/git_commit.txt")"
  current_commit="$(git rev-parse HEAD)"
  if [[ "${recorded_commit}" != "${current_commit}" ]]; then
    echo "Resume commit mismatch: recorded=${recorded_commit}, current=${current_commit}" >&2
    exit 1
  fi
  if ! sha256sum --check "${OUTPUT_DIR}/control/input_sha256.txt"; then
    echo "Resume input hashes differ from the original formal run." >&2
    exit 1
  fi
  if ! sha256sum --check "${OUTPUT_DIR}/control/model_files_sha256.txt"; then
    echo "Resume model files differ from the original formal run." >&2
    exit 1
  fi
  recorded_environment="$(<"${OUTPUT_DIR}/control/environment.txt")"
  current_environment="$(capture_training_environment)"
  if [[ "${recorded_environment}" != "${current_environment}" ]]; then
    echo "Resume Python/training package versions differ from the original formal run." >&2
    diff \
      <(printf '%s\n' "${recorded_environment}") \
      <(printf '%s\n' "${current_environment}") \
      >&2 || true
    exit 1
  fi
  completed_stage="$(checkpoint_epoch "${existing_checkpoint}")"
  if [[ "${completed_stage}" -lt 1 || "${completed_stage}" -gt 5 ]]; then
    echo "Latest checkpoint has invalid completed epoch: ${completed_stage}" >&2
    exit 1
  fi
  start_stage=$((completed_stage + 1))
elif [[ -e "${OUTPUT_DIR}/formal_training_started.json" ]]; then
  echo "A previous run started but produced no checkpoint. Use a new OUTPUT_DIR for a clean retry." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/control"
LR_AUDIT_PATH="${OUTPUT_DIR}/control/learning_rate_audit.jsonl"
WORKFLOW_LOG="${OUTPUT_DIR}/control/formal_training.log"
exec > >(tee -a "${WORKFLOW_LOG}") 2>&1

if [[ "${start_stage}" -gt 5 ]]; then
  echo "[$(date -Iseconds)] Formal 0805 training is already complete: ${existing_checkpoint}"
  exit 0
fi

if [[ "${start_stage}" -eq 1 ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked repository changes exist; formal training requires a clean committed worktree." >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/formal_training_started.json" "${FORMAL_CONFIG_PATH}" "${MODEL_PATH}" "${TRAIN_ENV}" <<'PY'
import datetime
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "formal_config": sys.argv[2],
    "model_path": sys.argv[3],
    "training_environment": sys.argv[4],
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  git rev-parse HEAD >"${OUTPUT_DIR}/control/git_commit.txt"
  git status --short --branch >"${OUTPUT_DIR}/control/git_status_before.txt"
  sha256sum \
    "${FORMAL_CONFIG_PATH}" \
    "${BASH_SOURCE[0]}" \
    "${MANIFEST_PATH}" \
    "${CORE_DATASET_PATH}" \
    "${ENDPOINT_POOL_DATASET_PATH}" \
    "${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_01.jsonl" \
    "${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_02.jsonl" \
    "${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_03.jsonl" \
    "${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_04.jsonl" \
    "${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_05.jsonl" \
    "${VALIDATION_DATASET_PATH}" \
    "${LR_PLUGIN_PATH}" \
    >"${OUTPUT_DIR}/control/input_sha256.txt"
  find "${MODEL_PATH}" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    >"${OUTPUT_DIR}/control/model_files_sha256.txt"
  capture_training_environment >"${OUTPUT_DIR}/control/environment.txt"
fi

for ((stage = start_stage; stage <= 5; stage++)); do
  printf -v stage_tag "%02d" "${stage}"
  endpoint_dataset="${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_${stage_tag}.jsonl"
  if [[ ! -f "${endpoint_dataset}" ]]; then
    echo "Stage ${stage} endpoint schedule is missing: ${endpoint_dataset}" >&2
    exit 1
  fi

  target_lr="${LEARNING_RATES[$((stage - 1))]}"
  declare -a resume_args=()
  if [[ "${stage}" -gt 1 ]]; then
    if [[ -z "${existing_checkpoint}" ]]; then
      echo "Stage ${stage} requires the previous full-state checkpoint." >&2
      exit 1
    fi
    previous_epoch="$(checkpoint_epoch "${existing_checkpoint}")"
    if [[ "${previous_epoch}" -ne $((stage - 1)) ]]; then
      echo "Stage ${stage} expected an epoch-$((stage - 1)) checkpoint, found epoch ${previous_epoch}." >&2
      exit 1
    fi
    resume_args+=(
      --resume_from_checkpoint "${existing_checkpoint}"
      --resume_only_model "${CFG[resume_only_model]}"
    )
  fi

  export QWEN36_0805_TRAIN_STAGE="${stage}"
  export QWEN36_0805_TARGET_LR="${target_lr}"
  export QWEN36_0805_LR_AUDIT_PATH="${LR_AUDIT_PATH}"

  echo "[$(date -Iseconds)] Starting formal stage ${stage}/5: lr=${target_lr}, endpoint=${endpoint_dataset}, resume=${existing_checkpoint:-none}"
  "${SWIFT_BIN}" sft \
    --model "${MODEL_PATH}" \
    --check_model "${CFG[check_model]}" \
    --dataset "${CORE_DATASET_PATH}" "${endpoint_dataset}" \
    --val_dataset "${VALIDATION_DATASET_PATH}" \
    --tuner_type "${CFG[tuner_type]}" \
    --torch_dtype "${CFG[torch_dtype]}" \
    --freeze_vit "${CFG[freeze_vit]}" \
    --freeze_aligner "${CFG[freeze_aligner]}" \
    --target_modules "${CFG[target_modules]}" \
    --lora_rank "${CFG[lora_rank]}" \
    --lora_alpha "${CFG[lora_alpha]}" \
    --lora_dropout "${CFG[lora_dropout]}" \
    --max_length "${CFG[max_length]}" \
    --truncation_strategy "${CFG[truncation_strategy]}" \
    --loss_scale "${CFG[loss_scale]}" \
    --is_binary_loss_scale "${CFG[is_binary_loss_scale]}" \
    --preserve_thinking "${CFG[preserve_thinking]}" \
    --add_non_thinking_prefix "${CFG[add_non_thinking_prefix]}" \
    --num_train_epochs "${stage}" \
    "${resume_args[@]}" \
    --per_device_train_batch_size "${CFG[per_device_train_batch_size]}" \
    --per_device_eval_batch_size "${CFG[per_device_eval_batch_size]}" \
    --gradient_accumulation_steps "${CFG[gradient_accumulation_steps]}" \
    --learning_rate "${target_lr}" \
    --lr_scheduler_type "${CFG[lr_scheduler_type]}" \
    --warmup_ratio "${CFG[warmup_ratio]}" \
    --gradient_checkpointing "${CFG[gradient_checkpointing]}" \
    --packing "${CFG[packing]}" \
    --split_dataset_ratio "${CFG[split_dataset_ratio]}" \
    --dataset_num_proc "${CFG[dataset_num_proc]}" \
    --dataloader_num_workers "${CFG[dataloader_num_workers]}" \
    --logging_steps "${CFG[logging_steps]}" \
    --eval_strategy "${CFG[eval_strategy]}" \
    --save_strategy "${CFG[save_strategy]}" \
    --save_total_limit "${CFG[save_total_limit]}" \
    --load_best_model_at_end "${CFG[load_best_model_at_end]}" \
    --seed "${CFG[seed]}" \
    --data_seed "${CFG[data_seed]}" \
    --report_to "${CFG[report_to]}" \
    --external_plugins "${LR_PLUGIN_PATH}" \
    --callbacks qwen36_0805_fixed_stage_lr \
    --add_version "${CFG[add_version]}" \
    --output_dir "${OUTPUT_DIR}"

  new_checkpoint="$(latest_checkpoint)"
  if [[ -z "${new_checkpoint}" || "${new_checkpoint}" == "${existing_checkpoint}" ]]; then
    echo "Stage ${stage} did not create a new checkpoint." >&2
    exit 1
  fi
  completed_epoch="$(checkpoint_epoch "${new_checkpoint}")"
  if [[ "${completed_epoch}" -ne "${stage}" ]]; then
    echo "Stage ${stage} checkpoint reports epoch ${completed_epoch}." >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${LR_AUDIT_PATH}" "${stage}" "${target_lr}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = int(sys.argv[2])
target = float(sys.argv[3])
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
rows = [row for row in rows if int(row["stage"]) == stage]
events = {row["event"] for row in rows}
required = {"train_begin", "epoch_begin", "step_begin", "log"}
if not required.issubset(events):
    raise SystemExit(f"stage {stage} LR audit lacks events: {sorted(required - events)}")
for row in rows:
    if not math.isclose(float(row["target_lr"]), target, rel_tol=1e-9, abs_tol=1e-12):
        raise SystemExit(f"stage {stage} target LR audit mismatch")
    values = [float(value) for value in row["optimizer_lrs"]]
    if not values or any(
        not math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-12)
        for value in values
    ):
        raise SystemExit(f"stage {stage} optimizer LR audit mismatch: {values}")
print(f"stage {stage} LR audit passed: rows={len(rows)}, target={target}")
PY

  existing_checkpoint="${new_checkpoint}"
  echo "[$(date -Iseconds)] Completed formal stage ${stage}/5: ${existing_checkpoint}"
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}/formal_training_complete.json" "${existing_checkpoint}" "${OUTPUT_DIR}" <<'PY'
import datetime
import json
import math
import re
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(sys.argv[3])
checkpoint_epochs = {}
for checkpoint in root.glob("checkpoint-*"):
    if not checkpoint.is_dir() or not re.fullmatch(r"checkpoint-\d+", checkpoint.name):
        continue
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        continue
    state = json.loads(state_path.read_text(encoding="utf-8"))
    epoch = float(state.get("epoch", -1))
    rounded = round(epoch)
    if math.isclose(epoch, rounded, abs_tol=1e-6):
        checkpoint_epochs[rounded] = str(checkpoint)
if sorted(checkpoint_epochs) != [1, 2, 3, 4, 5]:
    raise SystemExit(
        f"formal run must retain epoch 1..5 checkpoints, found {sorted(checkpoint_epochs)}"
    )
payload = {
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "final_checkpoint": sys.argv[2],
    "completed_stages": 5,
    "checkpoints_by_epoch": checkpoint_epochs,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
echo "[$(date -Iseconds)] Formal 0805 five-stage training complete: ${existing_checkpoint}"
