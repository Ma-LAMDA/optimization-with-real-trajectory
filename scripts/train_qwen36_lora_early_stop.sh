#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-07-31}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_decision_train.jsonl}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_decision_validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen36-27b-reasoning-lora-early-stop}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EARLY_STOP_INTERVAL="${EARLY_STOP_INTERVAL:-3}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_ONLY_MODEL="${RESUME_ONLY_MODEL:-false}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable was not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v "${SWIFT_BIN}" >/dev/null 2>&1; then
  echo "Swift executable was not found: ${SWIFT_BIN}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import fla" >/dev/null 2>&1; then
  echo "flash-linear-attention is required by Qwen3.6 linear attention." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Training dataset does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${VALIDATION_DATASET_PATH}" ]]; then
  echo "Validation dataset does not exist: ${VALIDATION_DATASET_PATH}" >&2
  exit 1
fi

if [[ "${EVAL_STEPS}" -le 0 || "${EARLY_STOP_INTERVAL}" -le 0 ]]; then
  echo "EVAL_STEPS and EARLY_STOP_INTERVAL must both be positive." >&2
  exit 1
fi
if [[ ! "${MAX_STEPS}" =~ ^(-1|[1-9][0-9]*)$ ]]; then
  echo "MAX_STEPS must be -1 or a positive integer." >&2
  exit 1
fi
if [[ "${RESUME_ONLY_MODEL}" != "false" && "${RESUME_ONLY_MODEL}" != "true" ]]; then
  echo "RESUME_ONLY_MODEL must be false or true." >&2
  exit 1
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi

declare -a continuation_args=()
if [[ "${MAX_STEPS}" -gt 0 ]]; then
  continuation_args+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  continuation_args+=(
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
    --resume_only_model "${RESUME_ONLY_MODEL}"
  )
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/validate_100x10_sft.py --data-root "${DATA_ROOT}"

exec "${SWIFT_BIN}" sft \
  --model "${MODEL_PATH}" \
  --check_model false \
  --dataset "${DATASET_PATH}" \
  --val_dataset "${VALIDATION_DATASET_PATH}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length "${MAX_LENGTH}" \
  --truncation_strategy delete \
  --loss_scale default \
  --add_non_thinking_prefix false \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  "${continuation_args[@]}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --gradient_checkpointing true \
  --packing false \
  --split_dataset_ratio 0 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --logging_steps 1 \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${EVAL_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --early_stop_interval "${EARLY_STOP_INTERVAL}" \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --output_dir "${OUTPUT_DIR}"
