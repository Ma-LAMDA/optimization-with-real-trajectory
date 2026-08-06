#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-08-05}"
TRAIN_EPOCH_INDEX="${TRAIN_EPOCH_INDEX:-1}"
if [[ ! "${TRAIN_EPOCH_INDEX}" =~ ^[1-5]$ ]]; then
  echo "TRAIN_EPOCH_INDEX must be an integer from 1 to 5." >&2
  exit 1
fi
printf -v TRAIN_EPOCH_TAG "%02d" "${TRAIN_EPOCH_INDEX}"
SEMANTIC_POOL_PATH="${SEMANTIC_POOL_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train.jsonl}"
CORE_DATASET_PATH="${CORE_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_core.jsonl}"
ENDPOINT_POOL_DATASET_PATH="${ENDPOINT_POOL_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl}"
ENDPOINT_SCHEDULE_DATASET_PATH="${ENDPOINT_SCHEDULE_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_${TRAIN_EPOCH_TAG}.jsonl}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_causal_path_validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen36-27b-0805-causal-path-reasoning-quick-epoch-${TRAIN_EPOCH_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Keep the 0804 quick experiment parameters unchanged.  Only dataset and
# output paths differ.
MAX_LENGTH="${MAX_LENGTH:-16384}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
if [[ "${NUM_TRAIN_EPOCHS}" != "1" ]]; then
  echo "This entry consumes one epoch-specific endpoint-group schedule; NUM_TRAIN_EPOCHS must be 1." >&2
  exit 1
fi
EVAL_STEPS="${EVAL_STEPS:-40}"
EARLY_STOP_INTERVAL="${EARLY_STOP_INTERVAL:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

for executable in "${PYTHON_BIN}" "${SWIFT_BIN}"; do
  if ! command -v "${executable}" >/dev/null 2>&1; then
    echo "Required executable was not found: ${executable}" >&2
    exit 1
  fi
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -B scripts/convert_0805_causal_path_reasoning_sft.py
"${PYTHON_BIN}" -B scripts/validate_0805_causal_path_reasoning_sft.py
if [[ ! -f "${SEMANTIC_POOL_PATH}" || ! -f "${CORE_DATASET_PATH}" || ! -f "${ENDPOINT_POOL_DATASET_PATH}" || ! -f "${ENDPOINT_SCHEDULE_DATASET_PATH}" || ! -f "${VALIDATION_DATASET_PATH}" ]]; then
  echo "0805 causal-path semantic pool or epoch schedule is missing." >&2
  exit 1
fi
"${PYTHON_BIN}" -B scripts/check_0804_best1_token_lengths.py \
  --model "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --train "${CORE_DATASET_PATH}" \
  --extra-train "${ENDPOINT_POOL_DATASET_PATH}" \
  --validation "${VALIDATION_DATASET_PATH}"

mkdir -p "${OUTPUT_DIR}"

exec "${SWIFT_BIN}" sft \
  --model "${MODEL_PATH}" \
  --check_model false \
  --dataset "${CORE_DATASET_PATH}" "${ENDPOINT_SCHEDULE_DATASET_PATH}" \
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
  --is_binary_loss_scale false \
  --preserve_thinking true \
  --add_non_thinking_prefix false \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
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
