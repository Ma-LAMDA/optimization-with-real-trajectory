#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/2026-07-28/sft/qwen3_6_27b_reasoning_decision_train.jsonl}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${REPO_ROOT}/data/2026-07-28/sft/qwen3_6_27b_reasoning_decision_validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen36-27b-reasoning-lora-smoke}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF

if ! command -v swift >/dev/null 2>&1; then
  echo "swift was not found in PATH; activate the qwen36-sft environment first." >&2
  exit 1
fi

if ! python -c "import fla" >/dev/null 2>&1; then
  echo "flash-linear-attention is required by Qwen3.6 linear attention." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${VALIDATION_DATASET_PATH}" ]]; then
  echo "Validation dataset does not exist: ${VALIDATION_DATASET_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
python scripts/validate_codex_run_sft.py

exec swift sft \
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
  --max_length 4096 \
  --truncation_strategy delete \
  --loss_scale default \
  --add_non_thinking_prefix false \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --gradient_checkpointing true \
  --packing false \
  --split_dataset_ratio 0 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 2 \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --output_dir "${OUTPUT_DIR}"
