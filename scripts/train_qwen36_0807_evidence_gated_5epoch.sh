#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-08-07}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_ROOT}/output/qwen36-27b-0807-evidence-gated-5epoch}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWIFT_BIN="${SWIFT_BIN:-swift}"
RUN_MODE="${RUN_MODE:-train}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEMANTIC_POOL="${DATA_ROOT}/sft/qwen3_6_27b_0807_train_semantic_pool.jsonl"
CORE_POOL="${DATA_ROOT}/sft/qwen3_6_27b_0807_core_pool.jsonl"
ENDPOINT_POOL="${DATA_ROOT}/sft/qwen3_6_27b_0807_endpoint_pool.jsonl"
VALIDATION="${DATA_ROOT}/sft/qwen3_6_27b_0807_validation.jsonl"
LR_CALLBACK="${REPO_ROOT}/scripts/qwen36_0807_epoch_lr_callback.py"
LEARNING_RATES=(2e-5 1.5e-5 1e-5 6e-6 3e-6)
EXPECTED_TRAIN_ROWS_PER_STAGE=216
WORLD_SIZE=2
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=$((WORLD_SIZE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
EXPECTED_OPTIMIZER_STEPS_PER_STAGE=27
FIXED_VALIDATION_EPOCH=3
FIXED_VALIDATION_CHECKPOINT_SUFFIX=81

if [[ "${RUN_MODE}" != "prepare" && "${RUN_MODE}" != "dry-run" && "${RUN_MODE}" != "train" ]]; then
  echo "RUN_MODE must be prepare, dry-run, or train." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable was not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1" || "${NPROC_PER_NODE}" != "${WORLD_SIZE}" ]]; then
  echo "0807 formal training requires CUDA_VISIBLE_DEVICES=0,1 and NPROC_PER_NODE=2." >&2
  exit 1
fi
if [[ "${EFFECTIVE_BATCH_SIZE}" -ne 8 ]]; then
  echo "0807 formal training effective batch must remain 8." >&2
  exit 1
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -B scripts/convert_0807_evidence_gated_reasoning_sft.py
"${PYTHON_BIN}" -B scripts/validate_0807_evidence_gated_reasoning_sft.py
"${PYTHON_BIN}" -B scripts/independent_validate_0807_sft.py

for path in "${SEMANTIC_POOL}" "${CORE_POOL}" "${ENDPOINT_POOL}" "${VALIDATION}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required 0807 dataset is missing: ${path}" >&2
    exit 1
  fi
done
for epoch in 1 2 3 4 5; do
  printf -v tag "%02d" "${epoch}"
  for component in \
    "${DATA_ROOT}/sft/qwen3_6_27b_0807_core_epoch_${tag}.jsonl" \
    "${DATA_ROOT}/sft/qwen3_6_27b_0807_endpoint_epoch_${tag}.jsonl"; do
    if [[ ! -f "${component}" ]]; then
      echo "Required epoch component is missing: ${component}" >&2
      exit 1
    fi
  done
done

if [[ "${RUN_MODE}" == "prepare" || "${RUN_MODE}" == "train" ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -B scripts/check_0807_target_tokenizer_preflight.py \
    --model "${MODEL_PATH}" \
    --max-length "${MAX_LENGTH}" \
    --output "${DATA_ROOT}/sft/TARGET_TOKENIZER_PREFLIGHT.json" \
    --dataset "train_core_pool=${CORE_POOL}" \
    --dataset "train_endpoint_pool=${ENDPOINT_POOL}" \
    --dataset "validation=${VALIDATION}"
  "${PYTHON_BIN}" -B scripts/convert_0807_evidence_gated_reasoning_sft.py
  "${PYTHON_BIN}" -B scripts/validate_0807_evidence_gated_reasoning_sft.py
  "${PYTHON_BIN}" -B scripts/independent_validate_0807_sft.py
fi
if [[ "${RUN_MODE}" == "prepare" ]]; then
  exit 0
fi

mkdir -p "${BASE_OUTPUT_DIR}"
PLAN_PATH="${BASE_OUTPUT_DIR}/training_plan.jsonl"
: >"${PLAN_PATH}"
resume_checkpoint=""
for epoch in 1 2 3 4 5; do
  printf -v tag "%02d" "${epoch}"
  epoch_dir="${BASE_OUTPUT_DIR}/epoch_${tag}"
  core="${DATA_ROOT}/sft/qwen3_6_27b_0807_core_epoch_${tag}.jsonl"
  endpoint="${DATA_ROOT}/sft/qwen3_6_27b_0807_endpoint_epoch_${tag}.jsonl"
  learning_rate="${LEARNING_RATES[$((epoch - 1))]}"
  train_rows=$(($(wc -l <"${core}") + $(wc -l <"${endpoint}")))
  if [[ "${train_rows}" -ne "${EXPECTED_TRAIN_ROWS_PER_STAGE}" ]]; then
    echo "Epoch ${tag} has ${train_rows} rows; expected ${EXPECTED_TRAIN_ROWS_PER_STAGE}." >&2
    exit 1
  fi
  optimizer_steps=$(((train_rows + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE))
  if [[ "${optimizer_steps}" -ne "${EXPECTED_OPTIMIZER_STEPS_PER_STAGE}" ]]; then
    echo "Epoch ${tag} has ${optimizer_steps} optimizer steps; expected ${EXPECTED_OPTIMIZER_STEPS_PER_STAGE}." >&2
    exit 1
  fi
  expected_start_step=$(((epoch - 1) * EXPECTED_OPTIMIZER_STEPS_PER_STAGE))
  expected_end_step=$((epoch * EXPECTED_OPTIMIZER_STEPS_PER_STAGE))
  "${PYTHON_BIN}" - "${PLAN_PATH}" "${epoch}" "${learning_rate}" "${core}" "${endpoint}" "${epoch_dir}" "${resume_checkpoint}" "${expected_start_step}" "${expected_end_step}" "${optimizer_steps}" "${WORLD_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${EFFECTIVE_BATCH_SIZE}" <<'PY'
import json,sys
from pathlib import Path
path,epoch,lr,core,endpoint,output,resume,start,end,steps,world_size,accum,effective_batch=sys.argv[1:]
with Path(path).open("a",encoding="utf-8") as handle:
    handle.write(json.dumps({
        "epoch":int(epoch), "fixed_learning_rate":float(lr),
        "core_schedule":core, "endpoint_schedule":endpoint,
        "output_dir":output, "resume_from_checkpoint":resume or None,
        "resume_optimizer_and_scheduler":bool(resume),
        "distributed_strategy":"ddp", "world_size":int(world_size),
        "per_device_train_batch_size":1,
        "gradient_accumulation_steps":int(accum),
        "effective_batch_size":int(effective_batch), "lr_scheduler_type":"constant",
        "warmup_ratio":0.0, "cumulative_num_train_epochs":int(epoch),
        "expected_start_global_step":int(start),
        "expected_end_global_step":int(end),
        "expected_optimizer_steps":int(steps),
    },ensure_ascii=False)+"\n")
PY
  if [[ "${RUN_MODE}" == "dry-run" ]]; then
    resume_checkpoint="${epoch_dir}/checkpoint-${expected_end_step}"
    continue
  fi
  if ! command -v "${SWIFT_BIN}" >/dev/null 2>&1; then
    echo "Swift executable was not found: ${SWIFT_BIN}" >&2
    exit 1
  fi
  declare -a resume_args=()
  if [[ "${epoch}" -gt 1 ]]; then
    if [[ -z "${resume_checkpoint}" || ! -d "${resume_checkpoint}" ]]; then
      echo "Epoch ${tag} requires the previous real checkpoint." >&2
      exit 1
    fi
    resume_args+=(--resume_from_checkpoint "${resume_checkpoint}" --resume_only_model false)
  fi
  mkdir -p "${epoch_dir}"
  export FORCED_EPOCH_LR="${learning_rate}"
  export EPOCH_LR_AUDIT_PATH="${epoch_dir}/learning_rate_audit.jsonl"
  : >"${EPOCH_LR_AUDIT_PATH}"
  export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PYTORCH_CUDA_ALLOC_CONF
  "${SWIFT_BIN}" sft \
    --model "${MODEL_PATH}" \
    --check_model false \
    --dataset "${core}" "${endpoint}" \
    --val_dataset "${VALIDATION}" \
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
    --num_train_epochs "${epoch}" \
    "${resume_args[@]}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${learning_rate}" \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --gradient_checkpointing true \
    --packing false \
    --split_dataset_ratio 0 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --logging_steps 1 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end false \
    --seed 42 \
    --data_seed 42 \
    --report_to none \
    --external_plugins "${LR_CALLBACK}" \
    --callbacks forced_epoch_lr \
    --add_version false \
    --output_dir "${epoch_dir}"
  "${PYTHON_BIN}" - "${EPOCH_LR_AUDIT_PATH}" "${learning_rate}" "${expected_start_step}" "${expected_end_step}" "${optimizer_steps}" <<'PY'
import json,sys
from pathlib import Path
path,target=Path(sys.argv[1]),float(sys.argv[2])
expected_start,expected_end,expected_steps=map(int,sys.argv[3:])
rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
events=[row["event"] for row in rows]
if not rows or events[0] != "train_begin" or "step_begin" not in events or events[-1] != "train_end":
    raise SystemExit(f"incomplete learning-rate audit: {events[:3]} ... {events[-3:]}")
if any(abs(float(value)-target)>1e-12 for row in rows for value in row["optimizer_lrs"]):
    raise SystemExit("learning-rate audit contains a mismatched optimizer LR")
step_rows=[row for row in rows if row["event"]=="step_begin"]
step_globals=[int(row["global_step"]) for row in step_rows]
if int(rows[0]["global_step"]) != expected_start or int(rows[-1]["global_step"]) != expected_end:
    raise SystemExit(
        f"global-step boundary mismatch: {rows[0]['global_step']}..{rows[-1]['global_step']} "
        f"!= {expected_start}..{expected_end}"
    )
if len(step_rows) != expected_steps or step_globals != list(range(expected_start, expected_end)):
    raise SystemExit(
        f"optimizer-step sequence mismatch: count={len(step_rows)} globals={step_globals[:3]}...{step_globals[-3:]}"
    )
print(
    f"learning-rate audit passed: optimizer_steps={len(step_rows)} "
    f"global_steps={expected_start}..{expected_end}"
)
PY
  resume_checkpoint="${epoch_dir}/checkpoint-${expected_end_step}"
  if [[ ! -d "${resume_checkpoint}" ]]; then
    echo "Epoch ${tag} did not produce expected checkpoint-${expected_end_step}." >&2
    exit 1
  fi
done

if [[ "${RUN_MODE}" == "dry-run" ]]; then
  echo "Dry-run plan written to ${PLAN_PATH}"
else
  fixed_validation_checkpoint="${BASE_OUTPUT_DIR}/epoch_$(printf '%02d' "${FIXED_VALIDATION_EPOCH}")/checkpoint-${FIXED_VALIDATION_CHECKPOINT_SUFFIX}"
  if [[ ! -d "${fixed_validation_checkpoint}" ]]; then
    echo "Fixed epoch-3 validation checkpoint is missing: ${fixed_validation_checkpoint}" >&2
    exit 1
  fi
  echo "Five-epoch resumed training completed. Final training checkpoint: ${resume_checkpoint}"
  echo "Fixed Agent validation checkpoint: ${fixed_validation_checkpoint}"
fi
