#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_ENV="${TRAIN_ENV:-/root/autodl-tmp/envs/qwen36-sft}"
VLLM_ENV="${VLLM_ENV:-/root/autodl-tmp/qwen3.6-27b/.venv}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${REPO_ROOT}/data/2026-07-31/sft/qwen3_6_27b_reasoning_decision_validation.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/qwen36-27b-validation-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
VLLM_LOG="${VLLM_LOG:-${OUTPUT_ROOT}/vllm.log}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
INCLUDE_BASE="${INCLUDE_BASE:-1}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-qwen36-27b-base}"
BASE_REPEATS="${BASE_REPEATS:-5}"
LORA_REPEATS="${LORA_REPEATS:-1}"
LORA_TARGETS="${LORA_TARGETS:-}"

TRAIN_PYTHON="${TRAIN_ENV}/bin/python"
VLLM_PYTHON="${VLLM_ENV}/bin/python"
VLLM_BIN="${VLLM_ENV}/bin/vllm"

if [[ "${INCLUDE_BASE}" != "0" && "${INCLUDE_BASE}" != "1" ]]; then
  echo "INCLUDE_BASE must be 0 or 1." >&2
  exit 1
fi
for repeat_count in "${BASE_REPEATS}" "${LORA_REPEATS}"; do
  if [[ ! "${repeat_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Repeat counts must be positive integers." >&2
    exit 1
  fi
done
if [[ ! -x "${TRAIN_PYTHON}" || ! -x "${VLLM_PYTHON}" || ! -x "${VLLM_BIN}" ]]; then
  echo "The configured training or vLLM environment is incomplete." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" || ! -f "${VALIDATION_DATASET_PATH}" ]]; then
  echo "The base model or validation dataset is missing." >&2
  exit 1
fi

declare -a TARGET_NAMES=()
declare -a TARGET_PATHS=()
if [[ -n "${LORA_TARGETS}" ]]; then
  read -r -a target_specs <<<"${LORA_TARGETS}"
  for target_spec in "${target_specs[@]}"; do
    target_name="${target_spec%%=*}"
    target_path="${target_spec#*=}"
    if [[ "${target_name}" == "${target_spec}" ]] \
      || [[ ! "${target_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "Invalid LoRA target: ${target_spec}" >&2
      exit 1
    fi
    if [[ ! -d "${target_path}" ]]; then
      echo "LoRA checkpoint is missing: ${target_path}" >&2
      exit 1
    fi
    TARGET_NAMES+=("${target_name}")
    TARGET_PATHS+=("${target_path}")
  done
fi
if [[ "${INCLUDE_BASE}" == "0" && "${#TARGET_NAMES[@]}" -eq 0 ]]; then
  echo "At least one evaluation target is required." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
GIT_COMMIT="$(git rev-parse HEAD)"
"${TRAIN_PYTHON}" scripts/validate_100x10_sft.py \
  --data-root "${REPO_ROOT}/data/2026-07-31"

ACTIVE_GPU_PROCESSES="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${ACTIVE_GPU_PROCESSES}" ]]; then
  echo "GPU compute processes already exist; refusing to start evaluation:" >&2
  echo "${ACTIVE_GPU_PROCESSES}" >&2
  exit 1
fi

declare -a vllm_args=(
  serve "${MODEL_PATH}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --dtype bfloat16
  --tensor-parallel-size 2
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --max-num-seqs 2
  --served-model-name "${BASE_MODEL_NAME}"
)
if [[ "${#TARGET_NAMES[@]}" -gt 0 ]]; then
  vllm_args+=(
    --enable-lora
    --max-lora-rank 8
    --max-loras 1
    --max-cpu-loras "${#TARGET_NAMES[@]}"
    --lora-modules
  )
  for target_index in "${!TARGET_NAMES[@]}"; do
    vllm_args+=(
      "${TARGET_NAMES[${target_index}]}=${TARGET_PATHS[${target_index}]}"
    )
  done
fi

VLLM_PID=""
cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT INT TERM

echo "[$(date -Iseconds)] Starting one TP=2 vLLM instance"
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0,1 \
  "${VLLM_BIN}" "${vllm_args[@]}" >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

API_MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
READY_DEADLINE=$((SECONDS + 1800))
while true; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  if "${VLLM_PYTHON}" - "${API_MODELS_URL}" \
    "${BASE_MODEL_NAME}" "${TARGET_NAMES[@]}" <<'PY'
import json
import sys
import urllib.request

url, *expected = sys.argv[1:]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = {item.get("id") for item in payload.get("data", [])}
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if set(expected) <= names else 1)
PY
  then
    break
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "vLLM did not become ready within 30 minutes." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 5
done

evaluate_target() {
  local target_name="$1"
  local checkpoint="$2"
  local repeat_count="$3"
  local target_root="${OUTPUT_ROOT}/${target_name}"
  local repeat_index
  local repeat_name

  echo "[$(date -Iseconds)] Evaluating ${target_name} ${repeat_count} time(s)"
  if [[ "${repeat_count}" == "1" ]]; then
    "${TRAIN_PYTHON}" scripts/evaluate_sft_validation.py \
      --dataset "${VALIDATION_DATASET_PATH}" \
      --output-dir "${target_root}" \
      --model "${target_name}" \
      --base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
      --max-tokens 8000 \
      --instance-count 1 \
      --workers 2 \
      --request-concurrency 2 \
      --git-commit "${GIT_COMMIT}" \
      --checkpoint "${checkpoint}"
    return
  fi

  for repeat_index in $(seq 1 "${repeat_count}"); do
    repeat_name="$(printf 'repeat_%02d' "${repeat_index}")"
    "${TRAIN_PYTHON}" scripts/evaluate_sft_validation.py \
      --dataset "${VALIDATION_DATASET_PATH}" \
      --output-dir "${target_root}/${repeat_name}" \
      --model "${target_name}" \
      --base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
      --max-tokens 8000 \
      --instance-count 1 \
      --workers 2 \
      --request-concurrency 2 \
      --git-commit "${GIT_COMMIT}" \
      --checkpoint "${checkpoint}"
  done
  "${TRAIN_PYTHON}" scripts/summarize_repeated_validation.py \
    --input-root "${target_root}" \
    --repeats "${repeat_count}" \
    --output "${target_root}/validation_summary.json"
}

if [[ "${INCLUDE_BASE}" == "1" ]]; then
  evaluate_target "${BASE_MODEL_NAME}" "${MODEL_PATH}" "${BASE_REPEATS}"
fi
for target_index in "${!TARGET_NAMES[@]}"; do
  evaluate_target \
    "${TARGET_NAMES[${target_index}]}" \
    "${TARGET_PATHS[${target_index}]}" \
    "${LORA_REPEATS}"
done

"${TRAIN_PYTHON}" scripts/summarize_validation_sweep.py \
  --input-root "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/validation_sweep_summary.json"

cleanup_vllm
VLLM_PID=""
trap - EXIT INT TERM
echo "[$(date -Iseconds)] Validation sweep completed: ${OUTPUT_ROOT}"
