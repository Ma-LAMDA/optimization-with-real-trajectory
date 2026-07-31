#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BRANCH="${BRANCH:-2026-07-31-sft}"
SYNC_REPO="${SYNC_REPO:-1}"
TRAIN_ENV="${TRAIN_ENV:-/root/autodl-tmp/envs/qwen36-sft}"
VLLM_ENV="${VLLM_ENV:-/root/autodl-tmp/qwen3.6-27b/.venv}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-07-31}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_reasoning_decision_validation.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/sft/manifest.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/output/qwen36-27b-lora-0731-${RUN_ID}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${RUN_ROOT}/train}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/validation_eval}"
WORKFLOW_LOG="${WORKFLOW_LOG:-${RUN_ROOT}/workflow.log}"
VLLM_LOG="${VLLM_LOG:-${RUN_ROOT}/vllm.log}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b-lora-0731}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EARLY_STOP_INTERVAL="${EARLY_STOP_INTERVAL:-3}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
REUSE_COMPLETED_TRAINING="${REUSE_COMPLETED_TRAINING:-0}"
TRAINING_GIT_COMMIT="${TRAINING_GIT_COMMIT:-}"
VALIDATION_REPEATS="${VALIDATION_REPEATS:-1}"

TRAIN_PYTHON="${TRAIN_ENV}/bin/python"
TRAIN_SWIFT="${TRAIN_ENV}/bin/swift"
VLLM_PYTHON="${VLLM_ENV}/bin/python"
VLLM_BIN="${VLLM_ENV}/bin/vllm"

mkdir -p "${RUN_ROOT}"
touch "${WORKFLOW_LOG}"
exec > >(tee -a "${WORKFLOW_LOG}") 2>&1

echo "[$(date -Iseconds)] Starting Qwen3.6-27B LoRA workflow ${RUN_ID}"
cd "${REPO_ROOT}"

if [[ "${SYNC_REPO}" == "1" ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked repository changes exist; refusing to update the branch." >&2
    exit 1
  fi
  git fetch origin "${BRANCH}"
  if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git switch "${BRANCH}"
  else
    git switch --track -c "${BRANCH}" "origin/${BRANCH}"
  fi
  git pull --ff-only origin "${BRANCH}"
fi

GIT_COMMIT="$(git rev-parse HEAD)"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${BRANCH}" ]]; then
  echo "Expected branch ${BRANCH}, found ${CURRENT_BRANCH}." >&2
  exit 1
fi
if [[ ! "${VALIDATION_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VALIDATION_REPEATS must be a positive integer." >&2
  exit 1
fi

for executable in \
  "${TRAIN_PYTHON}" \
  "${TRAIN_SWIFT}" \
  "${VLLM_PYTHON}" \
  "${VLLM_BIN}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Required executable is missing: ${executable}" >&2
    exit 1
  fi
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory is missing: ${MODEL_PATH}" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Regenerating the grouped train/validation split"
"${TRAIN_PYTHON}" scripts/convert_100x10_accepted_to_sft.py \
  --output-root "${DATA_ROOT}"
"${TRAIN_PYTHON}" scripts/validate_100x10_sft.py \
  --data-root "${DATA_ROOT}"
CURRENT_MANIFEST_SHA="$(
  "${TRAIN_PYTHON}" - "${MANIFEST_PATH}" <<'PY'
import hashlib
import pathlib
import sys

content = pathlib.Path(sys.argv[1]).read_bytes().replace(b"\r\n", b"\n")
print(hashlib.sha256(content).hexdigest())
PY
)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Data regeneration changed tracked repository files." >&2
  git status --short --untracked-files=no
  exit 1
fi

echo "[$(date -Iseconds)] Recording software and GPU state"
"${TRAIN_PYTHON}" - <<'PY'
import importlib.metadata as metadata
for name in (
    "ms-swift",
    "torch",
    "transformers",
    "peft",
    "flash-linear-attention",
    "triton",
):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=missing")
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

ACTIVE_GPU_PROCESSES="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${ACTIVE_GPU_PROCESSES}" ]]; then
  echo "GPU compute processes already exist; refusing to start training:" >&2
  echo "${ACTIVE_GPU_PROCESSES}" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Starting single-GPU LoRA SFT"
TRAINING_COMMIT_FILE="${TRAIN_OUTPUT_DIR}/training_git_commit.txt"
TRAINING_MANIFEST_SHA_FILE="${TRAIN_OUTPUT_DIR}/training_manifest_sha256.txt"
if [[ "${REUSE_COMPLETED_TRAINING}" == "1" ]] \
  && find "${TRAIN_OUTPUT_DIR}" -type f -name trainer_state.json -print -quit \
    | grep -q .; then
  echo "[$(date -Iseconds)] Reusing completed training state in ${TRAIN_OUTPUT_DIR}"
  if [[ -z "${TRAINING_GIT_COMMIT}" && -s "${TRAINING_COMMIT_FILE}" ]]; then
    TRAINING_GIT_COMMIT="$(tr -d '[:space:]' <"${TRAINING_COMMIT_FILE}")"
  fi
  if [[ -z "${TRAINING_GIT_COMMIT}" ]]; then
    echo "Reused training has no source commit. Set TRAINING_GIT_COMMIT once." >&2
    exit 1
  fi
  if [[ ! -s "${TRAINING_MANIFEST_SHA_FILE}" ]]; then
    echo "Reused training has no manifest hash; refusing cross-split reuse." >&2
    exit 1
  fi
  TRAINING_MANIFEST_SHA="$(
    tr -d '[:space:]' <"${TRAINING_MANIFEST_SHA_FILE}"
  )"
  if [[ "${TRAINING_MANIFEST_SHA}" != "${CURRENT_MANIFEST_SHA}" ]]; then
    echo "Training manifest differs from the current data split." >&2
    exit 1
  fi
else
  TRAINING_GIT_COMMIT="${GIT_COMMIT}"
  mkdir -p "${TRAIN_OUTPUT_DIR}"
  printf '%s\n' "${TRAINING_GIT_COMMIT}" >"${TRAINING_COMMIT_FILE}"
  printf '%s\n' "${CURRENT_MANIFEST_SHA}" >"${TRAINING_MANIFEST_SHA_FILE}"
  MODEL_PATH="${MODEL_PATH}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
  PYTHON_BIN="${TRAIN_PYTHON}" \
  SWIFT_BIN="${TRAIN_SWIFT}" \
  CUDA_VISIBLE_DEVICES=0 \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
  EVAL_STEPS="${EVAL_STEPS}" \
  EARLY_STOP_INTERVAL="${EARLY_STOP_INTERVAL}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  bash scripts/train_qwen36_lora_early_stop.sh
fi

TRAINING_SUMMARY="${RUN_ROOT}/training_summary.json"
"${TRAIN_PYTHON}" scripts/summarize_sft_training.py \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  --result "${TRAINING_SUMMARY}" \
  --git-commit "${TRAINING_GIT_COMMIT}"
BEST_CHECKPOINT="$(
  "${TRAIN_PYTHON}" scripts/summarize_sft_training.py \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --result "${TRAINING_SUMMARY}" \
    --git-commit "${TRAINING_GIT_COMMIT}" \
    --print-best-only
)"
echo "[$(date -Iseconds)] Best checkpoint: ${BEST_CHECKPOINT}"

VLLM_PID=""
cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[$(date -Iseconds)] Stopping vLLM process ${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT INT TERM

echo "[$(date -Iseconds)] Starting one TP=2 vLLM instance"
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0,1 \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs 2 \
  --enable-lora \
  --max-lora-rank 8 \
  --lora-modules "${SERVED_MODEL_NAME}=${BEST_CHECKPOINT}" \
  --served-model-name qwen36-27b-base \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

API_MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
READY_DEADLINE=$((SECONDS + 1800))
while true; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  if "${VLLM_PYTHON}" - "${API_MODELS_URL}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request

url, expected = sys.argv[1:3]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    names = {item.get("id") for item in payload.get("data", [])}
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if expected in names else 1)
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

echo "[$(date -Iseconds)] Evaluating ${VALIDATION_REPEATS} time(s) with two workers and concurrency two"
if [[ "${VALIDATION_REPEATS}" == "1" ]]; then
  "${TRAIN_PYTHON}" scripts/evaluate_sft_validation.py \
    --dataset "${VALIDATION_DATASET_PATH}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --model "${SERVED_MODEL_NAME}" \
    --base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --max-tokens 8000 \
    --instance-count 1 \
    --workers 2 \
    --request-concurrency 2 \
    --git-commit "${GIT_COMMIT}" \
    --checkpoint "${BEST_CHECKPOINT}"
else
  for repeat_index in $(seq 1 "${VALIDATION_REPEATS}"); do
    repeat_name="$(printf 'repeat_%02d' "${repeat_index}")"
    echo "[$(date -Iseconds)] Validation ${repeat_index}/${VALIDATION_REPEATS}"
    "${TRAIN_PYTHON}" scripts/evaluate_sft_validation.py \
      --dataset "${VALIDATION_DATASET_PATH}" \
      --output-dir "${EVAL_OUTPUT_DIR}/${repeat_name}" \
      --model "${SERVED_MODEL_NAME}" \
      --base-url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
      --max-tokens 8000 \
      --instance-count 1 \
      --workers 2 \
      --request-concurrency 2 \
      --git-commit "${GIT_COMMIT}" \
      --checkpoint "${BEST_CHECKPOINT}"
  done
  "${TRAIN_PYTHON}" scripts/summarize_repeated_validation.py \
    --input-root "${EVAL_OUTPUT_DIR}" \
    --repeats "${VALIDATION_REPEATS}" \
    --output "${EVAL_OUTPUT_DIR}/validation_summary.json"
fi

cleanup_vllm
VLLM_PID=""
trap - EXIT INT TERM

WORKFLOW_SUMMARY="${RUN_ROOT}/workflow_summary.json"
"${TRAIN_PYTHON}" scripts/finalize_lora_workflow.py \
  --training-summary "${TRAINING_SUMMARY}" \
  --validation-summary "${EVAL_OUTPUT_DIR}/validation_summary.json" \
  --manifest "${MANIFEST_PATH}" \
  --output "${WORKFLOW_SUMMARY}" \
  --run-id "${RUN_ID}" \
  --branch "${BRANCH}" \
  --git-commit "${GIT_COMMIT}" \
  --model-path "${MODEL_PATH}"

echo "[$(date -Iseconds)] Workflow completed: ${WORKFLOW_SUMMARY}"
